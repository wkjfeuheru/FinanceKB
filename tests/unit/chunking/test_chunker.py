import re
from pathlib import Path

from src.core.config import DOCLING_CHUNK_MAX_TOKENS, EMBEDDING_MAX_TOKENS
from src.chunking.chunker import (
    _CHARS_PER_TOKEN,
    HierarchicalChunker,
    _classify_heading_line,
    _inject_heading_structure,
)


def _workdir(name: str) -> Path:
    """仓库内的工作目录（受限环境下系统临时目录可能不可写）。"""
    path = Path(__file__).resolve().parents[1] / ".tmp" / "chunker" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parents_follow_complete_heading_sections():
    chunker = HierarchicalChunker(max_tokens=512)

    parents = chunker._split_into_parents(
        "# 第一章\n第一章正文。\n\n## 第一节\n第一节正文。\n\n# 第二章\n第二章正文。",
        "doc.md",
        "doc",
    )

    assert [parent.heading for parent in parents] == ["第一章", "第一节", "第二章"]
    assert parents[0].content == "# 第一章\n第一章正文。"
    assert parents[1].content == "## 第一节\n第一节正文。"
    assert parents[2].content == "# 第二章\n第二章正文。"


def test_unheaded_text_uses_natural_paragraphs_without_length_split():
    chunker = HierarchicalChunker(max_tokens=512)
    text = "第一段第一行。\n第一段第二行。\n\n第二段内容。"

    parents = chunker._split_into_parents(text, "doc.txt", "doc")

    assert [parent.content for parent in parents] == [
        "第一段第一行。\n第一段第二行。",
        "第二段内容。",
    ]


def test_child_char_budget_fits_embedding_window():
    """子块上限必须满足两条：不超嵌入窗口，且明显小于窗口（否则等于没在管粒度）。

    实测（bge-m3 分词器 / 18 篇真实研报）：最坏 token 密度 0.707 token/字符。
    历史值「×4 = 2048 字符 ≈ 1370 token」远超 512 窗口，导致 12.2% 的子块尾部
    进不了稠密向量；而把上限放到和窗口齐平（512 token）时，597 个父块里只有
    23% 被切开——上限形同虚设。故两条约束一起锁。
    """
    cap_chars = DOCLING_CHUNK_MAX_TOKENS * _CHARS_PER_TOKEN
    worst_case_tokens = cap_chars * 0.707
    assert worst_case_tokens <= EMBEDDING_MAX_TOKENS, (
        f"子块最坏情况 {worst_case_tokens:.0f} token 超过嵌入窗口 {EMBEDDING_MAX_TOKENS} token"
    )
    assert DOCLING_CHUNK_MAX_TOKENS <= EMBEDDING_MAX_TOKENS * 0.75, (
        f"子块上限 {DOCLING_CHUNK_MAX_TOKENS} token 太接近嵌入窗口 "
        f"{EMBEDDING_MAX_TOKENS} token——上限太大就等于没在管粒度"
    )


def test_long_complete_paragraph_remains_one_parent_but_children_split():
    chunker = HierarchicalChunker(max_tokens=10, overlap=0)
    paragraph = "。".join(["这是一个很长的自然段"] * 20) + "。"

    parents = chunker._split_into_parents(paragraph, "doc.txt", "doc")
    children = chunker._split_parent_into_children(parents[0].content)

    assert len(parents) == 1
    assert parents[0].content == paragraph
    assert len(children) > 1
    assert "这是一个很长的自然段" in "".join(children)


def test_tables_keep_full_parent_and_index_summary_in_child_processing():
    chunker = HierarchicalChunker(max_tokens=4, overlap=0)
    table = "| 项目 | 数值 |\n| --- | --- |\n| 收入 | 100 |\n| 支出 | 50 |"

    parent_content = f"表格前文。\n\n{table}\n\n表格后文。"
    children = chunker._split_parent_into_children(parent_content)
    joined = "\n".join(children)

    assert table not in children
    assert "表头：项目 | 数值" in joined
    assert "行标签：收入、支出" in joined
    assert "100" not in joined                 # 数据数字不进索引


def test_tables_with_equal_index_summaries_have_distinct_child_ids():
    markdown = (
        "# 财务数据\n\n"
        "| 项目 | 数值 |\n| --- | --- |\n| 收入 | 100 |\n\n"
        "| 项目 | 数值 |\n| --- | --- |\n| 收入 | 200 |\n"
    )
    chunker = HierarchicalChunker()
    result = chunker.chunk_markdown(markdown, source="report.pdf")
    first, second = [chunk for chunk in result.chunks if chunk.metadata["block_type"] == "table"]

    assert first.page_content == second.page_content
    assert first.metadata["content_hash"] == second.metadata["content_hash"]
    assert first.metadata["id"] != second.metadata["id"]
    assert [first.metadata["parent_id"], second.metadata["parent_id"]] == [
        table.id for table in result.tables
    ]
    assert [table.payload[-1][-1] for table in result.tables] == ["100", "200"]
    assert [chunk.metadata["id"] for chunk in result.chunks] == [
        chunk.metadata["id"] for chunk in chunker.chunk_markdown(markdown, source="report.pdf").chunks
    ]


# --- 标题结构注入：中文公文/法规无 Markdown 标题时的结构骨架 ---


def test_inject_heading_structure_normalizes_levels_per_document():
    """章/节/条/一、/（一）按由外到内依次占 ##..######，互不撞级。"""
    text = (
        "第一章 总则\n"
        "为规范公司行为，制定本办法。\n"
        "第一节 适用范围\n"
        "本办法适用于全体员工。\n"
        "第一条 目的\n"
        "为加强合规管理。\n"
        "一、基本原则\n"
        "遵循审慎合规。\n"
        "（一）合法合规\n"
        "不得违反法律法规。\n"
    )

    lines = _inject_heading_structure(text).split("\n")

    assert lines[0] == "## 第一章 总则"
    assert lines[2] == "### 第一节 适用范围"
    assert lines[4] == "#### 第一条 目的"
    # 「一、」与「第X节」不再撞同一级（旧实现两者都是 ###）
    assert lines[6] == "##### 一、基本原则"
    assert lines[8] == "###### （一）合法合规"
    # 正文行不受影响
    assert lines[1] == "为规范公司行为，制定本办法。"


def test_inject_heading_structure_maps_public_document_without_chapters():
    """只有 一、/（一）/1. 的公文体从 ## 起算，不会悬空到深层级。"""
    result = _inject_heading_structure(
        "一、总体要求\n（一）指导思想\n1. 坚持稳健经营\n二、重点任务"
    )

    assert result.split("\n") == [
        "## 一、总体要求",
        "### （一）指导思想",
        "#### 1. 坚持稳健经营",
        "## 二、重点任务",
    ]


def test_inject_heading_structure_covers_extended_markers():
    """编/款/项、半角括号、1.1 多级编号均可识别。"""
    result = _inject_heading_structure(
        "第一编 总则\n第二章 机构\n第三款 职责\n(一) 董事会\n1.1 具体措施"
    )

    assert result.split("\n") == [
        "## 第一编 总则",
        "### 第二章 机构",
        "#### 第三款 职责",
        "##### (一) 董事会",
        "###### 1.1 具体措施",
    ]


def test_inject_heading_structure_ignores_inline_numbers():
    """正文里的数字/编号不应被误判为标题。"""
    text = "3 名员工到岗。\n2023年收入增长。\n1.5倍于去年。"

    assert _inject_heading_structure(text) == text


def test_inject_heading_structure_supports_arabic_and_variant_numbers():
    result = _inject_heading_structure("第1条 适用范围\n第两百条 附则\n综述")

    assert result.split("\n") == [
        "### 第1条 适用范围",
        "### 第两百条 附则",
        "## 综述",
    ]


def test_inject_heading_structure_is_idempotent_and_keeps_existing_headings():
    text = "# 合规管理办法\n\n第一章 总则\n第一条 目的\n"

    once = _inject_heading_structure(text)

    assert "# 合规管理办法" in once
    assert "## 第一章 总则" in once
    assert "### 第一条 目的" in once
    assert _inject_heading_structure(once) == once


def test_inject_heading_structure_skips_fenced_code_blocks():
    text = "正文。\n```\n第一章 总则\n```\n第一章 总则"

    assert _inject_heading_structure(text).split("\n") == [
        "正文。",
        "```",
        "第一章 总则",
        "```",
        "## 第一章 总则",
    ]


def test_mixed_document_keeps_injected_hierarchy():
    """仅有一个 Markdown 标题的公文，注入后仍应切出章/条层级。"""
    chunker = HierarchicalChunker(max_tokens=512)
    markdown = "# 合规管理办法\n\n第一章 总则\n第一条 目的\n为加强管理。\n"

    chunks = chunker._build_chunks(markdown, "doc.md", "doc")

    headings = [parent.heading for parent in chunks.parents]
    assert "第一章 总则" in headings
    assert "第一条 目的" in headings


# --- 层级骨架：heading_path 与不同标记的层级隔离 ---


def test_distinct_marker_kinds_never_share_a_level():
    """撞车回归：同一文档里不同标记类型必须落在不同层级。"""
    text = "第一编 总则\n第一章 机构\n第一条 目的\n一、基本原则\n（一）合法合规\n"

    levels: dict[str, int] = {}
    for line in _inject_heading_structure(text).split("\n"):
        match = re.match(r"^(#+)\s+(.*)$", line)
        if not match:
            continue
        marker = _classify_heading_line(match.group(2))
        if marker:
            levels[marker] = len(match.group(1))

    assert levels == {
        "bian": 2,
        "zhang": 3,
        "tiao": 4,
        "cn_item": 5,
        "paren_cn": 6,
    }


def test_marker_kinds_beyond_available_levels_clamp_at_six():
    """标记类型多于 ##..###### 五级时在 ###### 收敛，内容不丢失。"""
    result = _inject_heading_structure(
        "第一编 总则\n第一章 机构\n第一节 范围\n第一条 目的\n一、原则\n（一）合规"
    )

    lines = result.split("\n")
    assert lines[0] == "## 第一编 总则"
    assert lines[-1] == "###### （一）合规"
    assert len(lines) == 6


def test_heading_path_tracks_nesting_and_resets_on_sibling():
    chunker = HierarchicalChunker(max_tokens=512)
    markdown = "# 标题\n## 第一章 总则\n### 第一节 范围\n#### 第一条 目的\n## 第二章 风险管理\n"

    parents = chunker._split_into_parents(markdown, "doc.md", "doc")

    assert [parent.heading_path for parent in parents] == [
        "标题",
        "标题 > 第一章 总则",
        "标题 > 第一章 总则 > 第一节 范围",
        "标题 > 第一章 总则 > 第一节 范围 > 第一条 目的",
        # 同级标题把栈中更深的祖先弹出
        "标题 > 第二章 风险管理",
    ]


def test_child_chunks_carry_heading_path():
    chunker = HierarchicalChunker(max_tokens=512)
    markdown = "第一章 总则\n第一条 目的\n为加强管理。\n"

    chunks = chunker._build_chunks(markdown, "doc.md", "doc")

    paths = {chunk.metadata["heading_path"] for chunk in chunks.chunks}
    assert "第一章 总则" in paths
    assert "第一章 总则 > 第一条 目的" in paths


# --- 研报结构化：表格双份、页码归属 ---------------------------------------

_TABLE_MARKDOWN = (
    "| 年份 | 营收 | 净利 |\n"
    "| --- | --- | --- |\n"
    "| 2026E | 100 | 12 |\n"
    "| 2027E | 130 | 16 |\n"
    "| 2028E | 160 | 20 |\n"
)


def test_table_child_is_atomic_and_full_table_kept_separately():
    """向量库只留「表头 + 摘要」作索引，整表另存；表格不与相邻文本合并。"""
    chunker = HierarchicalChunker(max_tokens=512)
    markdown = (
        f"## 盈利预测\n\n前文说明。\n\n{_TABLE_MARKDOWN}\n后文说明。\n"
    )

    result = chunker.chunk_markdown(markdown, source="s.pdf", title="t")

    table_children = [c for c in result.chunks if c.metadata["block_type"] == "table"]
    assert len(table_children) == 1
    index_text = table_children[0].page_content
    assert "盈利预测" in index_text          # 章节路径（heading_path）进索引
    assert "表头：年份 | 营收 | 净利" in index_text
    assert "2026E" in index_text             # 行标签预览覆盖表体指标
    assert "2028E" in index_text             # 不止首行——深处的行标签也在
    assert "100" not in index_text           # 数据数字不进索引
    assert "前文说明" not in index_text       # 且不与相邻文本合并
    assert "后文说明" not in index_text

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.row_count == 3
    assert table.payload == [
        ["年份", "营收", "净利"],
        ["2026E", "100", "12"],
        ["2027E", "130", "16"],
        ["2028E", "160", "20"],
    ]
    assert table.markdown == _TABLE_MARKDOWN.strip()
    assert table.source == "s.pdf"
    assert table.heading_path == "盈利预测"
    # 索引块指向整表，检索侧据此做独立展开
    assert table_children[0].metadata["parent_id"] == table.id


def test_text_children_keep_parent_linkage_and_text_block_type():
    chunker = HierarchicalChunker(max_tokens=512)
    result = chunker.chunk_markdown("## 投资要点\n\n燃机订单超预期。\n", source="s.pdf")

    child = result.chunks[0]
    assert child.metadata["block_type"] == "text"
    assert child.metadata["parent_id"] == result.parents[0].id


def test_child_start_page_uses_printed_page_from_content_list():
    """content_list 物理第 3 页（page_idx=2）映射后切块记印刷第 1 页。"""
    from src.ingest.mineru_parser import _blocks_from_content_list

    payload = [
        {
            "type": "text",
            "text": "燃机订单超预期，AIDC 一体化打开成长空间。",
            "page_idx": 2,
        }
    ]
    blocks = _blocks_from_content_list(payload, page_map=(0, 0, 1, 2, 3))
    chunker = HierarchicalChunker(max_tokens=512)
    result = chunker.chunk_markdown(
        "## 投资要点\n\n燃机订单超预期，AIDC 一体化打开成长空间。\n",
        source="s.pdf",
        blocks=blocks,
    )

    assert [block.page for block in blocks] == [1]
    assert [chunk.metadata["start_page"] for chunk in result.chunks] == [1]


def test_child_pages_follow_blocks_and_stay_zero_when_unmatched():
    """页码来自 content_list 的分页文本块；匹配不上留 0，绝不猜。"""
    from src.ingest.pdf_assets import ContentBlock

    chunker = HierarchicalChunker(max_tokens=512)
    markdown = "## 投资要点\n\n燃机订单超预期，AIDC 一体化打开成长空间。\n"

    matched = chunker.chunk_markdown(
        markdown,
        source="s.pdf",
        blocks=(ContentBlock(text="## 投资要点 燃机订单超预期，AIDC 一体化打开成长空间。", page=3),),
    )
    assert [c.metadata["start_page"] for c in matched.chunks] == [3]
    assert [c.metadata["end_page"] for c in matched.chunks] == [3]

    unmatched = chunker.chunk_markdown(
        markdown,
        source="s.pdf",
        blocks=(ContentBlock(text="完全无关的另一篇文档内容。", page=9),),
    )
    assert [c.metadata["start_page"] for c in unmatched.chunks] == [0]
    assert [c.metadata["end_page"] for c in unmatched.chunks] == [0]


def test_child_spanning_two_pages_widens_page_range():
    from src.ingest.pdf_assets import ContentBlock

    chunker = HierarchicalChunker(max_tokens=512)
    markdown = "## 产能\n\n第一页讲产能利用率提升。\n\n第二页讲新增产线投产节奏。\n"

    result = chunker.chunk_markdown(
        markdown,
        source="s.pdf",
        blocks=(
            ContentBlock(text="## 产能 第一页讲产能利用率提升。", page=3),
            ContentBlock(text="第二页讲新增产线投产节奏。", page=4),
        ),
    )

    chunk = result.chunks[0]
    assert (chunk.metadata["start_page"], chunk.metadata["end_page"]) == (3, 4)


def test_parse_and_chunk_carries_blocks_from_parser(monkeypatch):
    """解析层产出的 blocks 必须一路传到切块结果，否则页码恒为 0。

    这里自己写一份临时 md 当输入，不依赖仓库里的样本语料——样本随时可能被
    清理，测试不该因此变成红条。
    """
    from src.ingest import pdf_assets
    from src.ingest.mineru_parser import ParseResult

    source = _workdir("parser-blocks") / "note.md"
    source.write_text("## 投资要点\n\n燃机订单超预期。\n", encoding="utf-8")

    block = pdf_assets.ContentBlock(text="## 投资要点 燃机订单超预期。", page=7)
    monkeypatch.setattr(
        "src.ingest.get_parser",
        lambda: type(
            "_P",
            (),
            {"parse": lambda self, path, source="", title="": ParseResult(
                markdown="## 投资要点\n\n燃机订单超预期。\n", blocks=(block,)
            )},
        )(),
    )

    result = HierarchicalChunker().parse_and_chunk(source)

    assert result.chunks[0].metadata["start_page"] == 7


# --- 表格索引文本：表头 + 摘要（描述表展示了什么内容） -----------------------

def test_table_index_text_caps_label_preview():
    """行标签预览有界：条数封顶，语义信号（科目名）保留，数据数字不进。"""
    from src.chunking.chunker import (
        _TABLE_LABEL_PREVIEW_COUNT,
        _table_index_text,
    )

    rows = "\n".join(f"| 科目{i} | {i} |" for i in range(50))
    table = f"| 指标 | 数值 |\n| --- | --- |\n{rows}"

    text = _table_index_text(table, "盈利预测 > 财务摘要")

    assert "章节：盈利预测 > 财务摘要" in text
    assert "表头：指标 | 数值" in text
    assert "科目0" in text
    assert f"科目{_TABLE_LABEL_PREVIEW_COUNT - 1}" in text
    assert f"科目{_TABLE_LABEL_PREVIEW_COUNT}" not in text   # 预览封顶


def test_table_index_text_without_heading_or_title_still_composes():
    """无表题、无章节的表：表头 + 行标签本身就是可用的摘要。"""
    from src.chunking.chunker import _table_index_text

    table = "| 指标 | 数值 |\n| --- | --- |\n| 收入 | 100 |"

    text = _table_index_text(table)

    assert text == "表头：指标 | 数值\n行标签：收入"


def test_parse_and_chunk_carries_figure_captions_from_parser(monkeypatch):
    """解析层抽取的原文图注必须一路传到切块结果，图片块组装才有语义来源。"""
    from src.ingest import pdf_assets
    from src.ingest.mineru_parser import ParseResult

    source = _workdir("parser-captions") / "note.md"
    source.write_text("## 投资要点\n\n燃机订单超预期。\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.ingest.get_parser",
        lambda: type(
            "_P",
            (),
            {"parse": lambda self, path, source="", title="": ParseResult(
                markdown="## 投资要点\n\n燃机订单超预期。\n",
                figure_captions={4: ["图12 燃机订单结构"]},
            )},
        )(),
    )

    result = HierarchicalChunker().parse_and_chunk(source)

    assert result.figure_captions == {4: ["图12 燃机订单结构"]}
