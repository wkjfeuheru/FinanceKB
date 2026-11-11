"""基于层级父子结构的文档解析与切块."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from src.chunking.recursive_splitter import RecursiveCharacterTextSplitter
from src.chunking.html_table import restore_html_table

from src.core.config import (
    CHILD_OVERLAP_TOKENS,
    DOCLING_CHUNK_MAX_TOKENS,
    ENABLE_SEMANTIC_CHUNKER,
    SEMANTIC_BREAK_THRESHOLD,
    SEMANTIC_DEVICE,
    SEMANTIC_EMBED_MODEL,
    SEMANTIC_MIN_TOKENS,
)
from src.ingest.pdf_assets import ContentBlock, ImageAsset

# 子块预算的 token → 字符换算系数。
#
# 实测（bge-m3 分词器，18 篇真实研报 MinerU 产物）：中文财经语料均值 1 token ≈ 1.5
# 字符，而数字/英文密集的块最坏密度 ≈ 0.707 token/字符 → EMBEDDING_MAX_TOKENS(512)
# 的窗口最多只容得下约 724 字符。取 1.4 折算，使 DOCLING_CHUNK_MAX_TOKENS 真正等于
# 「块内 token 上限」：512 × 1.4 = 716 字符（最坏密度下 ≈506 token，仍在窗口内）。
# 历史值是 ×4（512 → 2048 字符 ≈ 1370 token），导致 12.2% 的块尾部被嵌入截断、
# 只有 BM25 能看到；回归守卫见 tests/unit/chunking/test_chunker.py。
_CHARS_PER_TOKEN = 1.4


@dataclass
class ParentChunk:
    """父块（按完整 Markdown 章节或自然段切分）.

    Attributes:
        heading: 本父块的叶子标题（不含上级路径）。
        heading_path: 从顶层到本父块的标题路径，形如
                      ``第一章 总则 > 第一节 适用范围 > 第一条 目的``；
                      无标题结构时为空串。
    """

    id: str
    heading: str
    content: str
    source: str
    title: str
    heading_path: str = ""


@dataclass
class BlockPiece:
    """父块内的一个原子单元。

    Attributes:
        kind: ``text`` 或 ``table``。
        text: 进入向量索引的文本；表格这里是「表头 + 摘要」索引文本
              （:func:`_table_index_text`，``_build_chunks`` 会带章节路径重组）。
        full_text: 表格的完整 Markdown（仅 ``kind == "table"``），进 PostgreSQL。
    """

    kind: str
    text: str
    full_text: str = ""


@dataclass
class TableChunk:
    """整表：既进 PostgreSQL（结构化 JSONB + 原始 Markdown），也供检索侧展开。

    向量库只存 ``BlockPiece.text``（表头 + 摘要）作索引，整表靠 ``id`` 关联，
    因此表体不会被 3000 字符的父块上限丢弃；输出拼上下文时才按 ``id``
    展开成完整表格。
    """

    id: str
    parent_id: str
    markdown: str
    payload: list[list[str]]
    row_count: int
    source: str
    heading_path: str = ""
    start_page: int = 0


@dataclass
class DoclingChunks:
    """层级父子切块结果.

    Attributes:
        chunks: 子块 Document 列表（每个子块带 page_content + metadata）。
        markdown: 仅 PDF 解析路径返回完整 markdown（扫描版 PDF 上传后复用）；
                  md/txt 直接切块路径为 None。
        parents: 父块列表（按完整 Markdown 章节或自然段切分）。
        images: 普通 PDF 解析时提取的嵌入图片元数据，形如
                ``{"temp_path", "temp_dir", "key", "page", "ext", "sha256", "caption"}``；
                扫描版 PDF / md / txt 路径为空列表。主进程负责落盘到存储后端。
        tables: 整表列表（进 PostgreSQL，不进向量库）。
        blocks: 解析层给出的分页文本块（``ContentBlock``）。流水线会重建
                ``chunks``，因此这里必须留存，否则页码归属在重切块后全部丢失。
        figure_captions: 解析层从 content_list 抽取的「页码 -> 原文图注列表」，
                供图片块组装索引文本（图注是图片语义召回的重要来源）。与
                ``images`` 一样**不进切块进程池载荷**，由流水线在池返回后重挂载。
    """

    chunks: list[Document] = field(default_factory=list)
    markdown: str | None = None
    parents: list[ParentChunk] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    tables: list[TableChunk] = field(default_factory=list)
    blocks: list[ContentBlock] = field(default_factory=list)
    figure_captions: dict[int, list[str]] = field(default_factory=dict)


class HierarchicalChunker:
    """将文档按完整章节/自然段切为父块，再递归切分子块."""

    def __init__(
        self,
        max_tokens: int = DOCLING_CHUNK_MAX_TOKENS,
        overlap: int = CHILD_OVERLAP_TOKENS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def parse_and_chunk(
        self,
        file_path: str | Path,
        *,
        source: str = "",
        title: str = "",
    ) -> DoclingChunks:
        """解析文件并返回层级父子切块结果。"""
        path = Path(file_path)
        source = source or path.name
        title = title or Path(source).stem

        from src.ingest import get_parser

        result = get_parser().parse(path, source=source, title=title)
        return self._build_chunks(
            result.markdown.strip(), source, title,
            is_pdf=path.suffix.lower() == ".pdf",
            images=result.images,
            blocks=result.blocks,
            figure_captions=getattr(result, "figure_captions", None) or {},
        )

    def chunk_markdown(
        self,
        markdown: str,
        *,
        source: str,
        title: str = "",
        is_pdf: bool = True,
        images: Sequence[ImageAsset] = (),
        blocks: Sequence[ContentBlock] = (),
        figure_captions: dict[int, list[str]] | None = None,
    ) -> DoclingChunks:
        """从 Markdown 文本直接切块（MinerU 解析结果复用此路径）。

        与 parse_and_chunk_from_text 的区别：保留 markdown（is_pdf=True 时），
        便于后续上传解析后的 .md 文件到对象存储。

        ``blocks`` 是解析层给出的分页文本块，用于把切块结果归属到页码；
        ``figure_captions`` 是解析层抽取的原文图注，供图片块组装索引文本。
        """
        if not markdown or not markdown.strip():
            raise ValueError(f"解析结果为空：{source}")
        return self._build_chunks(
            markdown.strip(), source, title,
            is_pdf=is_pdf, images=images, blocks=blocks,
            figure_captions=figure_captions,
        )

    def _build_chunks(
        self,
        markdown: str,
        source: str,
        title: str,
        *,
        is_pdf: bool = False,
        images: Sequence[ImageAsset] = (),
        blocks: Sequence[ContentBlock] = (),
        figure_captions: dict[int, list[str]] | None = None,
    ) -> DoclingChunks:
        """内部方法：对 markdown 文本进行层级切块。"""
        if not markdown:
            raise ValueError(f"解析结果为空：{source}")

        # 内容清洗：规则过滤（页码/版权/导航等噪声）+ 文档内段落级去重（固定流程）
        markdown = self._clean_markdown(markdown)
        if not markdown or not markdown.strip():
            raise ValueError(f"清洗后内容为空：{source}")

        # 中文公文/法规普遍没有 Markdown 标题：逐行识别章/节/条/一、/（一）并注入标题，
        # 为后续层级切块提供结构骨架。注入幂等（已有标题的行原样保留），
        # 因此「仅有一个标题的混合文档」也能补齐层级，而非整篇塌成单个父块。
        markdown = _inject_heading_structure(markdown)

        parents = self._split_into_parents(markdown, source, title)

        documents: list[Document] = []
        tables: list[TableChunk] = []
        for parent in parents:
            seen_content_hashes: set[str] = set()
            for piece in self._split_parent_into_blocks(parent.content):
                if piece.kind == "table":
                    # 表格索引文本在这里带章节路径重组：
                    # _split_parent_into_blocks 拿不到 parent，只能先给无章节版本
                    index_text = _table_index_text(piece.full_text, parent.heading_path)
                    if index_text:
                        piece.text = index_text
                index = len(documents)
                content_hash = hashlib.sha256(piece.text.strip().encode("utf-8")).hexdigest()
                identity = f"{parent.id}:{content_hash}"
                # Distinct tables can share a search summary. Keep the first occurrence's ID.
                if content_hash in seen_content_hashes:
                    identity = f"{identity}:{index}"
                seen_content_hashes.add(content_hash)
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                start_page, end_page = _resolve_pages(piece.text, blocks)
                if piece.kind == "table":
                    table = _make_table_chunk(
                        piece, parent, source, start_page, len(tables)
                    )
                    tables.append(table)
                    link_id = table.id
                else:
                    link_id = parent.id
                documents.append(
                    Document(
                        page_content=piece.text,
                        metadata={
                            "id": chunk_id,
                            "chunk_key": chunk_id,
                            "content_hash": content_hash,
                            "source": source,
                            "title": title,
                            "chunk": index,
                            # 表格子块指向整表（供检索侧独立展开），文本子块指向父块
                            "parent_id": link_id,
                            # 结构骨架：叶子标题 + 顶层到本块的完整标题路径
                            "heading": parent.heading,
                            "heading_path": parent.heading_path,
                            # 证据通路：块类型与页码（匹配不上为 0，不猜）
                            "block_type": piece.kind,
                            "start_page": start_page,
                            "end_page": end_page,
                        },
                    )
                )

        if not documents:
            raise ValueError(f"切块结果为空：{source}")

        return DoclingChunks(
            chunks=documents,
            markdown=markdown if is_pdf else None,
            parents=parents,
            images=list(images or []),
            tables=tables,
            blocks=list(blocks or []),
            figure_captions=dict(figure_captions or {}),
        )

    @staticmethod
    def _clean_markdown(markdown: str) -> str:
        """规则清洗 + 文档内段落级 SimHash 去重（固定流程，不可关闭）。"""
        from src.ingest.cleaner import (
            TextCleaner,
            deduplicate_paragraphs,
        )

        text = TextCleaner().clean(markdown)
        return deduplicate_paragraphs(text)

    def _split_into_parents(
        self, markdown: str, source: str, title: str
    ) -> list[ParentChunk]:
        """按完整 Markdown 章节或自然段切分父块。

        标题级别参与结构还原：用栈维护「顶层 → 当前」的标题路径，父块除叶子
        ``heading`` 外还携带完整 ``heading_path``，为层级检索/引用提供骨架。

        父块长度**不参与**切分决策：结构完整性优先，章节不允许被腰斩。
        长度兜底在检索侧——``_expand_to_parents`` 只还原 50–3000 字符的父块。
        子块长度由 ``_split_parent_into_children`` 控制。
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(markdown))

        parents: list[ParentChunk] = []

        if not matches:
            # 无标题文档按空行分隔的自然段切分，不按长度拆散段落。
            paragraphs = re.split(r"\n\s*\n+", markdown)
            for paragraph in paragraphs:
                chunk = paragraph.strip()
                if not chunk:
                    continue
                parents.append(
                    _make_parent(chunk, "", source, title, len(parents))
                )
            return parents

        # 处理第一个标题之前的正文
        if matches[0].start() > 0:
            preamble = markdown[: matches[0].start()].strip()
            if preamble:
                parents.append(
                    _make_parent(preamble, "", source, title, len(parents))
                )

        # (级别, 标题) 栈：弹出所有级别不低于当前标题的祖先，再压入自身
        stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            heading_path = " > ".join(text for _, text in stack)

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()

            parents.append(
                _make_parent(
                    content, heading_text, source, title, len(parents), heading_path
                )
            )

        return parents

    def _split_parent_into_blocks(self, content: str) -> list[BlockPiece]:
        """按原子单元切分父块，返回 ``(类型, 文本, 整表)`` 序列。

        表格**独占**一个块：既不与相邻文本合并，也不丢弃表体。
        索引文本取 :func:`_table_index_text`（表头 + 摘要；无章节路径的
        临时版本，``_build_chunks`` 会带 ``heading_path`` 重组），
        整表由调用方另存。
        """
        if not content.strip():
            return []

        units = _extract_atomic_units(content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.max_tokens * _CHARS_PER_TOKEN),
            chunk_overlap=int(self.overlap * _CHARS_PER_TOKEN),
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        max_chars = int(self.max_tokens * _CHARS_PER_TOKEN)
        blocks: list[BlockPiece] = []
        pending: list[str] = []

        def flush_pending() -> None:
            """把累积的文本片段按长度合并成文本块。"""
            if not pending:
                return
            current = ""
            for piece in pending:
                if not current:
                    current = piece
                elif len(current) + len(piece) + 2 <= max_chars:
                    current = current + "\n\n" + piece
                else:
                    blocks.append(BlockPiece(kind="text", text=current))
                    current = piece
            if current:
                blocks.append(BlockPiece(kind="text", text=current))
            pending.clear()

        for kind, text in units:
            if kind == "table":
                flush_pending()
                summary = _table_index_text(text)
                if summary:
                    blocks.append(
                        BlockPiece(kind="table", text=summary, full_text=text.strip())
                    )
            else:
                pending.extend(
                    piece.strip() for piece in text_splitter.split_text(text) if piece.strip()
                )
        flush_pending()
        return blocks

    def _split_parent_into_children(self, content: str) -> list[str]:
        """将父块内容切分为子块；表格作为原子单元保护."""
        return [piece.text for piece in self._split_parent_into_blocks(content)]



def _make_parent(
    content: str,
    heading: str,
    source: str,
    title: str,
    index: int,
    heading_path: str = "",
) -> ParentChunk:
    """构造父块。

    id 里**必须**带上块序号：只按 ``source/heading/content`` 哈希时，同一篇文档里
    出现两段完全相同的块（研报在每页末尾重复的风险提示、免责声明很常见）会算出
    同一个 id，写入 PostgreSQL 时直接撞 ``parent_chunks_pkey``——实测 43 页研报
    因此整篇入库失败。带序号既保证唯一，又保持「同内容同位置 → 同 id」的可复现性。
    """
    chunk_id = hashlib.sha256(
        f"{source}:parent:{index}:{heading}:{content}".encode("utf-8")
    ).hexdigest()
    return ParentChunk(
        id=chunk_id,
        heading=heading,
        content=content,
        source=source,
        title=title,
        heading_path=heading_path or heading,
    )


# 中文数字与半角/全角阿拉伯数字（章/节/条等编号字符集）
_CN_NUM = "一二三四五六七八九十百千万零〇两"
_DIGITS = "0-9０-９"
_NUM_CHARS = f"{_CN_NUM}{_DIGITS}"

# 标题标记的规范嵌套顺序：越靠前越外层。
# 注入时只对「本文档实际出现」的标记按此顺序依次映射到 ##..######，
# 因此不同类型不会互撞同一级（旧实现里「一、」与「第X节」都映射 ### 即撞车）。
_MARKER_ORDER: tuple[str, ...] = (
    "bian",          # 第X编
    "summary",       # 综述 / 综　述
    "zhang",         # 第X章
    "jie",           # 第X节
    "tiao",          # 第X条
    "kuan",          # 第X款
    "xiang",         # 第X项
    "cn_item",       # 一、
    "paren_cn",      # （一） / (一)
    "arabic_dot",    # 1. / 1、 / 1：
    "arabic_dot2",   # 1.1
    "arabic_dot3",   # 1.1.1
    "paren_arabic",  # （1） / (1)
)

_UNIT_TO_MARKER = {
    "编": "bian",
    "章": "zhang",
    "节": "jie",
    "条": "tiao",
    "款": "kuan",
    "项": "xiang",
}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UNIT_PREFIX_RE = re.compile(rf"^第\s*[{_NUM_CHARS}]+\s*(?P<unit>[编章节条款项])")
_SUMMARY_RE = re.compile(r"^综[　\s]*述$")
_CN_ITEM_RE = re.compile(rf"^[{_CN_NUM}]+\s*[、.．]")
_PAREN_CN_RE = re.compile(rf"^[（(]\s*[{_CN_NUM}]+\s*[）)]")
_PAREN_ARABIC_RE = re.compile(rf"^[（(]\s*[{_DIGITS}]+\s*[）)]")
_MULTI_ARABIC_RE = re.compile(rf"^[{_DIGITS}]+(?:\.[{_DIGITS}]+)+(?=\s|$)")
# 分隔符后不能再跟数字：避免把正文里的「1.5倍」误判成单级编号标题
_SINGLE_ARABIC_RE = re.compile(rf"^[{_DIGITS}]+\s*[、.．:：](?![{_DIGITS}])")

# 行内标题长度上限：超过即视为正文，避免把长句误判为标题
_MARKER_MAX_CHARS = 80


def _classify_heading_line(text: str) -> str | None:
    """识别一行文本（已去掉 ``#`` 前缀）属于哪种标题标记；无法识别返回 ``None``。"""
    stripped = text.strip()
    if not stripped:
        return None

    match = _UNIT_PREFIX_RE.match(stripped)
    if match:
        return _UNIT_TO_MARKER[match.group("unit")]

    if _SUMMARY_RE.match(stripped):
        return "summary"

    if len(stripped) >= _MARKER_MAX_CHARS:
        return None

    if _CN_ITEM_RE.match(stripped):
        return "cn_item"
    if _PAREN_CN_RE.match(stripped):
        return "paren_cn"
    if _PAREN_ARABIC_RE.match(stripped):
        return "paren_arabic"

    # 多级编号 1.1 / 1.1.1（点分多段，无尾随分隔符，要求后接空白或行尾）
    match = _MULTI_ARABIC_RE.match(stripped)
    if match:
        return "arabic_dot2" if match.group(0).count(".") == 1 else "arabic_dot3"

    # 单级编号 1. / 1、 / 1： / 1．
    if _SINGLE_ARABIC_RE.match(stripped):
        return "arabic_dot"

    return None


def _inject_heading_structure(markdown: str) -> str:
    """为缺少 Markdown 标题层级的中文公文/法规逐行注入标题结构。

    支持标记的规范嵌套顺序（由外到内）::

        第X编 > 综述 > 第X章 > 第X节 > 第X条 > 第X款 > 第X项
              > 一、 > （一） > 1. > 1.1 > 1.1.1 > （1）

    层级不写死为固定常量，而是取「本文档实际出现的标记」按上述顺序依次映射到
    ``##``..``######``：只有 ``一、``/``（一）`` 的公文会得到 ``##``/``###``，
    而含 ``章``/``节``/``条``/``一、``/``（一）`` 的法规会得到 ``##``..``######``，
    两套编号体系不再互撞同一级。``#`` 一级保留给文档标题。

    按行独立判定且幂等：围栏代码块内的行原样保留；解析器自带的（非标记）标题
    保持原样，而标记标题按本次映射统一层级，因此重复调用结果不变。
    """
    lines = markdown.split("\n")

    # 第一遍：统计文档中实际出现的标记，并记录「非标记」既有标题的层级
    present: set[str] = set()
    existing_levels: list[int] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading_match = _HEADING_LINE_RE.match(stripped)
        if heading_match:
            marker = _classify_heading_line(heading_match.group(2))
            if marker is None:
                # 解析器自带标题：作为注入层级的起点参考
                existing_levels.append(len(heading_match.group(1)))
            else:
                # 已注入的标记标题：计入 present，保证重复调用稳定
                present.add(marker)
            continue

        marker = _classify_heading_line(stripped)
        if marker:
            present.add(marker)

    if not present:
        return markdown

    ordered = [marker for marker in _MARKER_ORDER if marker in present]

    # 起始层级：默认 ##；若文档已有解析器标题，则从其下一层开始（放得下才算）
    base = 2
    if existing_levels:
        candidate = max(2, max(existing_levels) + 1)
        if candidate + len(ordered) - 1 <= 6:
            base = candidate

    levels = {marker: min(6, base + index) for index, marker in enumerate(ordered)}

    # 第二遍：按映射重写
    result: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue

        heading_match = _HEADING_LINE_RE.match(stripped)
        if heading_match:
            marker = _classify_heading_line(heading_match.group(2))
            if marker is None:
                result.append(line)
            else:
                text = heading_match.group(2).strip()
                result.append(f"{'#' * levels[marker]} {text}")
            continue

        marker = _classify_heading_line(stripped)
        if marker:
            result.append(f"{'#' * levels[marker]} {stripped}")
            continue

        result.append(line)

    return "\n".join(result)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """把二维数组渲染成 Markdown 竖线表（补�齐列数）。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [
        [html.escape(str(cell), quote=False)
         .replace('\\', '\\\\').replace('|', '\\|')
         .replace('\r\n', '\n').replace('\n', '<br>') for cell in row]
        + [""] * (width - len(row))
        for row in rows
    ]
    head = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([head, separator, *body])


def _table_markdown(table: str) -> str:
    """统一成 Markdown 竖线表。

    HTML 表要转换而不是原样保留：这张表会被展开进 LLM 上下文，也会作为
    「完整表」展示给分析师，``<table><tr><td>`` 标签既费 token 又难读。
    """
    if _HTML_TABLE_RE.search(table):
        layout = restore_html_table(table)
        if layout.error:
            return table.strip()
        markdown = _rows_to_markdown(layout.rows)
        if layout.merges:
            markdown += '\n\n<!-- 合并范围：' + ', '.join(layout.merges) + ' -->'
        return markdown or table.strip()
    return table.strip()


def normalize_html_tables(text: str) -> str:
    """用入库时相同的布局规则压缩正文内完整 HTML 表；失败时保留原文。"""
    return _HTML_TABLE_RE.sub(lambda match: "\n" + _table_markdown(match[0]) + "\n", text)


#: 行标签预览的条数与字符上限：描述「表里有哪些科目/条目」，不是数据本体
_TABLE_LABEL_PREVIEW_COUNT = 15
_TABLE_LABEL_PREVIEW_CHARS = 300


def _row_label_preview(rows: list[list[str]]) -> str:
    """首列去重标签预览（有界）。

    财务表的行标签就是指标名（营业收入/净利润/毛利率…），是「净利润预测」
    这类问题的主要查询词；只取标签不取数字，覆盖面比旧的「首行数据」更广、
    噪音更少（不止首行的指标都能命中）。
    """
    seen: set[str] = set()
    labels: list[str] = []
    used = 0
    for row in rows:
        if not row:
            continue
        label = str(row[0]).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        used += len(label) + 1
        if len(labels) >= _TABLE_LABEL_PREVIEW_COUNT or used >= _TABLE_LABEL_PREVIEW_CHARS:
            break
    return "、".join(labels)


def _table_index_text(table: str, heading_path: str = "") -> str:
    """构造表格索引文本：「表头 + 摘要（描述表展示了什么内容）」。

    组成（全部来自确定性信号，不调 LLM）：

    * **表题**：表格自身的单格大标题行（如「资产负债表(亿)」）——
      「这张表展示什么」的最直接来源；
    * **章节**：``heading_path``（表所在章节路径）；
    * **表头**：多格表头行（列名是 BM25/稠密召回的主要命中面）；
    * **行标签预览**：首列去重后的前若干个值（:func:`_row_label_preview`）。

    检索侧召回与重排全程只看这份紧凑文本；完整表格在输出拼上下文时才按
    ``table_id`` 从整表存储展开（见 chat_service.expand_table_docs）。
    HTML 与 Markdown 表统一经 :func:`_table_payload` 转成行再抽取。
    """
    layout = restore_html_table(table) if _HTML_TABLE_RE.search(table) else None
    if layout and layout.error:
        return table.strip()
    rows = layout.rows if layout else _table_payload(table)
    if not rows:
        return ""

    # 表题：开头连续的单格行（研报财务表第一行常是一格大标题，真正表头在下一行）
    title_cells: list[str] = []
    body_start = 0
    for row in rows:
        if sum(bool(str(cell).strip()) for cell in row) == 1 and str(row[0]).strip():
            title_cells.append(str(row[0]).strip())
            body_start += 1
        else:
            break
    body = rows[body_start:]

    lines: list[str] = []
    if title_cells:
        lines.append("表格：" + " ".join(title_cells))
    if heading_path:
        lines.append(f"章节：{heading_path}")
    if body:
        header_count = max(1, layout.header_rows - body_start) if layout else 1
        lines.append("表头：" + " / ".join(" | ".join(str(cell).strip() for cell in row) for row in body[:header_count]))
    else:
        header_count = 0
    labels = _row_label_preview(body[header_count:] if body else [])
    if labels:
        lines.append(f"行标签：{labels}")
    if not lines:
        # 空表/全是空单元格：返回空让调用方跳过（与旧的空摘要行为一致）
        return ""
    return "\n".join(lines)


_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?$")


def _table_payload(table: str) -> list[list[str]]:
    """把表格转成二维数组（去掉分隔行，首行为表头）；HTML 与 Markdown 都支持。"""
    if _HTML_TABLE_RE.search(table):
        return _html_table_rows(table)
    rows: list[list[str]] = []
    for line in table.splitlines():
        line = line.strip()
        if not line or line.startswith('<!--') or _TABLE_SEPARATOR_RE.fullmatch(line):
            continue
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        cells = re.split(r'(?<!\\)\|', line)
        # 先还原生成的换行标签，再解码实体；字面量 &lt;br&gt; 保持文字。
        rows.append([
            html.unescape(re.sub(r'\\([\\|])', r'\1', cell.strip()).replace('<br>', '\n'))
            for cell in cells
        ])
    return rows


def _make_table_chunk(
    piece: BlockPiece, parent: ParentChunk, source: str, start_page: int, index: int
) -> TableChunk:
    """由表格块构造整表记录；``id`` 稳定，便于重复入库时覆盖。"""
    payload = _table_payload(piece.full_text)
    table_id = hashlib.sha256(
        f"{parent.id}:table:{index}:{piece.full_text}".encode("utf-8")
    ).hexdigest()
    return TableChunk(
        id=table_id,
        parent_id=parent.id,
        # 统一成 Markdown：HTML 表在这个字段里也要是干净可读的竖线表，
        # 它会被展开进 LLM 上下文、也会作为「完整表」展示
        markdown=_table_markdown(piece.full_text),
        payload=payload,
        row_count=max(0, len(payload) - 1),
        source=source,
        heading_path=parent.heading_path,
        start_page=start_page,
    )


# 页码归属的匹配强度门槛：
# * ``_PAGE_PROBE_CHARS``：前缀式匹配（块比 chunk 长、chunk 是块的开头）所需的公共长度
# * ``_PAGE_MIN_CHARS``：包含式匹配（块整段出现在 chunk 里）所需的最小块长度，
#   太短的块（如「风险提示」）容易在多页重现，宁可不归属
_PAGE_PROBE_CHARS = 12
_PAGE_MIN_CHARS = 5
_PAGE_WIDEN_CHARS = 40


def _normalize_text(text: str) -> str:
    """去掉全部空白后比较，规避解析层与切块层换行差异导致的假不匹配。"""
    return re.sub(r"\s+", "", text or "")


def _shared_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _page_match_score(needle: str, haystack: str) -> int:
    """这个分页块与切块文本的匹配强度；0 表示不足以归属页码。

    两条判据，按证据强度：

    1. **块整段出现在 chunk 里**（chunk 就是由这个块拼出来的）——最强证据。
       content_list 的块文本是「裸文本」，而切块文本可能带 ``#`` / ``◼`` 等
       注入标记，所以不能用前缀相等来判断；包含关系天然容忍前后装饰。
    2. **chunk 开头与块开头一致**且公共长度足够——用于块比 chunk 长的情形
       （块被递归切分器切开）。
    """
    if not needle or not haystack:
        return 0
    if len(haystack) >= _PAGE_MIN_CHARS and haystack in needle:
        return len(haystack)
    shared = _shared_prefix_len(needle, haystack)
    if shared >= _PAGE_PROBE_CHARS:
        return shared
    if shared >= len(needle) and shared >= _PAGE_MIN_CHARS:
        return shared
    return 0


def _resolve_pages(text: str, blocks: Sequence[ContentBlock]) -> tuple[int, int]:
    """用分页文本块把切块结果归属到页码区间。

    先按匹配强度定位起始页，再用「块首内容出现在本块中」扩展结束页，
    以覆盖跨页合并的块。匹配不上返回 ``(0, 0)``——页码宁可缺失，不可乱填。
    """
    needle = _normalize_text(text)
    if not needle or not blocks:
        return 0, 0

    best_page, best_score = 0, 0
    for block in blocks:
        score = _page_match_score(needle, _normalize_text(block.text))
        if score > best_score:
            best_score, best_page = score, block.page
    if best_score == 0:
        return 0, 0

    start = end = best_page
    for block in blocks:
        haystack = _normalize_text(block.text)
        if not haystack:
            continue
        probe = haystack[:_PAGE_WIDEN_CHARS]
        if len(probe) >= _PAGE_MIN_CHARS and probe in needle:
            start = min(start, block.page)
            end = max(end, block.page)
    return start, end


_MD_TABLE_RE = re.compile(
    r"((?:^[ \t]*\|.*\|[ \t]*(?:\r?\n|$))+)",
    re.MULTILINE,
)
# MinerU 对研报里的财务表 / 盈利预测表输出的是 HTML ``<table>``，不是竖线表。
# 不识别它，这些表就既进不了 PostgreSQL，也做不了「表头 + 首行」索引，
# 还会被递归切分器从标签中间切断（出现 ``<td colspan=`` 这样的碎片）。
_HTML_TABLE_RE = re.compile(r"(<table\b.*?</table>)", re.DOTALL | re.IGNORECASE)


def _html_table_rows(table: str) -> list[list[str]]:
    """共用有界布局还原器；异常表格返回空 payload，原 HTML 另行保留。"""
    return restore_html_table(table).rows


def _extract_atomic_units(content: str) -> list[tuple[str, str]]:
    """提取表格（Markdown 竖线表 与 HTML 表）作为原子单元，其余作为文本。"""
    matches = sorted(
        [*_MD_TABLE_RE.finditer(content), *_HTML_TABLE_RE.finditer(content)],
        key=lambda match: match.start(),
    )
    units: list[tuple[str, str]] = []
    last_end = 0
    for match in matches:
        if match.start() < last_end:  # 已被前一个单元覆盖（嵌套/重叠）
            continue
        if match.start() > last_end:
            text = content[last_end : match.start()]
            if text.strip():
                units.append(("text", text))
        units.append(("table", match.group()))
        last_end = match.end()
    if last_end < len(content):
        text = content[last_end:]
        if text.strip():
            units.append(("text", text))
    return units


# 向后兼容别名
DoclingHybridChunker = HierarchicalChunker


_SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；]?|\n+")


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表（保留句末标点）。"""
    sentences = [m.group(0) for m in _SENTENCE_RE.finditer(text)]
    return [s for s in sentences if s.strip()]


class SemanticChunker(HierarchicalChunker):
    """语义分块：用本地句子嵌入检测语义断点，按主题边界切分子块。

    * 相邻句子余弦相似度低于阈值且当前段已积累足够长度时切分；
    * 父块构建逻辑与层级切块一致（子块仍归属标题父块）；
    * 表格沿用原子单元保护，不参与语义切分；
    * 模型延迟加载，仅在启用该切块器时初始化。
    """

    def __init__(
        self,
        max_tokens: int = DOCLING_CHUNK_MAX_TOKENS,
        overlap: int = CHILD_OVERLAP_TOKENS,
        *,
        break_threshold: float = SEMANTIC_BREAK_THRESHOLD,
        min_tokens: int = SEMANTIC_MIN_TOKENS,
        embed_model: str = SEMANTIC_EMBED_MODEL,
        device: str = SEMANTIC_DEVICE,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            overlap=overlap,
        )
        self.break_threshold = break_threshold
        self.min_tokens = min_tokens
        self.embed_model = embed_model
        self.device = device
        self._model = None

    def _get_model(self):
        """延迟加载 sentence-transformer 模型（进程内复用）。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model, device=self.device)
        return self._model

    def _split_parent_into_blocks(self, content: str) -> list[BlockPiece]:
        """语义分块不走原子单元循环，这里按语义块补齐类型标记。

        表格仍由 :func:`_extract_atomic_units` 识别，索引文本仍是
        「表头 + 摘要」（无章节版本）；但语义路径不保证表格独占一块
        （与历史行为一致，避免改变既有切块结果）。
        """
        summaries = {
            _table_index_text(text)
            for kind, text in _extract_atomic_units(content)
            if kind == "table"
        }
        return [
            BlockPiece(kind="table" if piece in summaries else "text", text=piece)
            for piece in self._split_parent_into_children(content)
        ]

    def _split_parent_into_children(self, content: str) -> list[str]:
        """文本单元走语义断点切分，表格保持原子。"""
        if not content.strip():
            return []

        units = _extract_atomic_units(content)
        pieces: list[str] = []
        for kind, text in units:
            if kind == "table":
                summary = _table_index_text(text)
                if summary:
                    pieces.append(summary)
            else:
                pieces.extend(self._semantic_split(text))

        # 与小尺寸父子切块相同的合并逻辑：小片段合并，超长截断保护
        max_chars = int(self.max_tokens * _CHARS_PER_TOKEN)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece.strip():
                continue
            if not current:
                current = piece
            elif len(current) + len(piece) + 2 <= max_chars:
                current = current + "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        """基于相邻句嵌入相似度切分语义段。"""
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        import numpy as np

        model = self._get_model()
        embeddings = model.encode(
            sentences, normalize_embeddings=True, batch_size=16
        )

        segments: list[str] = []
        current = [sentences[0]]
        current_len = len(sentences[0])
        for i in range(1, len(sentences)):
            similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
            if similarity < self.break_threshold and current_len >= self.min_tokens:
                segments.append("".join(current).strip())
                current = [sentences[i]]
                current_len = len(sentences[i])
            else:
                current.append(sentences[i])
                current_len += len(sentences[i])
        segments.append("".join(current).strip())

        return [s for s in segments if s]


def get_chunker() -> HierarchicalChunker:
    """根据配置开关返回对应切块器实例（语义 > 层级）。"""
    if ENABLE_SEMANTIC_CHUNKER:
        return SemanticChunker()
    return HierarchicalChunker()
