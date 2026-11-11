"""MinerU 产物定位：文件名带文档名前缀，且同目录还有 v2 版本。

这两个坑都来自真实产物实测：用精确名 ``content_list.json`` 找不到任何文件
（真实文件名是 ``H3_AP..._content_list.json``），于是静默退化成 pymupdf 页文本，
页码归属几乎全为 0；而 ``*content_list.json`` 又不能把嵌套结构的 v2 选中。
"""

from pathlib import Path

from src.ingest.mineru_parser import (
    _blocks_from_content_list,
    _drop_embedded_images,
    _figure_captions_from_content_list,
    _find_content_list,
)


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_finds_content_list_with_document_name_prefix(tmp_path):
    auto = tmp_path / "H3_AP1" / "auto"
    markdown = _write(auto / "H3_AP1.md", "# t")
    expected = _write(auto / "H3_AP1_content_list.json")

    assert _find_content_list(markdown) == expected


def test_does_not_pick_the_v2_variant(tmp_path):
    auto = tmp_path / "doc" / "auto"
    markdown = _write(auto / "doc.md", "# t")
    flat = _write(auto / "doc_content_list.json")
    _write(auto / "doc_content_list_v2.json", "[]")

    assert _find_content_list(markdown) == flat


def test_returns_none_when_no_content_list(tmp_path):
    auto = tmp_path / "doc" / "auto"
    markdown = _write(auto / "doc.md", "# t")

    assert _find_content_list(markdown) is None


def test_searches_one_level_up_as_well(tmp_path):
    markdown = _write(tmp_path / "doc" / "auto" / "deep" / "doc.md", "# t")
    expected = _write(tmp_path / "doc" / "auto" / "doc_content_list.json")

    assert _find_content_list(markdown) == expected


def test_chart_caption_counts_as_block_text():
    """真实产物里图表注解字段是 chart_caption，不是 img_caption。"""
    payload = [
        {"type": "chart", "chart_caption": ["图表1 盈利预测"], "page_idx": 2},
        {"type": "text", "text": "正文", "page_idx": 0},
    ]

    blocks = _blocks_from_content_list(payload)

    assert [(b.text, b.page) for b in blocks] == [("图表1 盈利预测", 3), ("正文", 1)]


# --- 原文图注抽取（图片块索引文本的语义来源） -------------------------------

def test_figure_captions_extracted_per_page():
    """图注按页聚合（1-based 页码）：图片块组装时做页级对齐。"""
    payload = [
        {"type": "image", "img_caption": ["图1 燃机订单结构"], "page_idx": 2},
        {"type": "chart", "chart_caption": ["图表2 盈利预测"], "page_idx": 2},
        {"type": "text", "text": "正文", "page_idx": 0},
    ]

    captions = _figure_captions_from_content_list(payload)

    assert captions == {3: ["图1 燃机订单结构", "图表2 盈利预测"]}


def test_figure_captions_skip_invalid_entries():
    """空图注 / 空白字符串 / 非法页码 / 非字典条目一律跳过，不抛异常。"""
    payload = [
        {"type": "image", "img_caption": []},                       # 空列表
        {"type": "image", "img_caption": "  "},                     # 空白字符串
        {"type": "image", "img_caption": ["图3"], "page_idx": -1},  # 页码非法
        "not-a-dict",
    ]

    assert _figure_captions_from_content_list(payload) == {}


def test_figure_captions_accept_plain_string_value():
    """部分产物的图注字段是纯字符串而非列表，同样要能用。"""
    payload = [{"type": "image", "img_caption": "图4 产能布局", "page_idx": 5}]

    assert _figure_captions_from_content_list(payload) == {6: ["图4 产能布局"]}


def test_figure_captions_use_printed_page_map():
    payload = [{"type": "image", "img_caption": ["图1 订单结构"], "page_idx": 2}]

    assert _figure_captions_from_content_list(payload, page_map=(0, 0, 1, 2, 3)) == {
        1: ["图1 订单结构"]
    }


def test_drop_embedded_data_uri_images():
    """MinerU 4 会把图片内联成 data URI，正文切块不能带上这些字节。"""
    markdown = (
        "# 标题\n\n"
        "![图](data:image/jpeg;base64,/9j/4AAQ)\n\n"
        "利润增速同比超 20%。\n"
        "![远程](https://example.com/a.png)\n"
    )

    cleaned = _drop_embedded_images(markdown)

    assert "data:image" not in cleaned
    assert "利润增速同比超 20%。" in cleaned
    assert "https://example.com/a.png" in cleaned


def test_mineru4_image_caption_and_list_items():
    """MinerU 4 的 content_list 用 image_caption，列表正文在 list_items。"""
    payload = [
        {"type": "image", "image_caption": ["图1 订单结构"], "page_idx": 0},
        {"type": "list", "list_items": ["要点一", "要点二"], "page_idx": 1},
    ]

    assert [(b.text, b.page) for b in _blocks_from_content_list(payload)] == [
        ("图1 订单结构", 1),
        ("要点一 要点二", 2),
    ]
    assert _figure_captions_from_content_list(payload) == {1: ["图1 订单结构"]}
