"""解析产物字形噪声：符号字体私有区字形不得让整篇文档解析失败。

真实故障（2026-10-24，机械设备行业点评报告）：该 PDF 的 5 个小标题用了
Wingdings 项目符号，文本层把它们搬成私有区码位 ``U+F06E``；MinerU 照抄进
Markdown 后，``decode_text`` 的编码可信度检查算出 0.998 < 0.999，于是整篇
文档被判「无法可信解码文本」而入库失败。

定位结论：编码契约已知（MinerU 自己写 UTF-8），可疑字符只可能是字形级噪声，
因此必须按字形清洗而不是按编码失败处理；只有替换字符占比过高（文本层大面积
失败）才整篇失败。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunking.chunk_worker import chunk_markdown_inline
from src.ingest import mineru_parser
from src.ingest.cleaner import TextCleaner
from src.ingest.mineru_parser import (
    MAX_UNMAPPED_GLYPH_RATIO,
    TextDecodeError,
    _blocks_from_content_list,
    _figure_captions_from_content_list,
    clean_parser_text,
    decode_parser_output,
    parse_with_mineru,
)

#: Wingdings 的方块项目符号：真实样本里出现 5 次（每页小标题前一个）
_BULLET = "\uf06e"

_MARKDOWN = (
    "# 机械设备行业点评报告\n\n"
    f"## {_BULLET} 具身AI加速发展，看好人形机器人产业应用\n\n"
    "模型端迭代与物理AI赋能共同推动产业化落地，行业规模有望持续扩张。\n\n"
    f"## {_BULLET} 投资建议\n\n"
    "建议关注本体、零部件与场景应用三条主线。\n"
)


def _workdir() -> Path:
    path = Path(__file__).resolve().parents[1] / ".tmp" / "symbol-glyphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 字形清洗
# ---------------------------------------------------------------------------


def test_clean_parser_text_drops_symbol_glyphs_and_keeps_body():
    audit = clean_parser_text(f"## {_BULLET} 投资要点\n营业收入 100.5 亿元（含预收）")

    assert audit.symbol_glyphs == 1
    assert _BULLET not in audit.text
    # 正文（中文、金额、全角标点）一字不动，只少了那个项目符号
    assert "投资要点" in audit.text and "100.5 亿元（含预收）" in audit.text
    assert audit.text == "##  投资要点\n营业收入 100.5 亿元（含预收）"


def test_clean_parser_text_counts_control_and_replacement_chars():
    audit = clean_parser_text("正文\x00残留\ufffd字\n第二行\t制表符保留")

    assert audit.control_chars == 1 and audit.unmapped_glyphs == 1
    assert audit.text == "正文残留字\n第二行\t制表符保留"


def test_text_cleaner_also_drops_symbol_glyphs():
    """兜底：不经解析层的 Markdown（存量产物重切）也不能带出私有区字形。"""
    result = TextCleaner().clean_with_audit(f"## {_BULLET} 投资要点")

    assert _BULLET not in result.text
    assert any(entry.rule == "ocr_garbage" for entry in result.removals)


# ---------------------------------------------------------------------------
# 产物解码：编码严格，字形宽容
# ---------------------------------------------------------------------------


def test_decode_parser_output_accepts_symbol_font_bullets():
    """本故障的回归点：带 Wingdings 项目符号的产物必须解析成功。"""
    text, stats = decode_parser_output(_MARKDOWN.encode("utf-8"), source="report.pdf")

    assert _BULLET not in text
    assert "具身AI加速发展" in text and "模型端迭代与物理AI赋能" in text
    assert stats["symbol_glyph_count"] == 2
    assert stats["unmapped_glyph_count"] == 0
    assert stats["parser_encoding"] == "utf-8"


def test_decode_parser_output_rejects_non_utf8_bytes():
    with pytest.raises(TextDecodeError, match="不是合法 UTF-8"):
        decode_parser_output("投资要点".encode("gbk"), source="report.pdf")


def test_decode_parser_output_rejects_wholesale_unmapped_glyphs():
    """替换字符占比过高说明文本层大面积失败，不能把残缺正文入库。"""
    body = "正文" * 10
    broken = ("\ufffd" * 10 + body).encode("utf-8")

    assert 10 / (10 + len(body)) > MAX_UNMAPPED_GLYPH_RATIO
    with pytest.raises(TextDecodeError, match="字形映射大面积失败"):
        decode_parser_output(broken, source="scan.pdf")


def test_decode_parser_output_tolerates_a_few_unmapped_glyphs():
    body = "投资要点与盈利预测" * 20
    text, stats = decode_parser_output((body + "\ufffd").encode("utf-8"))

    assert "\ufffd" not in text
    assert stats["unmapped_glyph_count"] == 1


# ---------------------------------------------------------------------------
# content_list 派生文本（分页块 / 图注）
# ---------------------------------------------------------------------------


def test_blocks_and_captions_drop_symbol_glyphs():
    payload = [
        {"type": "text", "text": f"{_BULLET} 投资要点", "page_idx": 0},
        {"type": "image", "img_caption": [f"{_BULLET} 图1 产能布局"], "page_idx": 1},
    ]

    blocks = _blocks_from_content_list(payload)
    captions = _figure_captions_from_content_list(payload)

    assert [block.text for block in blocks] == ["投资要点", "图1 产能布局"]
    assert captions == {2: ["图1 产能布局"]}


# ---------------------------------------------------------------------------
# 端到端：解析 → 清洗 → 切块（真实链路，只在 MinerU 边界打桩）
# ---------------------------------------------------------------------------


def test_parse_and_chunk_survives_symbol_glyphs(monkeypatch):
    workdir = _workdir()
    monkeypatch.setattr(mineru_parser, "MINERU_OUTPUT_DIR", str(workdir / "mineru-out"))
    monkeypatch.setattr(mineru_parser, "extract_images", lambda _path: ())

    def fake_parse_binary(input_path: Path, output_dir: Path) -> Path:
        target = Path(output_dir) / input_path.stem / "auto"
        target.mkdir(parents=True, exist_ok=True)
        markdown = target / f"{input_path.stem}.md"
        markdown.write_text(_MARKDOWN, encoding="utf-8")
        return markdown

    monkeypatch.setattr(mineru_parser, "_parse_binary_document", fake_parse_binary)

    pdf = workdir / "机械设备行业点评报告.pdf"
    pdf.write_bytes(b"%FAKE-PDF-BYTES%")

    result = parse_with_mineru(pdf, source=pdf.name)

    assert _BULLET not in result.markdown
    assert result.stats["symbol_glyph_count"] == 2
    assert result.stats["text_encoding"] == "utf-8"

    chunks = chunk_markdown_inline(result.markdown, source=pdf.name, is_pdf=True)
    headings = {chunk.metadata["heading"] for chunk in chunks.chunks}
    assert "具身AI加速发展，看好人形机器人产业应用" in headings
    assert all(_BULLET not in chunk.page_content for chunk in chunks.chunks)
