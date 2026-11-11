"""纯文本解码的编码守卫回归（RAG-02）。

``docs/rag-review.md`` 的问题：TXT / Markdown 固定以 UTF-8 + ``errors="replace"``
读取，GBK / UTF-16 内容被替换成 U+FFFD 乱码后继续入库，公司名、金额与关键词
静默损坏，用户也拿不到编码异常提示。

验收标准：
* UTF-8、UTF-8 BOM、UTF-16 BOM 文本的中文、金额和标点完整保留；
* GBK 样本通过明确配置或可信检测正确读取；不支持时明确报错；
* 损坏或无法确定编码的文件不以替换乱码的方式成功入库，并记录实际使用的编码。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest import mineru_parser
from src.ingest.mineru_parser import (
    TextDecodeError,
    decode_text,
    parse_with_mineru,
    read_text_file,
)

_TEXT = "# 投资要点\n\n贵州茅台2026H1营业收入 100.5 亿元，同比增长 12%（含预收款）。\n"


def _workdir() -> Path:
    path = Path(__file__).resolve().parents[1] / ".tmp" / "text-encoding"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(name: str, payload: bytes) -> Path:
    path = _workdir() / name
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# 可无损读取的编码
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,payload,expected_encoding",
    [
        ("utf8.txt", _TEXT.encode("utf-8"), "utf-8"),
        ("bom.md", b"\xef\xbb\xbf" + _TEXT.encode("utf-8"), "utf-8-sig"),
        ("utf16le.txt", _TEXT.encode("utf-16"), "utf-16"),
        # 无 BOM 的 UTF-16：靠隔字节 NUL 特征探测，编码名如实记录为具体字节序
        ("utf16be.txt", _TEXT.encode("utf-16-be"), "utf-16-be"),
    ],
)
def test_bom_and_utf8_texts_are_lossless(name, payload, expected_encoding):
    """UTF-8 与带 BOM 的文本必须逐字保留（中文、金额、标点都不能变）。"""
    text, encoding = read_text_file(_write(name, payload))

    assert text == _TEXT
    assert encoding == expected_encoding
    assert "\ufffd" not in text
    assert "100.5 亿元" in text and "（含预收款）" in text


def test_gbk_sample_is_decoded_via_audited_fallback():
    """GBK 样本按可配置回退链正确读取，并记录实际使用的编码。"""
    text, encoding = read_text_file(_write("gbk.txt", _TEXT.encode("gbk")))

    assert text == _TEXT
    assert encoding in {"gb18030", "gbk"}
    assert "\ufffd" not in text


def test_explicit_encoding_override_is_honoured(monkeypatch):
    """业务明确支持的其他编码可显式指定（可配置、可审计）。"""
    monkeypatch.setattr(mineru_parser, "TEXT_ENCODING", "gbk")
    text, encoding = decode_text(_TEXT.encode("gbk"), source="gbk.txt")

    assert text == _TEXT and encoding == "gbk"


def test_fallback_chain_can_be_disabled(monkeypatch):
    """关闭回退链后不支持的编码必须明确报错，而不是硬解出乱码。"""
    monkeypatch.setattr(mineru_parser, "TEXT_ENCODING_FALLBACKS", ())

    with pytest.raises(TextDecodeError, match="无法可信解码"):
        decode_text(_TEXT.encode("gbk"), source="gbk.txt")


# ---------------------------------------------------------------------------
# 不可信 / 损坏输入
# ---------------------------------------------------------------------------


def test_truncated_utf16_raises_instead_of_replacement_characters():
    """损坏的 BOM 内容必须明确失败，不得以替换字符入库。"""
    with pytest.raises(TextDecodeError, match="UTF-16"):
        decode_text(b"\xff\xfe\x00\x41\x00", source="broken.txt")


def test_undecodable_bytes_raise_with_actionable_message():
    """无法确定编码时给出可操作提示（转 UTF-8 或显式指定编码）。"""
    with pytest.raises(TextDecodeError) as excinfo:
        decode_text(b"\x81\x30\x81\x30\xff\xff\xfe\xfe", source="unknown.bin")

    message = str(excinfo.value)
    assert "unknown.bin" in message
    assert "TEXT_ENCODING" in message


def test_parse_records_actual_encoding_for_audit():
    """解析结果必须记录实际使用的编码（排障与审计依据）。"""
    result = parse_with_mineru(_write("audit.md", _TEXT.encode("gbk")), source="audit.md")

    assert result.markdown == _TEXT
    assert result.stats["text_encoding"] in {"gb18030", "gbk"}


def test_parse_fails_explicitly_when_encoding_is_untrusted(monkeypatch):
    """编码不可信时解析阶段就失败，文档不会带着乱码进索引。"""
    monkeypatch.setattr(mineru_parser, "TEXT_ENCODING_FALLBACKS", ())
    path = _write("bad.txt", _TEXT.encode("gbk"))

    with pytest.raises(mineru_parser.MinerUParseError, match="无法可信解码"):
        parse_with_mineru(path, source="bad.txt")
