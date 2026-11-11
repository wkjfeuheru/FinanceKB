"""Local MinerU document parsing adapter using the Python API."""
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import (
    MINERU_BACKEND,
    MINERU_KEEP_ARTIFACTS_ON_ERROR,
    MINERU_METHOD,
    MINERU_OUTPUT_DIR,
    TEXT_ENCODING,
    TEXT_ENCODING_FALLBACKS,
)
from src.ingest.page_labels import detect_printed_pages_from_path, map_physical_page
from src.ingest.pdf_assets import (
    ContentBlock,
    ImageAsset,
    extract_blocks,
    extract_images,
)

logger = logging.getLogger(__name__)

#: 纯文本后缀：直接按文本读取，不走 MinerU 版面解析
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

_REPLACEMENT_CHAR = "\ufffd"


class TextDecodeError(ValueError):
    """文本编码无法可信确定（不支持的编码或解码结果不可信）。"""


def _text_confidence(text: str) -> float:
    """解码结果可信度（0~1）：替换字符、控制字符与私用区字符都会拉低。"""
    if not text:
        return 1.0
    suspicious = text.count(_REPLACEMENT_CHAR)
    suspicious += sum(
        1 for char in text
        if (ord(char) < 32 and char not in "\t\n\r") or 0xE000 <= ord(char) <= 0xF8FF
        or 0xD800 <= ord(char) <= 0xDFFF
    )
    return 1.0 - suspicious / len(text)


def _utf16_without_bom_encoding(data: bytes) -> str | None:
    """无 BOM 的 UTF-16 探测：ASCII 区会出现大量「隔字节 NUL」。

    不做这层探测时，无 BOM 的 UTF-16 会成为「非法 UTF-8」，随后被宽松回退编码
    （GB18030）解成一片看似正常的汉字——正是本次要消除的静默乱码。
    """
    sample = data[:4096]
    if len(sample) < 8:
        return None
    half = len(sample) / 2
    even_nulls = sample[0::2].count(0) / half
    odd_nulls = sample[1::2].count(0) / half
    if odd_nulls > 0.2 and even_nulls < 0.05:
        return "utf-16-le"
    if even_nulls > 0.2 and odd_nulls < 0.05:
        return "utf-16-be"
    return None


def decode_text(
    data: bytes, *, encoding: str | None = None, source: str = ""
) -> tuple[str, str]:
    """按「BOM → 严格主编码 → 可配置回退链」解码纯文本，返回 ``(文本, 实际编码)``。

    为什么不能直接 ``errors="replace"``：GBK / UTF-16 内容会被替换成 U+FFFD 乱码后
    继续入库，公司名、金额与关键词静默损坏，用户也得不到编码问题的提示。

    GB18030 之类的宽松编码几乎不会报错，因此回退结果必须再通过可信度检查：
    含替换字符或控制字符一律视为失败，宁可明确报错，也不入库乱码。
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        # BOM 是权威声明：UTF-16 双字节序
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError as exc:
            raise TextDecodeError(f"UTF-16 解码失败（source={source}）：{exc}") from exc
    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return data.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise TextDecodeError(f"UTF-8 BOM 解码失败（source={source}）：{exc}") from exc

    probed = _utf16_without_bom_encoding(data)
    if probed:
        logger.warning(
            "文本无 BOM 但呈现 %s 特征，按该编码解码（source=%s）", probed, source or "unknown"
        )
        try:
            return data.decode(probed), probed
        except UnicodeDecodeError as exc:
            raise TextDecodeError(f"{probed} 解码失败（source={source}）：{exc}") from exc

    primary = (encoding or TEXT_ENCODING or "utf-8").strip().lower() or "utf-8"
    attempts = [primary, *[item for item in TEXT_ENCODING_FALLBACKS if item != primary]]
    failures: list[str] = []
    for candidate in attempts:
        try:
            text = data.decode(candidate, errors="strict")
        except (UnicodeDecodeError, LookupError) as exc:
            failures.append(f"{candidate}: {type(exc).__name__}")
            continue
        confidence = _text_confidence(text)
        if confidence < 0.999:
            failures.append(f"{candidate}: 解码结果不可信（可信度 {confidence:.3f}）")
            continue
        if candidate != primary:
            logger.warning(
                "文本未按 %s 解码，实际使用 %s（source=%s）：请确认该编码是预期来源",
                primary, candidate, source or "unknown",
            )
        return text, candidate
    raise TextDecodeError(
        f"无法可信解码文本（source={source}，尝试：{'; '.join(failures)}）；"
        "请将文件转换为 UTF-8，或用 TEXT_ENCODING 显式指定业务支持的编码"
    )


def read_text_file(path: str | Path, *, source: str = "") -> tuple[str, str]:
    """读取纯文本文件并返回 ``(文本, 实际编码)``；编码不可信时抛异常。"""
    target = Path(path)
    return decode_text(target.read_bytes(), source=source or target.name)


# ---------------------------------------------------------------------------
# 解析产物（MinerU 自写的 UTF-8）的字形级清洗
# ---------------------------------------------------------------------------

#: 符号字体（Wingdings / Symbol / 部分中文排版字体）在 PDF 文本层会落成 Unicode
#: 私有区码位（U+E000–U+F8FF）。真实样本：``U+F06E`` 是 Wingdings 的方块项目符号，
#: MinerU 按码位原样搬运。这类码位不是正文——留着会污染标题与嵌入向量，
#: 更会因为「疑似乱码」让整篇文档解析失败（见 :func:`decode_parser_output`）。
_SYMBOL_GLYPH_RE = re.compile(r"[\ue000-\uf8ff]+")

#: 控件字符（制表符、换行、回车除外）：PDF 文本层偶有残留，同样不是正文。
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: 字形无法映射时 MinerU 落成替换字符 U+FFFD。个别字形丢失不影响整篇，
#: 占比超过该阈值说明文本层大面积失败，必须明确失败而不是入库残缺正文。
MAX_UNMAPPED_GLYPH_RATIO = 0.02


@dataclass(frozen=True)
class ParserTextAudit:
    """解析产物文本的清洗结果与噪声计数（计数写入 ``stats`` 供排障）。"""

    text: str
    symbol_glyphs: int = 0
    unmapped_glyphs: int = 0
    control_chars: int = 0

    @property
    def suspicious(self) -> int:
        """本次清洗清掉的噪声字符总数。"""
        return self.symbol_glyphs + self.unmapped_glyphs + self.control_chars


def clean_parser_text(text: str) -> ParserTextAudit:
    """清除解析产物里的**字形级**噪声，返回清洗结果与各计数。

    只清「不携带任何文本信息」的码位：符号字体私有区字形、控件字符、
    替换字符 U+FFFD。正文（含中文标点、金额、百分比）一字不动。
    """
    unmapped = text.count(_REPLACEMENT_CHAR)
    symbol_glyphs = sum(
        len(match.group()) for match in _SYMBOL_GLYPH_RE.finditer(text)
    )
    control_chars = sum(
        len(match.group()) for match in _CONTROL_CHAR_RE.finditer(text)
    )
    cleaned = _SYMBOL_GLYPH_RE.sub("", text)
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
    cleaned = cleaned.replace(_REPLACEMENT_CHAR, "")
    return ParserTextAudit(
        text=cleaned,
        symbol_glyphs=symbol_glyphs,
        unmapped_glyphs=unmapped,
        control_chars=control_chars,
    )


def decode_parser_output(
    data: bytes, *, source: str = ""
) -> tuple[str, dict[str, int | str]]:
    """解码 **MinerU 自己写的** Markdown，返回 ``(文本, 清洗统计)``。

    与 :func:`decode_text` 的分工正是本次故障的根因所在：

    * 纯文本文件的编码是**未知**的（可能是 GBK / UTF-16），因此任何可疑字符
      都必须让整篇失败，否则乱码会静默入库；
    * MinerU 产物的编码是**已知契约**（UTF-8），可疑字符只可能是字形级噪声。
      把它们一并当成「编码不可信」，等于让一篇带 Wingdings 项目符号的正常
      研报永远解析不了（真实故障：``U+F06E`` 出现 5 次，可信度 0.998 < 0.999）。

    因此判定分层：编码必须严格合法（非 UTF-8 字节直接失败）；字形噪声逐字
    清掉并计数；只有替换字符占比超过 :data:`MAX_UNMAPPED_GLYPH_RATIO`
    （文本层大面积失败）才判定整篇不可信。
    """
    payload = data[3:] if data.startswith(b"\xef\xbb\xbf") else data
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TextDecodeError(
            f"MinerU 输出不是合法 UTF-8（source={source or 'unknown'}）：{exc}"
        ) from exc

    audit = clean_parser_text(text)
    total = len(text)
    ratio = audit.unmapped_glyphs / total if total else 0.0
    if ratio > MAX_UNMAPPED_GLYPH_RATIO:
        raise TextDecodeError(
            f"MinerU 输出字形映射大面积失败（source={source or 'unknown'}）："
            f"替换字符 {audit.unmapped_glyphs}/{total}（{ratio:.1%} > "
            f"{MAX_UNMAPPED_GLYPH_RATIO:.0%}），文本层不可信；"
            "请确认该 PDF 是文本层文件而非扫描件"
        )
    if audit.suspicious:
        logger.info(
            "清理解析产物字形噪声（source=%s）：符号字形 %d、替换字符 %d、控件字符 %d",
            source or "unknown",
            audit.symbol_glyphs,
            audit.unmapped_glyphs,
            audit.control_chars,
        )
    stats: dict[str, int | str] = {
        "parser_encoding": "utf-8",
        "symbol_glyph_count": audit.symbol_glyphs,
        "unmapped_glyph_count": audit.unmapped_glyphs,
        "control_char_count": audit.control_chars,
    }
    return audit.text, stats


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    parser: str = "mineru"
    parser_version: str = "unknown"
    stats: dict[str, int | str] = field(default_factory=dict)
    # 内嵌图片与分页文本块：Markdown 里拿不到，页码溯源与图片索引都依赖它们。
    images: tuple[ImageAsset, ...] = ()
    blocks: tuple[ContentBlock, ...] = ()
    # 原文图注（content_list 的 image_caption / img_caption / chart_caption）：页码 -> 图注列表。
    # 图片块的索引文本靠它兜底——视觉描述失败或超外发上限的图，
    # 至少还能靠原文图注被语义召回。
    figure_captions: dict[int, list[str]] = field(default_factory=dict)


class MinerUParseError(RuntimeError):
    """MinerU was unavailable or did not produce valid output."""


_CONTENT_LIST_SUFFIX = "content_list.json"
# content_list 的分块类型字段：正文优先，表格与代码体兜底，最后才用图表注解。
_BLOCK_TEXT_KEYS = ("text", "table_body", "code_body")
# MinerU 3 图片注解是 img_caption；4.x content_list 写成 image_caption。
# 图表注解字段是 chart_caption。
_BLOCK_CAPTION_KEYS = ("img_caption", "image_caption", "table_caption", "chart_caption")


def _find_content_list(markdown_path: Path) -> Path | None:
    """定位 MinerU 的 ``*_content_list.json``（与 Markdown 同级或上一层）。

    两个来自真实产物的要点：

    * 文件名**带文档名前缀**（``H3_AP..._content_list.json``），必须用
      ``*content_list.json`` 通配。用精确名 ``content_list.json`` 永远匹配不到，
      会静默退化成 pymupdf 页文本，页码归属随之几乎全为 0；
    * 同目录还有 ``*_content_list_v2.json``（嵌套结构，字段与这里解析的扁平格式
      不同），**不能**选中。``*content_list.json`` 恰好只匹配扁平版。
    """
    candidates: list[Path] = []
    for root in (markdown_path.parent, markdown_path.parent.parent):
        if root.is_dir():
            candidates.extend(root.rglob(f"*{_CONTENT_LIST_SUFFIX}"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def _blocks_from_content_list(
    payload: Any, page_map: tuple[int, ...] | None = None
) -> tuple[ContentBlock, ...]:
    """把 MinerU content_list 转成 ``(文本, 页码)`` 序列。

    content_list 的``page_idx``是 0-based 物理页，先转成 1-based 再映射为印刷页。
    对字段缺失 / 类型不符 / 空文本一律跳过，不抛异常——页码宁可缺失，不可乱填。
    """
    if not isinstance(payload, list):
        return ()
    blocks: list[ContentBlock] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_page = item.get("page_idx")
        if isinstance(raw_page, bool) or not isinstance(raw_page, int) or raw_page < 0:
            continue
        text = ""
        for key in _BLOCK_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            items = item.get("list_items")
            if isinstance(items, list):
                joined = " ".join(str(part).strip() for part in items if str(part).strip())
                if joined:
                    text = joined
        if not text:
            for key in _BLOCK_CAPTION_KEYS:
                value = item.get(key)
                if isinstance(value, list):
                    joined = " ".join(
                        str(part).strip() for part in value if str(part).strip()
                    )
                    if joined:
                        text = joined
                        break
        if not text:
            continue
        # 分块文本与 Markdown 同源：符号字体项目符号在这里同样要清掉，
        # 否则它会随页码溯源进入图片/表格的索引文本。
        text = clean_parser_text(text).text.strip()
        if not text:
            continue
        blocks.append(
            ContentBlock(text=text, page=map_physical_page(raw_page + 1, page_map))
        )
    return tuple(blocks)


# 图注字段：MinerU 3 是 img_caption，4.x 是 image_caption；图表是 chart_caption。
_FIGURE_CAPTION_KEYS = ("img_caption", "image_caption", "chart_caption")


def _figure_captions_from_content_list(
    payload: Any, page_map: tuple[int, ...] | None = None
) -> dict[int, list[str]]:
    """从 content_list 抽取「页码 -> 图注列表」（印刷页，1-based）。

    原文图注是图片块索引文本的重要语义来源：视觉描述失败、图片过小被拦
    或超过外发上限的图，至少还能靠图注被召回。字段缺失/类型不符一律跳过，
    不抛异常——图注宁缺，不拼错。
    """
    captions: dict[int, list[str]] = {}
    if not isinstance(payload, list):
        return captions
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_page = item.get("page_idx")
        if isinstance(raw_page, bool) or not isinstance(raw_page, int) or raw_page < 0:
            continue
        for key in _FIGURE_CAPTION_KEYS:
            value = item.get(key)
            if isinstance(value, list):
                text = " ".join(
                    str(part).strip() for part in value if str(part).strip()
                )
            elif isinstance(value, str):
                text = value.strip()
            else:
                continue
            text = clean_parser_text(text).text.strip() if text else ""
            if text:
                captions.setdefault(
                    map_physical_page(raw_page + 1, page_map), []
                ).append(text)
    return captions


def _load_content_list(markdown_path: Path) -> Any | None:
    content_list_path = _find_content_list(markdown_path)
    if content_list_path is None:
        return None
    try:
        return json.loads(content_list_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - 产物异常只降级，不影响入库
        logger.warning("读取 content_list 失败（%s）：%s", content_list_path, exc)
        return None


def _extract_blocks(
    payload: Any, input_path: Path, page_map: tuple[int, ...] | None = None
) -> tuple[tuple[ContentBlock, ...], str]:
    """优先用 MinerU 自己的分块（与 Markdown 对齐更好），否则退回 PDF 文本块。

    ``payload`` 是已加载的 content_list（调用方加载一次，图注抽取复用）。
    返回 ``(blocks, 来源标记)``，来源写入 ``stats`` 便于排障。
    """
    blocks = _blocks_from_content_list(payload, page_map)
    if blocks:
        return blocks, "content_list"
    blocks = extract_blocks(input_path)
    return blocks, "pdf_text" if blocks else "none"


_MINERU_TIERS = frozenset({"flash", "basic", "standard", "advanced"})
_MINERU_OCR_MODES = frozenset({"auto", "txt", "ocr"})
# MinerU 4 默认把版面图片内联成 data URI。图片另由 pymupdf 抽取，
# 内联字节进 Markdown 后会被切进正文。
_DATA_URI_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*data:image/[^)]+\)",
    re.IGNORECASE,
)


def _drop_embedded_images(markdown: str) -> str:
    """去掉 Markdown 里的 data URI 图片，保留周围文字。"""
    return _DATA_URI_IMAGE_RE.sub("", markdown)


def _mineru_tier() -> str:
    """把配置折成 MinerU 4 的解析档位。

    3.x 的 ``pipeline`` 是本地版面模型，对应 4.x 的 ``basic``。
    已经写成 flash/basic/standard/advanced 的值原样使用。
    """
    backend = str(MINERU_BACKEND or "").strip().lower()
    if backend in _MINERU_TIERS:
        return backend
    return "basic"


def _mineru_ocr_mode() -> str:
    method = str(MINERU_METHOD or "").strip().lower()
    if method in _MINERU_OCR_MODES:
        return method
    return "auto"


def _parse_binary_document(input_path: Path, output_dir: Path) -> Path | None:
    """用 MinerU 4 的本地 SDK 解析，并落下 Markdown 与 content_list。"""
    from mineru.parser import parse
    from mineru.render import RenderFormat, render

    result = parse(
        input_path,
        tier=_mineru_tier(),
        ocr_mode=_mineru_ocr_mode(),
        image_analysis=False,
    )
    # 空的 image_renderer 让图片块、表格回退图不再内联；图表仍可能带 data URI，下面再清一次。
    markdown = _drop_embedded_images(result.markdown(image_renderer=lambda _block: ""))
    if not isinstance(markdown, str) or not markdown.strip():
        return None
    target = output_dir / input_path.stem / "auto"
    target.mkdir(parents=True, exist_ok=True)
    markdown_path = target / f"{input_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    content_list = render(result.middle_json, RenderFormat.CONTENT_LIST)
    content_list_path = target / f"{input_path.stem}_content_list.json"
    content_list_path.write_text(
        json.dumps(content_list, ensure_ascii=False),
        encoding="utf-8",
    )
    return markdown_path


def parse_with_mineru(path: str | Path, *, source: str = "") -> ParseResult:
    """Parse one document with local MinerU and return its Markdown output."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"待解析文件不存在：{input_path}")
    if input_path.suffix.lower() in TEXT_SUFFIXES:
        # 纯文本走无损直读：编码必须可信，绝不用替换字符兜底
        try:
            text, encoding = read_text_file(input_path, source=source or input_path.name)
        except TextDecodeError as exc:
            raise MinerUParseError(str(exc)) from exc
        return ParseResult(
            markdown=text,
            stats={"input_type": "text", "text_encoding": encoding},
        )

    Path(MINERU_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="mineru-", dir=MINERU_OUTPUT_DIR))
    keep_dir = False
    try:
        markdown_path = _parse_binary_document(input_path, work_dir)
        if markdown_path is None:
            raise MinerUParseError(f"MinerU 未生成 Markdown 输出：{source or input_path.name}")
        # MinerU 自己写 UTF-8（编码契约已知），因此这里只做字形级清洗与
        # 「大面积字形映射失败」判定，不能套用纯文本文件的严格编码可信度门。
        markdown, parser_stats = decode_parser_output(
            markdown_path.read_bytes(), source=source or input_path.name
        )
        markdown = markdown.strip()
        if not markdown:
            raise MinerUParseError(f"MinerU 输出为空：{source or input_path.name}")
        # 注意：必须在 finally 删除 work_dir 之前读取 content_list。
        # content_list 只加载一次：分页文本块与图注抽取共用同一份 payload。
        content_list_payload = _load_content_list(markdown_path)
        page_map = detect_printed_pages_from_path(input_path)
        blocks, blocks_source = _extract_blocks(
            content_list_payload, input_path, page_map
        )
        images = extract_images(input_path)
        figure_captions = _figure_captions_from_content_list(
            content_list_payload, page_map
        )
        from mineru.version import __version__

        return ParseResult(
            markdown=markdown,
            parser_version=__version__,
            stats={
                "input_type": input_path.suffix.lower(),
                "output": str(markdown_path),
                "text_encoding": parser_stats["parser_encoding"],
                "symbol_glyph_count": parser_stats["symbol_glyph_count"],
                "unmapped_glyph_count": parser_stats["unmapped_glyph_count"],
                "control_char_count": parser_stats["control_char_count"],
                "blocks_source": blocks_source,
                "block_count": len(blocks),
                "image_count": len(images),
                "figure_caption_count": sum(len(v) for v in figure_captions.values()),
            },
            images=images,
            blocks=blocks,
            figure_captions=figure_captions,
        )
    except Exception as exc:
        keep_dir = MINERU_KEEP_ARTIFACTS_ON_ERROR
        if isinstance(exc, MinerUParseError):
            raise
        raise MinerUParseError(
            f"MinerU 解析失败（source={source or input_path.name}）：{exc}"
        ) from exc
    finally:
        if keep_dir:
            logger.warning("保留 MinerU 失败产物：%s", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


__all__ = [
    "MAX_UNMAPPED_GLYPH_RATIO",
    "MinerUParseError",
    "ParseResult",
    "ParserTextAudit",
    "TextDecodeError",
    "clean_parser_text",
    "decode_parser_output",
    "decode_text",
    "parse_with_mineru",
    "read_text_file",
]
