"""把 PDF 物理页映射成页面上印出来的页码。

封面、目录通常没有阿拉伯数字页码，记 0。正文从印出的 1 起。
检测失败时退回物理页（1..N），不猜。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER_FRAC = 0.12
_FOOTER_FRAC = 0.12
_YEAR_MIN, _YEAR_MAX = 1900, 2099
_ARABIC_RE = re.compile(
    r"^(?:第\s*)?(\d{1,3})(?:\s*页)?$|"
    r"^[-–—−]\s*(\d{1,3})\s*[-–—−]$"
)
_COMMON_SHARE = 0.5
_MIN_RUN = 2


def detect_printed_pages_from_path(path: str | Path) -> tuple[int, ...]:
    """按文件路径检测印刷页码；打不开时退回空元组。"""
    try:
        import pymupdf

        document = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001 - 页码映射失败不得阻断解析
        logger.warning("打开 PDF 失败，无法检测印刷页码（%s）：%s", path, exc)
        return ()
    try:
        return detect_printed_pages(document)
    finally:
        document.close()


def detect_printed_pages_from_bytes(payload: bytes) -> tuple[int, ...]:
    """从 PDF 字节检测印刷页码；无法解析时退回空元组。"""
    try:
        import pymupdf

        document = pymupdf.open(stream=payload, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        logger.warning("从字节打开 PDF 失败，无法检测印刷页码：%s", exc)
        return ()
    try:
        return detect_printed_pages(document)
    finally:
        document.close()


def detect_printed_pages(document: object) -> tuple[int, ...]:
    """对已打开的 pymupdf 文档检测印刷页码。"""
    page_count = int(getattr(document, "page_count", 0) or 0)
    if page_count <= 0:
        return ()
    physical = tuple(range(1, page_count + 1))
    labeled = _from_page_labels(document, page_count)
    if labeled is not None:
        return labeled
    detected = _from_header_footer(document, page_count)
    origin = _origin_from_detected(detected)
    if origin is None:
        return physical
    return tuple(0 if index < origin else index - origin + 1 for index in range(page_count))


def map_physical_page(physical: int, page_map: Sequence[int] | None) -> int:
    """物理页（1-based）→ 印刷页；无映射时保持物理页。"""
    if not isinstance(physical, int) or isinstance(physical, bool) or physical < 1:
        return 0
    if not page_map:
        return physical
    if physical > len(page_map):
        return physical
    return int(page_map[physical - 1])


def printed_to_physical(printed: int, page_map: Sequence[int]) -> int | None:
    """印刷页 → 第一张对应的物理页；找不到返回 None。"""
    if not isinstance(printed, int) or isinstance(printed, bool) or printed < 1:
        return None
    for index, value in enumerate(page_map):
        if value == printed:
            return index + 1
    return None


def _from_page_labels(document: object, page_count: int) -> tuple[int, ...] | None:
    """仅当 PageLabels 与默认物理编号不同时才采用。"""
    parsed: list[int | None] = []
    for index in range(page_count):
        try:
            raw = str(document[index].get_label() or "").strip()
        except Exception:  # noqa: BLE001
            return None
        parsed.append(_parse_page_token(raw))
    if all(value == index + 1 for index, value in enumerate(parsed)):
        return None
    origin = _origin_from_detected(parsed)
    if origin is None:
        return None
    return tuple(0 if index < origin else index - origin + 1 for index in range(page_count))


def _from_header_footer(document: object, page_count: int) -> list[int | None]:
    per_page: list[list[int]] = [[] for _ in range(page_count)]
    for index in range(page_count):
        try:
            page = document[index]
            height = float(page.rect.height)
            blocks = page.get_text("blocks")
        except Exception:  # noqa: BLE001
            continue
        header_limit = height * _HEADER_FRAC
        footer_limit = height * (1.0 - _FOOTER_FRAC)
        for block in blocks:
            if len(block) < 5 or not isinstance(block[4], str):
                continue
            y0, y1 = float(block[1]), float(block[3])
            center = (y0 + y1) / 2.0
            if header_limit < center < footer_limit:
                continue
            token = _parse_page_token(str(block[4]).strip())
            if token is not None:
                per_page[index].append(token)
    counts: dict[int, int] = {}
    for numbers in per_page:
        for number in set(numbers):
            counts[number] = counts.get(number, 0) + 1
    rare = {
        number
        for number, count in counts.items()
        if count == 1 or count / page_count < _COMMON_SHARE
    }
    detected: list[int | None] = []
    for numbers in per_page:
        unique = sorted({number for number in numbers if number in rare})
        detected.append(unique[0] if unique else None)
    return detected


def _origin_from_detected(detected: Sequence[int | None]) -> int | None:
    """连续 1,2,3… 的最长 run 起点（0-based）；不够长则放弃。"""
    best_start: int | None = None
    best_length = 0
    index = 0
    length = len(detected)
    while index < length:
        if detected[index] != 1:
            index += 1
            continue
        cursor = index + 1
        expected = 2
        while cursor < length and detected[cursor] == expected:
            expected += 1
            cursor += 1
        run = cursor - index
        if run > best_length:
            best_start, best_length = index, run
        index = cursor
    if best_start is None or best_length < _MIN_RUN:
        return None
    return best_start


def _parse_page_token(text: str) -> int | None:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return None
    match = _ARABIC_RE.fullmatch(compact)
    if not match:
        return None
    value = int(next(group for group in match.groups() if group))
    if _YEAR_MIN <= value <= _YEAR_MAX:
        return None
    if value < 1:
        return None
    return value


__all__ = [
    "detect_printed_pages",
    "detect_printed_pages_from_bytes",
    "detect_printed_pages_from_path",
    "map_physical_page",
    "printed_to_physical",
]
