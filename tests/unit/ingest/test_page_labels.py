"""印刷页码检测：封面/目录为 0，正文从页脚或 PageLabels 的 1 起。"""

from pathlib import Path

import pymupdf

from src.ingest.page_labels import (
    detect_printed_pages,
    detect_printed_pages_from_path,
    map_physical_page,
    printed_to_physical,
)


def _workdir(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / ".tmp" / "page_labels" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pdf_with_footer_numbers(*, covers: int, body: int) -> bytes:
    """前 ``covers`` 页无页脚数字，随后正文页脚为 1, 2, …。"""
    document = pymupdf.open()
    for index in range(covers):
        page = document.new_page()
        page.insert_text((72, 72), f"cover {index + 1}")
    for index in range(body):
        page = document.new_page()
        page.insert_text((72, 72), f"body {index + 1}")
        page.insert_text((300, 800), str(index + 1))
    payload = document.tobytes()
    document.close()
    return payload


def test_footer_numbers_skip_unnumbered_front_matter(tmp_path=None):
    payload = _pdf_with_footer_numbers(covers=2, body=3)
    path = _workdir("front-matter") / "report.pdf"
    path.write_bytes(payload)

    assert detect_printed_pages_from_path(path) == (0, 0, 1, 2, 3)


def test_missing_footers_fall_back_to_physical_pages():
    document = pymupdf.open()
    for index in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1} text")
    mapping = detect_printed_pages(document)
    document.close()

    assert mapping == (1, 2, 3)


def test_map_and_reverse_printed_pages():
    page_map = (0, 0, 1, 2, 3)

    assert map_physical_page(1, page_map) == 0
    assert map_physical_page(3, page_map) == 1
    assert map_physical_page(5, page_map) == 3
    assert map_physical_page(3, None) == 3
    assert printed_to_physical(1, page_map) == 3
    assert printed_to_physical(3, page_map) == 5
    assert printed_to_physical(1, (1, 2, 3)) == 1
    assert printed_to_physical(0, page_map) is None
    assert printed_to_physical(9, page_map) is None
