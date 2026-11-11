"""原文页渲染接口。

页码级溯源的价值全在「能跳」：给出页码 → 立刻看到那一页。这条链路直接读
对象存储里的原始 PDF 并按需渲染，不落临时文件。
"""

import pymupdf
import pytest
from fastapi import HTTPException

from src.api.routes import documents
from src.core.config import PAGE_RENDER_DPI


def _pdf_bytes(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    payload = document.tobytes()
    document.close()
    return payload


class _Storage:
    """对象存储替身：只提供指定键，其余抛 FileNotFoundError（模拟真实后端）。"""

    def __init__(self, payload: bytes | None, *, available: set[str] | None = None):
        self._payload = payload
        self._available = available
        self.requested: list[str] = []

    async def download(self, key: str) -> bytes:
        self.requested.append(key)
        if self._payload is None or (self._available is not None and key not in self._available):
            raise FileNotFoundError(key)
        return self._payload


def _use_storage(
    monkeypatch, payload: bytes | None, *, available: set[str] | None = None
) -> _Storage:
    storage = _Storage(payload, available=available)
    monkeypatch.setattr(documents, "get_storage", lambda: storage)
    return storage


async def test_render_page_uses_namespaced_key(monkeypatch):
    """默认走隔离键：租户/集合/文档身份/版本四段分离，不再与解析产物撞键。"""
    storage = _use_storage(monkeypatch, _pdf_bytes(pages=2))

    response = await documents.render_document_page(
        "研报.pdf", 2, current_user="admin"
    )

    assert response.media_type == "image/png"
    assert response.body[:4] == b"\x89PNG"
    assert len(storage.requested) == 1
    key = storage.requested[0]
    assert key != "docs/研报.pdf", "不得继续读没有集合/版本隔离的旧键"
    assert key.endswith("/original/研报.pdf")
    assert "/current/" in key


async def test_render_page_falls_back_to_legacy_key(monkeypatch):
    """升级前的对象仍可预览：新键缺失时逐级回退到旧键（RAG-06 兼容要求）。"""
    storage = _use_storage(
        monkeypatch, _pdf_bytes(pages=2), available={"docs/研报.pdf"}
    )

    response = await documents.render_document_page(
        "研报.pdf", 1, current_user="admin"
    )

    assert response.body[:4] == b"\x89PNG"
    assert storage.requested[-1] == "docs/研报.pdf"
    assert len(storage.requested) > 1, "必须先尝试新键再回退"


def _pdf_with_printed_front_matter() -> bytes:
    """2 页封面无页脚 + 3 页正文页脚 1/2/3。"""
    document = pymupdf.open()
    for index in range(2):
        page = document.new_page()
        page.insert_text((72, 72), f"cover {index + 1}")
    for index in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"body {index + 1}")
        page.insert_text((300, 800), str(index + 1))
    payload = document.tobytes()
    document.close()
    return payload


async def test_render_printed_page_one_is_body_not_cover(monkeypatch):
    """请求印刷第 1 页应渲染正文，而不是封面。"""
    payload = _pdf_with_printed_front_matter()
    _use_storage(monkeypatch, payload)

    response = await documents.render_document_page(
        "研报.pdf", 1, current_user="admin"
    )

    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        body = document[2].get_pixmap(dpi=PAGE_RENDER_DPI).tobytes("png")
        cover = document[0].get_pixmap(dpi=PAGE_RENDER_DPI).tobytes("png")
    finally:
        document.close()

    assert response.body == body
    assert response.body != cover


async def test_render_unknown_printed_page_returns_404(monkeypatch):
    """印刷页码不存在时 404，不按物理页猜测。"""
    _use_storage(monkeypatch, _pdf_with_printed_front_matter())

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("研报.pdf", 5, current_user="admin")

    assert excinfo.value.status_code == 404


async def test_render_page_out_of_range_returns_404(monkeypatch):
    _use_storage(monkeypatch, _pdf_bytes(pages=2))

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("研报.pdf", 99, current_user="admin")

    assert excinfo.value.status_code == 404


async def test_render_page_rejects_zero_and_negative(monkeypatch):
    _use_storage(monkeypatch, _pdf_bytes())

    for page in (0, -3):
        with pytest.raises(HTTPException) as excinfo:
            await documents.render_document_page("研报.pdf", page, current_user="admin")
        assert excinfo.value.status_code == 400


async def test_missing_original_returns_404(monkeypatch):
    _use_storage(monkeypatch, None)

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("缺失.pdf", 1, current_user="admin")

    assert excinfo.value.status_code == 404


async def test_non_pdf_original_returns_400(monkeypatch):
    _use_storage(monkeypatch, "这不是 PDF".encode("utf-8"))

    with pytest.raises(HTTPException) as excinfo:
        await documents.render_document_page("笔记.md", 1, current_user="admin")

    assert excinfo.value.status_code == 400
