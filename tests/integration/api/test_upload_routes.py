"""两条上传路由的共享处理契约。

D-03 合并后 ``/documents/upload`` 与 ``/documents/upload-async`` 走同一
批处理实现，差异只剩指标与审计标签。这里钉住统一后的语义：

* 成功：返回全部 task_id，kb 原样透传，文件句柄全部关闭；
* 异常映射统一：``ValueError`` -> 400，流水线背压
  （``IngestionRejectedError``）-> 503，其余异常 -> 500；
* 任一失败都中止整批，但 ``finally`` 仍关闭全部上传句柄；
* 空文件列表直接 400。
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.routes import documents
from src.core.exceptions import IngestionRejectedError


class _FakeUpload:
    def __init__(self, filename: str):
        self.filename = filename
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeManager:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[list[str]] = []

    async def upload_documents_async(self, files):
        self.calls.append([f.filename for f in files])
        if self.error is not None:
            raise self.error
        return [{"task_id": f"t{i}", "filename": f.filename} for i, f in enumerate(files)]


def _use_manager(monkeypatch, error: Exception | None = None):
    manager = _FakeManager(error)
    audits: list[tuple[str, dict]] = []
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)
    monkeypatch.setattr(documents, "audit_log", lambda action, **kw: audits.append((action, kw)))
    return manager, audits


def _files(*names: str) -> list[_FakeUpload]:
    return [_FakeUpload(name) for name in names]


_ROUTES = (documents.upload_documents, documents.upload_documents_async)


async def test_upload_routes_share_submission_semantics(monkeypatch):
    manager, audits = _use_manager(monkeypatch)

    files_main = _files("a.md", "b.pdf")
    results = await documents.upload_documents(
        current_user="admin", files=files_main, kb="finance_report",
    )
    assert [r.task_id for r in results] == ["t0", "t1"]
    assert [r.filename for r in results] == ["a.md", "b.pdf"]
    assert manager.calls == [["a.md", "b.pdf"]]
    assert all(f.closed for f in files_main)
    assert audits[0][0] == "upload"
    assert audits[0][1]["user"] == "admin"

    files_alias = _files("c.txt")
    results = await documents.upload_documents_async(
        current_user="admin", files=files_alias, kb="finance_report",
    )
    assert [r.filename for r in results] == ["c.txt"]
    assert manager.calls[-1] == ["c.txt"]
    assert all(f.closed for f in files_alias)
    assert audits[1][0] == "upload_async"


@pytest.mark.parametrize("raised,status", [
    (ValueError("不支持的文件格式 .exe"), 400),
    (IngestionRejectedError("入库队列已满"), 503),
    (RuntimeError("意外错误"), 500),
])
async def test_upload_routes_map_errors_and_still_close_files(monkeypatch, raised, status):
    for route in _ROUTES:
        _use_manager(monkeypatch, error=raised)
        files = _files("a.md")

        with pytest.raises(HTTPException) as excinfo:
            await route(current_user="admin", files=files, kb="")

        assert excinfo.value.status_code == status
        assert all(f.closed for f in files), "失败路径也必须关闭全部句柄"


async def test_upload_routes_reject_unknown_kb(monkeypatch):
    _use_manager(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        await documents.upload_documents(
            current_user="admin", files=_files("a.md"), kb="arbitrary_collection",
        )
    assert excinfo.value.status_code == 400


async def test_upload_routes_reject_empty_file_list(monkeypatch):
    _use_manager(monkeypatch)

    for route in _ROUTES:
        with pytest.raises(HTTPException) as excinfo:
            await route(current_user="admin", files=[], kb="")

        assert excinfo.value.status_code == 400
