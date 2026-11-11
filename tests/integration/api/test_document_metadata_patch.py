"""研报元数据人工修正。

抽取链路的最后一道兜底：正则与模型都会抽错。若没有人工修正手段，
元数据维度就形同虚设——写进去一个非法报告类型只会让过滤静默失效。
"""

import copy
import threading

import pytest
from fastapi import HTTPException

from src.api.routes import documents
from src.api.schemas.document import DocumentMetadataPatch
from src.ingest.documents import DocumentManager
from src.index.milvus_kb import KnowledgeBase
from src.index.fingerprint_store import FingerprintStore


@pytest.fixture(autouse=True)
def isolated_fingerprints(tmp_path, monkeypatch):
    store = FingerprintStore(str(tmp_path / "fp.json"))
    monkeypatch.setattr(KnowledgeBase, "_get_fingerprint_store", lambda self: store)


class _FakeKB(KnowledgeBase):
    def __init__(self, *, missing: bool = False, has_report_fields: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self._missing = missing
        self._has_report_fields = has_report_fields
        self.collection_name = "test"
        self._write_lock = threading.RLock()
        self.rows = [] if missing else [{"id": str(index), "report_type": "行业",
                     "broker": ""} for index in range(3)]

    def _get_client(self):
        return self

    def ensure_collection(self):
        pass

    def _collection_has_field(self, client, name):
        return self._has_report_fields

    def query(self, **kwargs):
        return copy.deepcopy(self.rows)

    def upsert(self, *, collection_name, data):
        self.rows = copy.deepcopy(data)
        self.calls.append(("研报.pdf", data[0]))

    def flush(self, name):
        pass


def _manager(monkeypatch, kb: _FakeKB | None = None) -> DocumentManager:
    """注入假 KB：``DocumentManager.kb`` 是只读属性，这里直接构造时传入。"""
    manager = DocumentManager.__new__(DocumentManager)
    manager._kb = kb or _FakeKB()
    manager._collection_name = "finance_report"
    return manager


def test_valid_broker_patch_is_written_to_all_rows(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    result = manager.update_document_metadata("研报.pdf", {"broker": "中信证券"})

    _, sent = kb.calls[0]
    assert sent["broker"] == "中信证券"
    assert result["updated_count"] == 3


def test_invalid_report_type_is_rejected_not_silently_blanked(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    with pytest.raises(ValueError) as excinfo:
        manager.update_document_metadata("研报.pdf", {"report_type": "点评"})

    assert "report_type" in str(excinfo.value)
    assert kb.calls == []          # 非法值不得写库


def test_empty_string_clears_a_field(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    manager.update_document_metadata("研报.pdf", {"broker": ""})

    _, sent = kb.calls[0]
    assert sent["broker"] == ""


def test_untouched_fields_are_preserved(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    manager.update_document_metadata("研报.pdf", {"broker": "中信证券"})

    _, sent = kb.calls[0]
    assert sent["report_type"] == "行业"


async def test_endpoint_returns_422_for_invalid_report_type(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf",
            DocumentMetadataPatch(report_type="点评"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 422


async def test_endpoint_returns_404_when_document_missing(monkeypatch):
    manager = _manager(monkeypatch, _FakeKB(missing=True))
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "缺失.pdf",
            DocumentMetadataPatch(broker="中信证券"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 404


async def test_endpoint_rejects_empty_payload(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf", DocumentMetadataPatch(), current_user="admin"
        )

    assert excinfo.value.status_code == 400


async def test_endpoint_surfaces_stale_schema_as_500(monkeypatch):
    manager = _manager(monkeypatch, _FakeKB(has_report_fields=False))
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf",
            DocumentMetadataPatch(broker="中信证券"),
            current_user="admin",
        )

    assert excinfo.value.status_code == 500
    assert "rebuild_collection" in str(excinfo.value.detail)


@pytest.mark.parametrize(("field", "limit"), [("date", 32)])
async def test_endpoint_returns_422_for_overlong_public_metadata(monkeypatch, field, limit):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf", DocumentMetadataPatch(**{field: "a" * (limit + 1)}), current_user="admin"
        )

    assert excinfo.value.status_code == 422
    assert field in str(excinfo.value.detail)
    assert kb.calls == []


async def test_endpoint_date_patch_updates_existing_manifest(monkeypatch, tmp_path):
    from datetime import datetime
    from sqlalchemy import insert, select
    from src.core.config import TENANT_ID
    from src.storage.relational_db.document_manifest import DocumentManifestRepository, documents as manifest_documents
    kb = _FakeKB()
    kb.collection_name = "finance_report"
    repo = DocumentManifestRepository("sqlite://")
    manifest_documents.create(repo._engine)
    with repo._engine.begin() as connection:
        connection.execute(insert(manifest_documents).values(document_id="one", tenant_id=TENANT_ID,
            source="研报.pdf", title="标题", date="2026-10-01", updated_at=datetime.now()))
    monkeypatch.setattr(kb, "_get_manifest_repository", lambda: repo)
    store = kb._get_fingerprint_store()
    store.update("研报.pdf", "original-pdf-sha", "研报.pdf", 3)
    manager = _manager(monkeypatch, kb)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)
    await documents.patch_document_metadata("研报.pdf", DocumentMetadataPatch(date="2026-09-27"), current_user="admin")
    assert all(row["date"] == "2026-09-27" for row in kb.rows)
    with repo._engine.connect() as connection:
        assert connection.execute(select(manifest_documents.c.date)).scalar() == "2026-09-27"
    # 元数据修正不触碰文件指纹（否则下次上传会被误判为「已变更」而全量重跑）
    saved = FingerprintStore(str(store._path)).get("研报.pdf")
    assert saved["sha256"] == "original-pdf-sha"
