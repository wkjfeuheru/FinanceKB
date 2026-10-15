"""研报元数据人工修正。

抽取链路的最后一道兜底：正则与模型都会抽错。若没有人工修正手段，
元数据维度就形同虚设——写进去一个非法行业名只会让过滤静默失效。
"""

import copy
import threading

import pytest
from fastapi import HTTPException

from finance_rag.src.api.routes import documents
from finance_rag.src.schemas.document import DocumentMetadataPatch
from finance_rag.src.services.document_service import DocumentManager
from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase
from finance_rag.src.infrastructure.vector_store.fingerprint_store import FingerprintStore


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
                     "industry_l1": "食品饮料", "industry_l2": "白酒Ⅱ", "security_name": "",
                     "broker": "", "meta_source": "llm", "needs_review": False} for index in range(3)]

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
    manager._collection_name = "finance_kb"
    return manager


def test_valid_metadata_is_marked_manual(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    result = manager.update_document_metadata(
        "研报.pdf", {"industry_l1": "食品饮料",
                     "industry_l2": "白酒Ⅱ"}
    )

    _, sent = kb.calls[0]
    assert sent["industry_l2"] == "白酒Ⅱ"
    assert sent["meta_source"] == "manual"
    assert sent["needs_review"] is False
    assert result["updated_count"] == 3


def test_invalid_industry_is_rejected_not_silently_blanked(monkeypatch):
    kb = _FakeKB()
    manager = _manager(monkeypatch, kb)

    with pytest.raises(ValueError) as excinfo:
        manager.update_document_metadata("研报.pdf", {"industry_l1": "白酒"})

    assert "industry_l1" in str(excinfo.value)
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
    assert sent["industry_l1"] == "食品饮料"
    assert sent["industry_l2"] == "白酒Ⅱ"


async def test_endpoint_returns_422_for_invalid_industry(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)

    with pytest.raises(HTTPException) as excinfo:
        await documents.patch_document_metadata(
            "研报.pdf",
            DocumentMetadataPatch(industry_l1="白酒"),
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


@pytest.mark.parametrize(("field", "limit"), [("category", 64), ("date", 32)])
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


async def test_endpoint_date_patch_updates_existing_manifest_and_keeps_public_audit(monkeypatch, tmp_path):
    from datetime import datetime
    from sqlalchemy import insert, select
    from finance_rag.src.core.config import TENANT_ID
    from finance_rag.src.infrastructure.relational_db.document_manifest import DocumentManifestRepository, documents as manifest_documents
    from finance_rag.src.rag.ingestion.public_metadata import resolve_public_metadata
    kb = _FakeKB()
    kb.collection_name = "finance_kb"
    repo = DocumentManifestRepository("sqlite://")
    manifest_documents.create(repo._engine)
    with repo._engine.begin() as connection:
        connection.execute(insert(manifest_documents).values(document_id="one", tenant_id=TENANT_ID,
            source="研报.pdf", title="标题", date="2026-10-01", updated_at=datetime.now()))
    monkeypatch.setattr(kb, "_get_manifest_repository", lambda: repo)
    store = kb._get_fingerprint_store()
    store.update("研报.pdf", "original-pdf-sha", "研报.pdf", 3,
                 metadata={"public_metadata_audit": resolve_public_metadata("研报.pdf").audit})
    manager = _manager(monkeypatch, kb)
    monkeypatch.setattr(documents, "get_document_manager", lambda kb="": manager)
    await documents.patch_document_metadata("研报.pdf", DocumentMetadataPatch(date="2026-09-27"), current_user="admin")
    assert all(row["date"] == "2026-09-27" for row in kb.rows)
    with repo._engine.connect() as connection:
        assert connection.execute(select(manifest_documents.c.date)).scalar() == "2026-09-27"
    saved = FingerprintStore(str(store._path)).get("研报.pdf")
    assert saved["sha256"] == "original-pdf-sha"
    assert saved["public_metadata_audit"]["fields"]["date"]["status"] == "manual"
    assert saved["public_metadata_audit"]["fields"]["broker"]["status"] == "unknown"


def test_first_manual_patch_records_available_pdf_hash(monkeypatch):
    kb = _FakeKB()
    store = kb._get_fingerprint_store()
    store.update("研报.pdf", "original-pdf-sha", "研报.pdf", 3)
    _manager(monkeypatch, kb).update_document_metadata("研报.pdf", {"broker": "东吴证券"})
    assert store.get("研报.pdf")["public_metadata_audit"]["pdf_sha256"] == "original-pdf-sha"


def test_first_manual_patch_does_not_certify_untouched_legacy_date(monkeypatch):
    kb = _FakeKB()
    for row in kb.rows: row["date"] = "2026-10-01"
    _manager(monkeypatch, kb).update_document_metadata("东吴证券-2026-09-27.pdf", {"broker": "东吴证券"})
    saved = kb._get_fingerprint_store().get("东吴证券-2026-09-27.pdf")
    assert saved["public_metadata_audit"]["fields"]["date"]["status"] == "unknown"
