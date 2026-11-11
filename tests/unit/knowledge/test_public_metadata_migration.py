"""文档元数据 PATCH 的事务与补偿：执行真实 KB、指纹和 SQL 事务，Milvus 仅替换传输层。

公共元数据迁移流程（preflight/apply/verify 脚本与 audit/needs_review 概念）已删除，
本文件只保留仍然成立的事务语义：逐行校验、manifest 联动与失败补偿。
"""
import copy
import threading

import pytest
from sqlalchemy import insert, inspect, select

from src.core.config import REPORT_COLLECTION_NAME, TENANT_ID
from src.storage.relational_db.document_manifest import DocumentManifestRepository, documents
from src.index.fingerprint_store import FingerprintStore
from src.index.milvus_kb import KnowledgeBase


class RowsClient:
    def __init__(self):
        self.rows = [dict(id=str(i), source="报告.pdf", tenant_id=TENANT_ID,
                         report_type="宏观", date="2026-10-01", broker="旧机构",
                         content="保留正文", dense_vector=[0.6, 0.8], content_hash="same") for i in range(2)]
        self.fail_writes = 0

    def query(self, **kwargs):
        import re
        match = re.search(r'source == "([^"]+)"', kwargs.get("filter", ""))
        return copy.deepcopy([row for row in self.rows if not match or row["source"] == match.group(1)])

    def upsert(self, *, collection_name, data):
        assert collection_name == REPORT_COLLECTION_NAME
        if self.fail_writes:
            self.fail_writes -= 1
            self.rows[0] = copy.deepcopy(data[0])
            raise RuntimeError("partial transport failure")
        values = {row["id"]: row for row in self.rows}
        values.update({row["id"]: copy.deepcopy(row) for row in data})
        self.rows = list(values.values())

    def flush(self, name):
        pass


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection_name = REPORT_COLLECTION_NAME
    kb._write_lock = threading.RLock()
    kb._fingerprint_store = FingerprintStore(str(tmp_path / "fingerprints.json"))
    kb._fingerprint_store.update("报告.pdf", "pdf-sha", "报告.pdf", 2, simhash=12)
    client = RowsClient()
    repo = DocumentManifestRepository("sqlite://")
    monkeypatch.setattr(kb, "_get_client", lambda: client)
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_collection_has_field", lambda *args: True)
    monkeypatch.setattr(kb, "_get_manifest_repository", lambda: repo, raising=False)
    return kb, client, repo


def _patch(kb, values=None):
    """人工修正入口：只带需要改的字段，签名与新契约一致（无 update_source/metadata_audit）。"""
    return kb.update_document_metadata("报告.pdf",
                                       values or {"date": "2026-09-27", "broker": "东吴证券"})


@pytest.mark.parametrize("kind", ["bad_second_row", "illegal_field"])
def test_patch_validates_every_row_before_any_write(prepared, kind):
    kb, client, _ = prepared
    values = {"date": "2026-09-27", "broker": "东吴证券"}
    if kind == "bad_second_row": client.rows[1]["report_type"] = "错误"
    if kind == "illegal_field": values["security_name"] = "贵州茅台"  # 已删除字段按未知字段拒绝
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError): _patch(kb, values)
    assert client.rows == before


def test_manifest_updates_only_existing_tenant_source(prepared):
    from datetime import datetime
    kb, client, repo = prepared
    documents.create(repo._engine)
    with repo._engine.begin() as conn:
        conn.execute(insert(documents), [dict(document_id=key, tenant_id=tenant, source="报告.pdf", title="原标题", date="2026-10-01", updated_at=datetime.now())
                                        for key, tenant in [("mine", TENANT_ID), ("other", "another")]])
    result = _patch(kb)
    with repo._engine.connect() as conn:
        values = dict(conn.execute(select(documents.c.document_id, documents.c.date)).all())
    assert values == {"mine": "2026-09-27", "other": "2026-10-01"}
    assert result["manifest"]["status"] == "updated"
    assert inspect(repo._engine).get_table_names() == ["documents"]


@pytest.mark.parametrize("failure_count", [1, 2])
def test_partial_write_is_compensated_or_reported_as_compensation_failure(prepared, failure_count):
    kb, client, _ = prepared
    rows_before, fp_before = copy.deepcopy(client.rows), kb._fingerprint_store.get("报告.pdf")
    client.fail_writes = failure_count
    with pytest.raises(RuntimeError) as error: _patch(kb)
    assert error.value.report["status"] == "failed"
    assert error.value.report["compensation"]["milvus"] == ("restored" if failure_count == 1 else "failed")
    if failure_count == 1: assert client.rows == rows_before
    assert kb._fingerprint_store.get("报告.pdf") == fp_before


def test_fingerprint_persistence_failure_rolls_back_sql_and_rows(prepared, monkeypatch):
    from datetime import datetime
    kb, client, repo = prepared
    documents.create(repo._engine)
    with repo._engine.begin() as conn:
        conn.execute(insert(documents).values(document_id="mine", tenant_id=TENANT_ID, source="报告.pdf", title="标题", date="2026-10-01", updated_at=datetime.now()))
    rows_before = copy.deepcopy(client.rows)
    save = kb._fingerprint_store._save
    calls = []
    def fail_once():
        calls.append(1)
        if len(calls) == 1: raise OSError("disk full")
        save()
    monkeypatch.setattr(kb._fingerprint_store, "_save", fail_once)
    with pytest.raises(RuntimeError): _patch(kb)
    assert client.rows == rows_before
    with repo._engine.connect() as conn:
        assert conn.execute(select(documents.c.date)).scalar() == "2026-10-01"


def test_manifest_connection_failure_is_not_missing_table(prepared, monkeypatch):
    kb, client, repo = prepared
    before = copy.deepcopy(client.rows)
    def unavailable(): raise ConnectionError("PG offline")
    monkeypatch.setattr(repo._engine, "begin", unavailable)
    with pytest.raises(RuntimeError, match="PG offline"): _patch(kb)
    assert client.rows == before


def test_manifest_compensation_retains_explicit_recovery_outcome(prepared, monkeypatch):
    kb, client, _ = prepared
    client.fail_writes = 1
    with pytest.raises(RuntimeError) as error: _patch(kb)
    assert error.value.report["compensation"]["manifest"] == "skipped_missing_table"


def test_fingerprint_compensation_failure_is_not_reported_restored(prepared, monkeypatch):
    kb, client, _ = prepared
    def unavailable(): raise OSError("persistent disk full")
    monkeypatch.setattr(kb._fingerprint_store, "_save", unavailable)
    with pytest.raises(RuntimeError) as error: _patch(kb)
    assert error.value.report["compensation"]["fingerprint"] == "failed"
    assert "disk full" in error.value.report["compensation"]["fingerprint_error"]


def test_sql_compensation_refuses_concurrent_new_date(prepared):
    from datetime import datetime
    kb, _, repo = prepared
    documents.create(repo._engine)
    with repo._engine.begin() as connection:
        connection.execute(insert(documents).values(document_id="mine", tenant_id=TENANT_ID, source="报告.pdf", title="标题", date="2026-10-02", updated_at=datetime.now()))
    with pytest.raises(ValueError, match="changed during compensation"):
        repo.restore_metadata_dates(TENANT_ID, "报告.pdf", [{"document_id": "mine", "date": "2026-10-01"}], "2026-09-27")
    with repo._engine.connect() as connection:
        assert connection.execute(select(documents.c.date)).scalar() == "2026-10-02"
