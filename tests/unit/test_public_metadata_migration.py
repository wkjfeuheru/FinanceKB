"""公共标量迁移：执行真实 KB、指纹和 SQL 事务，Milvus 仅替换传输层。"""
import copy
import threading

import pytest
from sqlalchemy import create_engine, insert, inspect, select

from finance_rag.src.core.config import TENANT_ID
from finance_rag.src.infrastructure.relational_db.document_manifest import DocumentManifestRepository, documents
from finance_rag.src.infrastructure.vector_store.fingerprint_store import FingerprintStore
from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase
from finance_rag.src.rag.ingestion.public_metadata import resolve_public_metadata


class RowsClient:
    def __init__(self):
        self.rows = [dict(id=str(i), source="报告.pdf", tenant_id=TENANT_ID,
                         report_type="宏观", security_name="", industry_l1="", industry_l2="",
                         date="2026-10-01", broker="旧机构", meta_source="regex", needs_review=False,
                         content="保留正文", dense_vector=[0.6, 0.8], content_hash="same") for i in range(2)]
        self.fail_writes = 0

    def query(self, **kwargs):
        import re
        match = re.search(r'source == "([^"]+)"', kwargs.get("filter", ""))
        return copy.deepcopy([row for row in self.rows if not match or row["source"] == match.group(1)])

    def upsert(self, *, collection_name, data):
        assert collection_name == "finance_kb"
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
    kb.collection_name = "finance_kb"
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


def audit():
    result = resolve_public_metadata("报告.pdf", "发布日期：2026-09-27\n发布机构：东吴证券")
    return {**result.audit, "pdf_sha256": "pdf-sha"}


def public_update(kb, values=None, metadata_audit=None):
    return kb.update_document_metadata("报告.pdf", values or {"date": "2026-09-27", "broker": "东吴证券"},
                                       update_source="public_metadata", metadata_audit=metadata_audit or audit())


def test_public_update_preserves_vectors_and_fingerprint_and_persists_audit(prepared):
    kb, client, _ = prepared
    before = copy.deepcopy(client.rows)
    result = public_update(kb)
    assert result["updated_count"] == 2
    assert result["manifest"]["status"] == "skipped_missing_table"
    for old, row in zip(before, client.rows):
        assert row == {**old, "date": "2026-09-27", "broker": "东吴证券", "meta_source": "public"}
    saved = FingerprintStore(str(kb._fingerprint_store._path)).get("报告.pdf")
    assert saved["sha256"] == "pdf-sha" and saved["simhash"] == 12
    assert saved["public_metadata_audit"] == audit()


def test_unknown_date_clears_only_public_fields_and_keeps_type_review(prepared):
    kb, client, _ = prepared
    kb._fingerprint_store._data["报告.pdf"]["type_metadata_audit"] = {"industry_l2": {"status": "unknown"}}
    unknown = resolve_public_metadata("报告.pdf").audit
    unknown["pdf_sha256"] = "pdf-sha"
    public_update(kb, {"date": "", "broker": ""}, unknown)
    assert all(row["date"] == "" and row["needs_review"] for row in client.rows)
    assert kb._fingerprint_store.get("报告.pdf")["type_metadata_audit"] == {"industry_l2": {"status": "unknown"}}


def test_manual_patch_reads_trusted_audit_inside_lock(prepared):
    kb, client, _ = prepared
    public_update(kb)
    unknown = resolve_public_metadata("报告.pdf").audit
    class ArrivingLock:
        def __enter__(self):
            kb._fingerprint_store._data["报告.pdf"]["public_metadata_audit"] = unknown
            kb._fingerprint_store._data["报告.pdf"]["type_metadata_audit"] = {"industry_l2": {"status": "unknown"}}
        def __exit__(self, *args):
            pass
    kb._write_lock = ArrivingLock()
    result = kb.update_document_metadata("报告.pdf", {"broker": "中信证券"})
    saved = kb._fingerprint_store.get("报告.pdf")
    assert saved["public_metadata_audit"]["fields"]["date"]["status"] == "unknown"
    assert saved["public_metadata_audit"]["fields"]["broker"]["status"] == "manual"
    assert result["metadata"]["needs_review"] and client.rows[0]["meta_source"] == "manual"


@pytest.mark.parametrize("kind", ["manual", "bad_second_row", "illegal_field", "audit_mismatch"])
def test_public_update_validates_every_row_before_any_write(prepared, kind):
    kb, client, _ = prepared
    values = {"date": "2026-09-27", "broker": "东吴证券"}
    if kind == "manual": client.rows[1]["meta_source"] = "manual"
    if kind == "bad_second_row": client.rows[1]["report_type"] = "错误"
    if kind == "illegal_field": values["security_name"] = "贵州茅台"
    if kind == "audit_mismatch": values["date"] = "2026-09-28"
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError): public_update(kb, values)
    assert client.rows == before


def test_manifest_updates_only_existing_tenant_source(prepared):
    from datetime import datetime
    kb, client, repo = prepared
    documents.create(repo._engine)
    with repo._engine.begin() as conn:
        conn.execute(insert(documents), [dict(document_id=key, tenant_id=tenant, source="报告.pdf", title="原标题", date="2026-10-01", updated_at=datetime.now())
                                        for key, tenant in [("mine", TENANT_ID), ("other", "another")]])
    result = public_update(kb)
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
    with pytest.raises(RuntimeError) as error: public_update(kb)
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
    with pytest.raises(RuntimeError): public_update(kb)
    assert client.rows == rows_before
    with repo._engine.connect() as conn:
        assert conn.execute(select(documents.c.date)).scalar() == "2026-10-01"


def test_first_fingerprint_update_keeps_audits_across_incremental_update(tmp_path):
    store = FingerprintStore(str(tmp_path / "fp.json"))
    store.update("报告.pdf", "sha", "报告.pdf", 2, metadata={"public_metadata_audit": audit(), "type_metadata_audit": {"industry_l2": {"status": "unknown"}}})
    store.update("报告.pdf", "sha", "报告.pdf", 2)
    saved = FingerprintStore(str(store._path)).get("报告.pdf")
    assert saved["public_metadata_audit"] == audit()
    assert saved["type_metadata_audit"] == {"industry_l2": {"status": "unknown"}}


@pytest.fixture
def migration_case(prepared, tmp_path, monkeypatch):
    import gzip
    import json
    import hashlib
    import pymupdf
    from scripts import repair_report_metadata as migration
    kb, client, repo = prepared
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((50, 50), "Research report")
    page.insert_text((50, 100), "2026-09-27")
    document.save(str(pdf_root / "报告.pdf"))
    document.close()
    for row in client.rows:
        row["dense_vector"] = [1.0] + [0.0] * 1023
        row["parent_id"] = ""
    sha = hashlib.sha256((pdf_root / "报告.pdf").read_bytes()).hexdigest()
    kb._fingerprint_store.update("报告.pdf", sha, "报告.pdf", 2)
    snapshot = tmp_path / "before"
    snapshot.mkdir()
    artifacts = {}
    for filename, rows in [("vectors.jsonl.gz", client.rows), ("parent_chunks.jsonl.gz", []), ("table_chunks.jsonl.gz", [])]:
        with gzip.open(snapshot / filename, "wt", encoding="utf8") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
        artifacts[filename] = {"count": len(rows)}
    for filename, values in [("schema.json", {"fields": [{"name": key} for key in client.rows[0]]}),
                              ("fingerprints.json", kb._fingerprint_store._data), ("pdf_manifest.json", {"报告.pdf": sha})]:
        (snapshot / filename).write_text(json.dumps(values), encoding="utf8")
        artifacts[filename] = {}
    for filename in artifacts: artifacts[filename]["sha256"] = hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps({"collection": "finance_kb", "source_count": 1, "artifacts": artifacts}), encoding="utf8")
    monkeypatch.setattr(migration, "read_live_rows", lambda kb: copy.deepcopy(client.rows))
    monkeypatch.setattr(migration, "read_derived", lambda: {"parent_chunks": [], "table_chunks": []})
    return migration, kb, client, snapshot, pdf_root, tmp_path


def test_preflight_is_readonly_and_proposal_is_immutable(migration_case):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    before = copy.deepcopy(client.rows)
    result = migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    assert result["source_count"] == 1 and result["sources"][0]["patch"] == {"date": "2026-09-27", "broker": ""}
    assert client.rows == before
    with pytest.raises(FileExistsError): migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)


@pytest.mark.parametrize("drift", ["second_scalar", "vector", "pdf", "extra_source", "fingerprint"])
def test_apply_refuses_stale_proposal_before_writing(migration_case, drift):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    if drift == "second_scalar": client.rows[1]["broker"] = "人工并发值"
    if drift == "vector": client.rows[1]["dense_vector"] = [0.0, 1.0] + [0.0] * 1022
    if drift == "pdf": (pdf_root / "报告.pdf").write_bytes(b"changed")
    if drift == "extra_source": client.rows.append({**client.rows[0], "id": "new", "source": "new.pdf"})
    if drift == "fingerprint": kb._fingerprint_store._data["报告.pdf"]["simhash"] = 34
    before = copy.deepcopy(client.rows)
    result = migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)
    assert result["status"] == "failed"
    assert client.rows == before


def test_apply_then_verify_preserves_identity_and_updates_public_fields(migration_case):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    result = migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)
    assert result["status"] == "ok"
    assert client.rows[0]["date"] == "2026-09-27" and client.rows[0]["broker"] == ""
    assert client.rows[0]["needs_review"] is True
    verified = migration.verify(snapshot, root / "verify.json", kb=kb, pdf_root=pdf_root)
    assert verified["status"] == "ok"
    client.rows[1]["content"] = "corrupted"
    assert migration.verify(snapshot, root / "verify-bad.json", kb=kb, pdf_root=pdf_root)["status"] == "failed"


def test_manifest_connection_failure_is_not_missing_table(prepared, monkeypatch):
    kb, client, repo = prepared
    before = copy.deepcopy(client.rows)
    def unavailable(): raise ConnectionError("PG offline")
    monkeypatch.setattr(repo._engine, "begin", unavailable)
    with pytest.raises(RuntimeError, match="PG offline"): public_update(kb)
    assert client.rows == before


def test_manifest_drift_is_checked_under_sql_lock_before_vector_write(prepared):
    from datetime import datetime
    kb, client, repo = prepared
    documents.create(repo._engine)
    with repo._engine.begin() as conn:
        conn.execute(insert(documents).values(document_id="mine", tenant_id=TENANT_ID, source="报告.pdf", title="标题", date="2026-10-01", updated_at=datetime.now()))
    proposed_audit = {**audit(), "migration_manifest_before": {"status": "skipped_missing_table", "rows": []}}
    before = copy.deepcopy(client.rows)
    with pytest.raises(RuntimeError, match="manifest.*drift"):
        public_update(kb, metadata_audit=proposed_audit)
    assert client.rows == before


def test_verify_refuses_wrong_pdf_audit_even_when_scalars_match(migration_case):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    assert migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)["status"] == "ok"
    kb._fingerprint_store._data["报告.pdf"]["public_metadata_audit"]["pdf_sha256"] = "different-pdf"
    assert migration.verify(snapshot, root / "verify-bad-audit.json", kb=kb, pdf_root=pdf_root)["status"] == "failed"


def test_manifest_compensation_retains_explicit_recovery_outcome(prepared, monkeypatch):
    kb, client, _ = prepared
    client.fail_writes = 1
    with pytest.raises(RuntimeError) as error: public_update(kb)
    assert error.value.report["compensation"]["manifest"] == "skipped_missing_table"


def test_public_known_fields_do_not_clear_untracked_review_on_manual_patch(prepared):
    kb, client, _ = prepared
    for row in client.rows: row["needs_review"] = True
    result = kb.update_document_metadata("报告.pdf", {"date": "2026-09-27", "broker": "东吴证券"})
    assert result["metadata"]["needs_review"] is True


def test_ingestion_keeps_canonical_audits_for_later_patch(tmp_path, monkeypatch):
    from tests.unit.test_ingestion_pipeline import _FakeMilvusClient, _prepare_kb, _chunked_document
    client = _FakeMilvusClient()
    kb = _prepare_kb(monkeypatch, client)
    class DerivedRows:
        def delete_by_source(self, source):
            pass
        def store_batch(self, rows):
            pass
    monkeypatch.setattr(kb, "_get_parent_store", lambda: DerivedRows())
    monkeypatch.setattr(kb, "_get_table_store", lambda: DerivedRows())
    store = FingerprintStore(str(tmp_path / "new-fp.json"))
    monkeypatch.setattr(kb, "_get_fingerprint_store", lambda: store)
    parsed = _chunked_document("报告.pdf", ["正文"])
    parsed.metadata.update(public_metadata_audit=audit(), type_metadata_audit={"industry_l2": {"status": "unknown"}})
    kb.add_parsed_document(parsed, content_hash="pdf-sha", dense_vectors=[[1.0, 0.0, 0.0]])
    saved = FingerprintStore(str(store._path)).get("报告.pdf")
    assert saved["public_metadata_audit"] == audit()
    assert saved["type_metadata_audit"] == {"industry_l2": {"status": "unknown"}}
    assert client.inserted[0]["needs_review"] is True


@pytest.mark.parametrize("state", ["manual", "conflict"])
def test_manual_or_conflicting_proposal_requires_review_without_changes(migration_case, state, monkeypatch):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    if state == "manual":
        # 模拟 before 本来就是人工审核；重建测试快照摘要，不篡改生产提案。
        import gzip, json, hashlib
        client.rows[1]["meta_source"] = "manual"
        with gzip.open(snapshot / "vectors.jsonl.gz", "wt", encoding="utf8") as f:
            for row in client.rows: f.write(json.dumps(row) + "\n")
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf8"))
        manifest["artifacts"]["vectors.jsonl.gz"]["sha256"] = hashlib.sha256((snapshot / "vectors.jsonl.gz").read_bytes()).hexdigest()
        (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    else:
        # 解析边界替身返回明确冲突，真实 source_proposal/apply 门禁仍执行。
        from finance_rag.src.rag.ingestion.public_metadata import PublicMetadataResult
        actual_resolver = migration.resolve_public_metadata
        def conflict(*args, **kwargs):
            found = actual_resolver(*args, **kwargs)
            found.audit["fields"]["date"].update(status="conflict", value="")
            return PublicMetadataResult("", found.broker, True, found.audit)
        monkeypatch.setattr(migration, "resolve_public_metadata", conflict)
    before = copy.deepcopy(client.rows)
    proposal = migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    assert proposal["sources"][0]["status"] == state + "_review"
    result = migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)
    assert result["status"] == "failed" and result["sources"][0]["status"] == "pending_review"
    assert client.rows == before


def test_cli_failed_apply_returns_nonzero_and_does_not_overwrite_report(migration_case, monkeypatch):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    before = copy.deepcopy(client.rows)
    report = root / "report.json"
    assert migration.main(["apply", "--snapshot", str(snapshot), "--proposal", "missing.json",
                           "--confirm-collection", "wrong", "--report", str(report)]) == 1
    saved = report.read_bytes()
    assert migration.main(["apply", "--snapshot", str(snapshot), "--proposal", "missing.json",
                           "--confirm-collection", "wrong", "--report", str(report)]) == 1
    assert report.read_bytes() == saved and client.rows == before


def test_public_update_preserves_untracked_review_with_old_known_public_audit(prepared):
    kb, client, _ = prepared
    kb._fingerprint_store._data["报告.pdf"]["public_metadata_audit"] = audit()
    for row in client.rows: row["needs_review"] = True
    public_update(kb)
    assert all(row["needs_review"] for row in client.rows)


def test_migration_cannot_touch_source_containing_another_tenant(migration_case):
    migration, kb, client, snapshot, pdf_root, root = migration_case
    import gzip, json, hashlib
    client.rows[1]["tenant_id"] = "another"
    with gzip.open(snapshot / "vectors.jsonl.gz", "wt", encoding="utf8") as f:
        for row in client.rows: f.write(json.dumps(row) + "\n")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf8"))
    manifest["artifacts"]["vectors.jsonl.gz"]["sha256"] = hashlib.sha256((snapshot / "vectors.jsonl.gz").read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="tenant"):
        migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)


def test_apply_reports_partial_success_per_source(migration_case):
    import gzip, json, hashlib
    migration, kb, client, snapshot, pdf_root, root = migration_case
    second = "第二份.pdf"
    (pdf_root / second).write_bytes((pdf_root / "报告.pdf").read_bytes())
    client.rows.append({**copy.deepcopy(client.rows[0]), "id": "second", "source": second})
    sha = hashlib.sha256((pdf_root / second).read_bytes()).hexdigest()
    kb._fingerprint_store.update(second, sha, second, 1)
    with gzip.open(snapshot / "vectors.jsonl.gz", "wt", encoding="utf8") as f:
        for row in client.rows: f.write(json.dumps(row) + "\n")
    (snapshot / "fingerprints.json").write_text(json.dumps(kb._fingerprint_store._data), encoding="utf8")
    (snapshot / "pdf_manifest.json").write_text(json.dumps({"报告.pdf": sha, second: sha}), encoding="utf8")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf8"))
    manifest["source_count"] = 2
    manifest["artifacts"]["vectors.jsonl.gz"]["count"] = 3
    for name, spec in manifest["artifacts"].items():
        spec["sha256"] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    client.rows[1]["broker"] = "并发修改"
    result = migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)
    assert result["status"] == "failed"
    assert {item["source"]: item["status"] for item in result["sources"]} == {"报告.pdf": "failed", second: "ok"}
    assert client.rows[1]["broker"] == "并发修改"
    assert client.rows[2]["date"] == "2026-09-27"


def test_fingerprint_compensation_failure_is_not_reported_restored(prepared, monkeypatch):
    kb, client, _ = prepared
    def unavailable(): raise OSError("persistent disk full")
    monkeypatch.setattr(kb._fingerprint_store, "_save", unavailable)
    with pytest.raises(RuntimeError) as error: public_update(kb)
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


@pytest.mark.parametrize("legacy_audit", [
    {"industry_l2": {"status": "unknown", "value": "银行", "reasons": ["legacy_review"],
                     "evidence": [{"page": 3, "raw": "待人工检查的旧二级行业"}]}},
    {},
    None,
])
def test_public_update_preserves_legacy_type_payload_and_audit_exactly(prepared, legacy_audit):
    kb, client, _ = prepared
    for row in client.rows:
        row.update(report_type="行业", industry_l1="食品饮料", industry_l2="银行", needs_review=True)
    if legacy_audit is not None:
        kb._fingerprint_store.update_metadata("报告.pdf", {"type_metadata_audit": legacy_audit})
    before_rows = copy.deepcopy(client.rows)
    result = public_update(kb)
    assert result["updated_count"] == 2
    for old, row in zip(before_rows, client.rows):
        assert row == {**old, "date": "2026-09-27", "broker": "东吴证券", "meta_source": "public"}
    saved = FingerprintStore(str(kb._fingerprint_store._path)).get("报告.pdf")
    assert ("type_metadata_audit" in saved) == (legacy_audit is not None)
    assert saved.get("type_metadata_audit") == legacy_audit
    assert result["metadata"]["industry_l2"] == "银行"
    assert result["metadata"]["needs_review"] is True


def test_apply_legacy_invalid_type_combination_passes_verify_without_type_audit_drift(migration_case):
    import gzip, hashlib, json
    migration, kb, client, snapshot, pdf_root, root = migration_case
    legacy_audit = {"industry_l2": {"status": "unknown", "value": "银行", "reasons": ["legacy_review"],
                                    "evidence": [{"page": 3, "raw": "保留原审核证据"}]}}
    for row in client.rows:
        row.update(report_type="行业", industry_l1="食品饮料", industry_l2="银行", needs_review=True)
    kb._fingerprint_store.update_metadata("报告.pdf", {"type_metadata_audit": legacy_audit})
    with gzip.open(snapshot / "vectors.jsonl.gz", "wt", encoding="utf8") as handle:
        for row in client.rows: handle.write(json.dumps(row) + "\n")
    (snapshot / "fingerprints.json").write_text(json.dumps(kb._fingerprint_store._data), encoding="utf8")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf8"))
    for name in ("vectors.jsonl.gz", "fingerprints.json"):
        manifest["artifacts"][name]["sha256"] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    migration.preflight(snapshot, root / "proposal", kb=kb, pdf_root=pdf_root)
    applied = migration.apply(snapshot, root / "proposal/proposal.json", "finance_kb", root / "apply.json", kb=kb, pdf_root=pdf_root)
    assert applied["status"] == "ok"
    verified = migration.verify(snapshot, root / "verify.json", kb=kb, pdf_root=pdf_root)
    assert verified["status"] == "ok", verified
    assert kb._fingerprint_store.get("报告.pdf")["type_metadata_audit"] == legacy_audit
    assert all(row["industry_l2"] == "银行" and row["needs_review"] for row in client.rows)
