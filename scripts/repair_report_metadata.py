"""从原 PDF 提议公共标量修正；默认只读，apply 必须显式确认。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kb_snapshot import COLLECTION, _digest, _read_rows, snapshot_fields, verify_snapshot
from src.core.config import TENANT_ID
from src.chunking.public_metadata import RULES_VERSION, resolve_public_metadata

MUTABLE = {"date", "broker", "meta_source", "needs_review"}
AUDITS = {"public_metadata_audit", "type_metadata_audit"}
CODE_FILES = ["scripts/repair_report_metadata.py", "scripts/kb_snapshot.py",
              "src/chunking/public_metadata.py",
              "src/chunking/metadata_extractor.py",
              "src/index/milvus_kb.py",
              "src/index/fingerprint_store.py",
              "src/storage/relational_db/document_manifest.py",
              "src/ingest/documents.py"]


def digest_value(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str, separators=(",", ":")).encode()).hexdigest()


def code_version():
    return {name: _digest(ROOT / name) for name in CODE_FILES}


def grouped(rows):
    result = {}
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("duplicate row ID")
        ids.add(row["id"])
        result.setdefault(row["source"], []).append(row)
    return {source: sorted(items, key=lambda r: r["id"]) for source, items in sorted(result.items())}


def binding(rows):
    """保留每行完整 before scalars，正文与向量以摘要绑定。"""
    return [{"scalars": {k: v for k, v in row.items() if k not in {"content", "dense_vector", "sparse_vector"}},
             "content_sha256": digest_value(row.get("content")),
             "dense_sha256": digest_value(row.get("dense_vector"))} for row in sorted(rows, key=lambda r: r["id"])]


def identity(rows):
    return digest_value([{k: v for k, v in row.items() if k not in MUTABLE | {"sparse_vector"}}
                         for row in sorted(rows, key=lambda r: r["id"])])


def read_live_rows(kb):
    client = kb._get_client()
    fields = snapshot_fields(client.describe_collection(COLLECTION))
    iterator = client.query_iterator(collection_name=COLLECTION, filter="", output_fields=fields, batch_size=256)
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    if len(rows) != int(client.get_collection_stats(COLLECTION)["row_count"]):
        raise ValueError("live collection row count mismatch")
    return rows


def read_derived():
    from sqlalchemy import create_engine, text
    from src.core.config import DATABASE_URL
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                return {name: [dict(row) for row in connection.execute(
                    text(f"SELECT * FROM {name} WHERE collection=:collection"),
                    {"collection": COLLECTION}).mappings()] for name in ("parent_chunks", "table_chunks")}
    finally:
        engine.dispose()


def load_before(snapshot, pdf_root):
    snapshot, pdf_root = Path(snapshot), Path(pdf_root)
    manifest = verify_snapshot(snapshot, collection=COLLECTION)
    rows = grouped(_read_rows(snapshot / "vectors.jsonl.gz"))
    if any(row.get("tenant_id") != TENANT_ID for items in rows.values() for row in items):
        raise ValueError("snapshot contains rows outside configured tenant")
    pdfs = json.loads((snapshot / "pdf_manifest.json").read_text(encoding="utf8"))
    fingerprints = json.loads((snapshot / "fingerprints.json").read_text(encoding="utf8"))
    if set(pdfs) != {path.name for path in pdf_root.glob("*.pdf")}:
        raise ValueError("PDF source set drift")
    for source, sha in pdfs.items():
        if Path(source).name != source or _digest(pdf_root / source) != sha:
            raise ValueError(f"PDF SHA drift: {source}")
        if fingerprints[source].get("sha256") != sha:
            raise ValueError(f"snapshot fingerprint/PDF SHA mismatch: {source}")
    return manifest, rows, pdfs, fingerprints


def check_derived(snapshot):
    live = read_derived()
    for name, rows in live.items():
        expected = _read_rows(Path(snapshot) / (name + ".jsonl.gz"))
        if digest_value(sorted(rows, key=lambda r: r["id"])) != digest_value(sorted(expected, key=lambda r: r["id"])):
            raise ValueError(f"derived rows drift: {name}")
    return {name: len(rows) for name, rows in live.items()}


def get_kb(kb):
    if kb is None:
        from src.index.milvus_kb import get_knowledge_base
        kb = get_knowledge_base(COLLECTION)
    if kb.collection_name != COLLECTION:
        raise ValueError(f"collection must be {COLLECTION}")
    return kb


def source_proposal(source, rows, fingerprint, pdf_root, as_of):
    resolved = resolve_public_metadata(source, pdf_path=Path(pdf_root) / source, as_of=as_of)
    audit = resolved.audit
    statuses = {field: item["status"] for field, item in audit["fields"].items()}
    old_audit = fingerprint.get("public_metadata_audit", {})
    protected = any(row.get("meta_source") == "manual" for row in rows) or any(
        item.get("status") == "manual" for item in old_audit.get("fields", {}).values())
    contradictions = [field for field in ("date", "broker") if statuses[field] == "known"
                      and any(row.get(field, "") != audit["fields"][field]["value"] for row in rows)]
    state = "manual_review" if protected else "conflict_review" if "conflict" in statuses.values() else "ready"
    return {"source": source, "status": state, "field_status": statuses,
            "manual_contradictions": contradictions if protected else [],
            "patch": {"date": resolved.date, "broker": resolved.broker} if not protected or contradictions else None,
            "audit": audit, "before": binding(rows), "identity_sha256": identity(rows),
            "before_fingerprint": fingerprint}


def preflight(snapshot, output, *, kb=None, pdf_root=None):
    kb, pdf_root = get_kb(kb), Path(pdf_root or ROOT / "assets/reports")
    output = Path(output)
    if output.exists():
        raise FileExistsError("proposal output already exists")
    manifest, before, pdfs, fingerprints = load_before(snapshot, pdf_root)
    with kb._write_lock:
        live = grouped(read_live_rows(kb))
        if live != before:
            raise ValueError("before rows/scalars/identity drift")
        derived = check_derived(snapshot)
        proposals = []
        for source, rows in before.items():
            if kb.get_fingerprint_store().get(source) != fingerprints[source]:
                raise ValueError(f"fingerprint drift: {source}")
            item = source_proposal(source, rows, fingerprints[source], pdf_root, date.today())
            item["manifest_before"] = kb._get_manifest_repository().read_metadata_dates(TENANT_ID, source)
            proposals.append(item)
    result = {"format_version": 1, "collection": COLLECTION, "tenant_id": TENANT_ID,
              "snapshot_sha256": digest_value(manifest), "pdf_sha256": pdfs, "code_version": code_version(),
              "rules_version": RULES_VERSION, "as_of": date.today().isoformat(), "source_count": len(before),
              "created_at": datetime.now(timezone.utc).isoformat(), "derived": derived,
              "summary": {field: dict(Counter(item["field_status"][field] for item in proposals))
                          for field in ("date", "broker")}, "sources": proposals}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "proposal.json").open("x", encoding="utf8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    return result


def save_report(handle, result):
    handle.seek(0)
    json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


def apply(snapshot, proposal, confirmation, report, *, kb=None, pdf_root=None):
    """逐源补偿而非跨库原子事务；同进程 PATCH 与迁移共享集合锁。"""
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "sources": []}
    with report.open("x", encoding="utf8") as handle:
        save_report(handle, result)
        try:
            if confirmation != COLLECTION:
                raise ValueError(f"explicit --confirm-collection {COLLECTION} required")
            kb, pdf_root = get_kb(kb), Path(pdf_root or ROOT / "assets/reports")
            manifest, before, pdfs, fingerprints = load_before(snapshot, pdf_root)
            proposed = json.loads(Path(proposal).read_text(encoding="utf8"))
            if (proposed.get("snapshot_sha256") != digest_value(manifest) or proposed.get("pdf_sha256") != pdfs
                    or proposed.get("collection") != COLLECTION or proposed.get("tenant_id") != TENANT_ID
                    or proposed.get("rules_version") != RULES_VERSION or proposed.get("code_version") != code_version()):
                raise ValueError("proposal snapshot/PDF/code/rules binding mismatch")
            items = proposed["sources"]
            if len(items) != len(before) or {item["source"] for item in items} != set(before):
                raise ValueError("proposal source set mismatch")
            check_derived(snapshot)
            for item in items:
                source = item["source"]
                entry = None
                try:
                    with kb._write_lock:
                        live = grouped(read_live_rows(kb))
                        if set(live) != set(before):
                            raise ValueError("live source set drift")
                        if binding(live[source]) != binding(before[source]) or item["before"] != binding(before[source]):
                            raise ValueError("before scalars/identity drift")
                        if identity(live[source]) != item["identity_sha256"]:
                            raise ValueError("identity drift")
                        if _digest(pdf_root / source) != pdfs[source]:
                            raise ValueError("PDF SHA drift")
                        if kb.get_fingerprint_store().get(source) != fingerprints[source]:
                            raise ValueError("fingerprint/audit drift")
                        expected = source_proposal(source, before[source], fingerprints[source], pdf_root,
                                                   date.fromisoformat(proposed["as_of"]))
                        if any(item[key] != expected[key] for key in expected):
                            raise ValueError("proposal evidence/patch mismatch")
                        if item["status"] != "ready":
                            result["sources"].append({"source": source, "status": "pending_review",
                                                       "reason": item["status"]})
                            save_report(handle, result)
                            continue
                        audit = {**item["audit"], "migration_manifest_before": item["manifest_before"]}
                        # 在写前持久化恢复证据，进程中断留下 running 源供人工恢复。
                        entry = {"source": source, "status": "running", "before_rows": live[source],
                                 "before_fingerprint": fingerprints[source], "manifest_before": item["manifest_before"]}
                        result["sources"].append(entry)
                        save_report(handle, result)
                        updated = kb.update_document_metadata(source, item["patch"], update_source="public_metadata", metadata_audit=audit)
                        entry.update(status="ok", updated_count=updated["updated_count"], manifest=updated["manifest"])
                except Exception as exc:
                    failure = getattr(exc, "report", {"source": source, "status": "failed", "error": str(exc)})
                    if entry is None:
                        result["sources"].append(failure)
                    else:
                        entry.update(failure)
                save_report(handle, result)
            result["status"] = "ok" if all(item["status"] == "ok" for item in result["sources"]) else "failed"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        save_report(handle, result)
    return result


def verify(snapshot, report, *, kb=None, pdf_root=None):
    result = {"status": "running", "sources": []}
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as handle:
        try:
            kb, pdf_root = get_kb(kb), Path(pdf_root or ROOT / "assets/reports")
            _, before, pdfs, fingerprints = load_before(snapshot, pdf_root)
            with kb._write_lock:
                live = grouped(read_live_rows(kb))
                if set(live) != set(before):
                    raise ValueError("live source set drift")
                result["derived"] = check_derived(snapshot)
                for source, rows in live.items():
                    if identity(rows) != identity(before[source]):
                        raise ValueError(f"non-public fields/content/vector drift: {source}")
                    saved = kb.get_fingerprint_store().get(source) or {}
                    if {k: v for k, v in saved.items() if k not in AUDITS} != {k: v for k, v in fingerprints[source].items() if k not in AUDITS}:
                        raise ValueError(f"content fingerprint drift: {source}")
                    if saved.get("type_metadata_audit") != fingerprints[source].get("type_metadata_audit"):
                        raise ValueError(f"type audit drift: {source}")
                    audit = saved.get("public_metadata_audit")
                    changed = binding(rows) != binding(before[source])
                    if changed or any(row.get("meta_source") == "public" for row in rows):
                        if not audit:
                            raise ValueError(f"missing persisted public audit: {source}")
                        from src.chunking.public_metadata import validate_public_metadata_audit, public_audit_needs_review
                        audit = validate_public_metadata_audit(audit)
                        if audit["source"] != source or audit["pdf_sha256"] != pdfs[source]:
                            raise ValueError(f"audit source/PDF SHA mismatch: {source}")
                        resolved = resolve_public_metadata(source, pdf_path=pdf_root / source)
                        if any(audit[key] != resolved.audit[key] for key in resolved.audit):
                            raise ValueError(f"audit differs from current PDF evidence: {source}")
                        for row in rows:
                            if any(row[field] != audit["fields"][field]["value"] for field in ("date", "broker")):
                                raise ValueError(f"audit/scalar mismatch: {source}")
                            if row["meta_source"] != "public" or ((public_audit_needs_review(audit) or saved.get("type_metadata_audit")) and not row["needs_review"]):
                                raise ValueError(f"public provenance/review mismatch: {source}")
                    pg = kb._get_manifest_repository().read_metadata_dates(TENANT_ID, source)
                    if pg["rows"] and any(doc["date"] != rows[0]["date"] for doc in pg["rows"]):
                        raise ValueError(f"PG display date mismatch: {source}")
                    result["sources"].append({"source": source, "status": "ok", "rows": len(rows),
                                               "state": "public_updated" if changed else "unchanged_before", "manifest": pg["status"]})
            result["status"] = "ok"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        save_report(handle, result)
    return result


def main(argv=None):
    global COLLECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=COLLECTION, help="目标集合名（默认 finance_kb）")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "apply", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--snapshot", required=True)
        command.add_argument("--output" if name == "preflight" else "--report", required=True)
        if name == "apply":
            command.add_argument("--proposal", required=True)
            command.add_argument("--confirm-collection", required=True)
    args = parser.parse_args(argv)
    COLLECTION = args.collection
    try:
        if args.command == "preflight":
            result = preflight(args.snapshot, args.output)
            print(json.dumps({"source_count": result["source_count"], "summary": result["summary"]}, ensure_ascii=False))
            return 0
        if args.command == "apply":
            result = apply(args.snapshot, args.proposal, args.confirm_collection, args.report)
        else:
            result = verify(args.snapshot, args.report)
        print(json.dumps({key: value for key, value in result.items() if key != "sources"}, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
