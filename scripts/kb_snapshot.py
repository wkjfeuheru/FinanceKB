"""finance_kb 完整迁移快照；恢复仅针对快照明确指定的开发集合。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COLLECTION = "finance_kb"
REQUIRED_ARTIFACTS = {"vectors.jsonl.gz", "parent_chunks.jsonl.gz", "table_chunks.jsonl.gz",
                      "schema.json", "fingerprints.json", "pdf_manifest.json"}


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def snapshot_fields(schema: dict) -> list[str]:
    """BM25 输出由原文重算，Milvus 不允许读取其原始稀疏向量。"""
    generated = {name for function in schema.get("functions", [])
                 for name in function.get("output_field_names", [])}
    return [field["name"] for field in schema["fields"]
            if field["name"] not in generated and field["name"] != "sparse_vector"]


def verify_snapshot(directory: str | Path) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("collection") != COLLECTION:
        raise ValueError("snapshot collection must be finance_kb")
    for name, spec in manifest.get("artifacts", {}).items():
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError(f"unsafe artifact path: {name}")
        if _digest(path) != spec["sha256"]:
            raise ValueError(f"sha256 mismatch: {name}")
        if "count" in spec and len(_read_rows(path)) != spec["count"]:
            raise ValueError(f"count mismatch: {name}")
    if not REQUIRED_ARTIFACTS <= manifest.get('artifacts', {}).keys():
        raise ValueError('incomplete snapshot artifacts')
    rows = _read_rows(directory / 'vectors.jsonl.gz')
    parents = _read_rows(directory / 'parent_chunks.jsonl.gz')
    tables = _read_rows(directory / 'table_chunks.jsonl.gz')
    for name, items in [('vectors.jsonl.gz', rows), ('parent_chunks.jsonl.gz', parents), ('table_chunks.jsonl.gz', tables)]:
        if manifest['artifacts'][name].get('count') != len(items):
            raise ValueError(f'missing/invalid count: {name}')
        if len({row['id'] for row in items}) != len(items):
            raise ValueError(f'duplicate id: {name}')
    if not rows:
        raise ValueError('empty snapshot vectors')
    fingerprints = json.loads((directory / 'fingerprints.json').read_text(encoding='utf8'))
    pdfs = json.loads((directory / 'pdf_manifest.json').read_text(encoding='utf8'))
    sources = {row['source'] for row in rows}
    if sources != set(fingerprints) or sources != set(pdfs) or len(sources) != manifest.get('source_count'):
        raise ValueError('snapshot sources/fingerprints/PDF/source_count mismatch')
    schema = json.loads((directory / 'schema.json').read_text(encoding='utf8'))
    fields = set(snapshot_fields(schema))
    parent_ids, table_ids = {r['id'] for r in parents}, {r['id'] for r in tables}
    from finance_rag.src.rag.ingestion.metadata_extractor import build_canonical_metadata
    for row in rows:
        vector = row.get('dense_vector', [])
        if len(vector) != 1024 or any(not math.isfinite(float(x)) for x in vector):
            raise ValueError(f'invalid dense vector: {row["id"]}')
        if abs(math.sqrt(sum(float(x)**2 for x in vector)) - 1.0) > 0.001:
            raise ValueError(f'vector is not normalized: {row["id"]}')
        if not fields <= row.keys():
            raise ValueError(f'missing scalar fields: {row["id"]}')
        build_canonical_metadata(row)
        pid = row.get('parent_id')
        if pid and row.get('block_type') == 'text' and pid not in parent_ids:
            raise ValueError(f'missing parent link: {pid}')
        if pid and row.get('block_type') == 'table' and pid not in table_ids:
            raise ValueError(f'missing table link: {pid}')
    if any(row.get('collection') != COLLECTION or row['source'] not in sources for row in parents + tables):
        raise ValueError('derived collection/source mismatch')
    return manifest


def backup(directory: str | Path) -> dict:
    from sqlalchemy import create_engine, text
    from finance_rag.src.core.config import DATABASE_URL
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("backup output must be empty; existing snapshots are immutable")
    directory.mkdir(parents=True, exist_ok=True)
    kb = get_knowledge_base(COLLECTION)
    client = kb._get_client()
    schema = client.describe_collection(COLLECTION)
    fields = snapshot_fields(schema)
    iterator = client.query_iterator(collection_name=COLLECTION, filter="", output_fields=fields, batch_size=256)
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    count = int(client.get_collection_stats(COLLECTION)["row_count"])
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError(f"Milvus snapshot count mismatch: {len(rows)} != {count}")
    if not all(len(row["dense_vector"]) == 1024 for row in rows):
        raise ValueError("snapshot must contain all 1024-dimensional dense vectors")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    sql_rows = {}
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                for table in ("parent_chunks", "table_chunks"):
                    sql_rows[table] = [dict(row) for row in connection.execute(
                        text(f"SELECT * FROM {table} WHERE collection=:collection"), {"collection": COLLECTION}
                    ).mappings()]
    finally:
        engine.dispose()
    fingerprints = dict(kb.get_fingerprint_store()._data)
    sources = {row["source"] for row in rows}
    expected = {path.name for path in (ROOT / "assets/reports").glob("*.pdf")}
    if sources != expected or set(fingerprints) != expected:
        raise ValueError("snapshot must contain exactly the 18 report sources and fingerprints")
    parents = {row["id"] for row in sql_rows["parent_chunks"]}
    tables = {row["id"] for row in sql_rows["table_chunks"]}
    for row in rows:
        pid = row.get("parent_id")
        if pid and row.get("block_type") == "text" and pid not in parents:
            raise ValueError(f"missing text parent: {pid}")
        if pid and row.get("block_type") == "table" and pid not in tables:
            raise ValueError(f"missing table: {pid}")
    artifacts = {}
    for name, items in [("vectors.jsonl.gz", rows), ("parent_chunks.jsonl.gz", sql_rows["parent_chunks"]),
                        ("table_chunks.jsonl.gz", sql_rows["table_chunks"])]:
        path = directory / name
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        artifacts[name] = {"sha256": _digest(path), "count": len(items)}
    for name, payload in [("schema.json", schema), ("fingerprints.json", fingerprints),
                          ("pdf_manifest.json", {path.name: _digest(path) for path in sorted((ROOT / "assets/reports").glob("*.pdf"))})]:
        path = directory / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        artifacts[name] = {"sha256": _digest(path)}
    manifest = {"collection": COLLECTION, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_count": len(sources), "artifacts": artifacts,
                "generated_fields": "BM25 sparse_vector is regenerated from preserved content on restore"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_snapshot(directory)
    return manifest


def restore(directory: str | Path, confirmation: str) -> dict:
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    if confirmation != COLLECTION:
        raise ValueError("restore requires --confirm-collection finance_kb")
    directory = Path(directory).resolve()
    manifest = verify_snapshot(directory)
    required = {"vectors.jsonl.gz", "parent_chunks.jsonl.gz", "table_chunks.jsonl.gz", "schema.json", "fingerprints.json", "pdf_manifest.json"}
    if not required <= manifest["artifacts"].keys():
        raise ValueError("incomplete restore snapshot")
    rows = _read_rows(directory / "vectors.jsonl.gz")
    if not rows or not all(len(row.get("dense_vector", [])) == 1024 for row in rows):
        raise ValueError("invalid restore vectors")
    kb = get_knowledge_base(COLLECTION)
    schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    current = kb._get_client().describe_collection(COLLECTION)
    if schema["fields"] != current["fields"]:
        raise ValueError("restore schema differs from current schema")
    kb.rebuild_collection()
    for start in range(0, len(rows), 128):
        kb._get_client().upsert(collection_name=COLLECTION, data=rows[start:start + 128])
    kb._get_parent_store().store_batch(_read_rows(directory / "parent_chunks.jsonl.gz"))
    from finance_rag.src.rag.retrieval.table_store import TableStore
    TableStore(COLLECTION).store_batch(_read_rows(directory / "table_chunks.jsonl.gz"))
    store = kb.get_fingerprint_store()
    with store._lock:
        store._data = json.loads((directory / "fingerprints.json").read_text(encoding="utf-8"))
        store._save()
    kb._get_client().flush(COLLECTION)
    if int(kb._get_client().get_collection_stats(COLLECTION)["row_count"]) != len(rows):
        raise ValueError("restore count mismatch")
    if store._data != json.loads((directory / 'fingerprints.json').read_text(encoding='utf8')):
        raise ValueError('restore fingerprint mismatch')
    from sqlalchemy import create_engine, text
    from finance_rag.src.core.config import DATABASE_URL
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            for table, name in [('parent_chunks', 'parent_chunks.jsonl.gz'), ('table_chunks', 'table_chunks.jsonl.gz')]:
                actual = [dict(row) for row in connection.execute(
                    text(f'SELECT * FROM {table} WHERE collection=:collection'), {'collection': COLLECTION}).mappings()]
                expected = _read_rows(directory / name)
                fields = ('id', 'source', 'content', 'heading_path') if table == 'parent_chunks' else ('id', 'source', 'markdown', 'payload', 'row_count', 'heading_path')
                def canonical(items):
                    return sorted(json.dumps({key: row.get(key) for key in fields}, ensure_ascii=False, sort_keys=True, default=str) for row in items)
                if canonical(actual) != canonical(expected):
                    raise ValueError(f'restore derived contents mismatch: {table}')
    finally:
        engine.dispose()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "verify", "restore"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--confirm-collection", default="")
    args = parser.parse_args()
    if args.action == "backup":
        result = backup(args.snapshot)
    elif args.action == "restore":
        result = restore(args.snapshot, args.confirm_collection)
    else:
        result = verify_snapshot(args.snapshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
