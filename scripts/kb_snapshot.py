"""Milvus 集合完整快照 + PG 关系数据导出；恢复以快照声明的集合为准。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTION = "finance_kb"
PG_TABLES = ("documents", "document_versions", "document_chunks", "knowledge_bases")
PG_ARTIFACTS = tuple(f"{name}.jsonl" for name in PG_TABLES)
KNOWN_CATEGORIES = ("investment_research", "compliance_risk", "operations", "management")
REQUIRED_ARTIFACTS = {"vectors.jsonl.gz", "parent_chunks.jsonl.gz", "table_chunks.jsonl.gz",
                      "schema.json", "fingerprints.json", "pdf_manifest.json"}


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: list[dict], *, compressed: bool = False) -> None:
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return
    with path.open("wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def fingerprint_path_for_snapshot(collection: str) -> Path:
    """快照读写的指纹文件。

    历史集合 ``finance_kb`` 曾是默认集合，指纹位于基础文件；默认集合改为
    ``finance_report`` 后 ``fingerprint_path_for`` 会把它当成非默认集合，
    直接按集合名解析会指向一个不存在的新文件。
    """
    from src.core.config import MILVUS_FINGERPRINT_PATH
    from src.index.fingerprint_store import fingerprint_path_for

    if collection == COLLECTION:
        return Path(MILVUS_FINGERPRINT_PATH)
    return Path(fingerprint_path_for(collection))


def snapshot_fields(schema: dict) -> list[str]:
    """BM25 输出由原文重算，Milvus 不允许读取其原始稀疏向量。"""
    generated = {name for function in schema.get("functions", [])
                 for name in function.get("output_field_names", [])}
    return [field["name"] for field in schema["fields"]
            if field["name"] not in generated and field["name"] != "sparse_vector"]


def _verify_postgres(directory: Path, artifacts: dict) -> None:
    present = [name for name in PG_ARTIFACTS if name in artifacts]
    if not present:
        return
    rows = {name: _read_rows(directory / name) for name in present}
    documents = rows.get("documents.jsonl", [])
    versions = rows.get("document_versions.jsonl", [])
    chunks = rows.get("document_chunks.jsonl", [])
    for name, key in [("documents.jsonl", "document_id"),
                      ("document_versions.jsonl", "version_id"),
                      ("document_chunks.jsonl", "chunk_id")]:
        if name in rows and len({row[key] for row in rows[name]}) != len(rows[name]):
            raise ValueError(f"duplicate primary key: {name}")
    document_ids = {row["document_id"] for row in documents}
    version_ids = {row["version_id"] for row in versions}
    if any(row.get("document_id") not in document_ids for row in versions):
        raise ValueError("document_versions.jsonl has orphan document_id")
    if any(row.get("document_id") not in document_ids for row in chunks):
        raise ValueError("document_chunks.jsonl has orphan document_id")
    if any(row.get("version_id") not in version_ids for row in chunks):
        raise ValueError("document_chunks.jsonl has orphan version_id")


def verify_snapshot(directory: str | Path, *, collection: str = COLLECTION) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("collection") != collection:
        raise ValueError(f"snapshot collection must be {collection}")
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
    if len(sources) != manifest.get('source_count'):
        raise ValueError('snapshot sources/source_count mismatch')
    if manifest.get('allow_extra_sources'):
        if not sources <= set(fingerprints):
            raise ValueError('snapshot sources missing fingerprints')
    elif sources != set(fingerprints) or sources != set(pdfs):
        raise ValueError('snapshot sources/fingerprints/PDF/source_count mismatch')
    schema = json.loads((directory / 'schema.json').read_text(encoding='utf8'))
    fields = set(snapshot_fields(schema))
    parent_ids, table_ids = {r['id'] for r in parents}, {r['id'] for r in tables}
    from src.chunking.metadata_extractor import build_canonical_metadata
    for row in rows:
        vector = row.get('dense_vector', [])
        if len(vector) != 1024 or any(not math.isfinite(float(x)) for x in vector):
            raise ValueError(f'invalid dense vector: {row["id"]}')
        if abs(math.sqrt(sum(float(x)**2 for x in vector)) - 1.0) > 0.001:
            raise ValueError(f'vector is not normalized: {row["id"]}')
        if not fields <= row.keys():
            raise ValueError(f'missing scalar fields: {row["id"]}')
        # 非研报行没有 report_type；canonical 校验只适用于研报元数据
        if row.get('report_type') and row.get('category') in ("", "investment_research"):
            build_canonical_metadata(row)
        pid = row.get('parent_id')
        if pid and row.get('block_type') == 'text' and pid not in parent_ids:
            raise ValueError(f'missing parent link: {pid}')
        if pid and row.get('block_type') == 'table' and pid not in table_ids:
            raise ValueError(f'missing table link: {pid}')
    if any(row.get('collection') != manifest['collection'] or row['source'] not in sources for row in parents + tables):
        raise ValueError('derived collection/source mismatch')
    _verify_postgres(directory, manifest.get('artifacts', {}))
    if 'sources_by_category.json' in manifest.get('artifacts', {}):
        categories = json.loads((directory / 'sources_by_category.json').read_text(encoding='utf8'))
        if set(categories.get('milvus', {})) != sources:
            raise ValueError('sources_by_category.json does not cover snapshot sources')
    return manifest


def _export_postgres(connection, directory: Path) -> tuple[dict, dict]:
    from sqlalchemy import inspect, text

    artifacts: dict = {}
    statuses: dict[str, str] = {}
    inspector = inspect(connection)
    for table in PG_TABLES:
        if not inspector.has_table(table):
            statuses[table] = "missing"
            continue
        rows = [dict(row) for row in connection.execute(text(f'SELECT * FROM "{table}"')).mappings()]
        path = directory / f"{table}.jsonl"
        _write_rows(path, rows)
        artifacts[f"{table}.jsonl"] = {"sha256": _digest(path), "count": len(rows)}
        statuses[table] = f"exported:{len(rows)}"
    return artifacts, statuses


def _source_categories(rows: list[dict], key: str) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for row in rows:
        mapping.setdefault(str(row.get("source", "")), set()).add(str(row.get(key) or ""))
    return {source: sorted(values) for source, values in mapping.items()}


def _write_source_categories(directory: Path, rows: list[dict], pg_documents: list[dict]) -> None:
    milvus = _source_categories(rows, "category")
    postgres = _source_categories(pg_documents, "category")
    all_categories = {value for values in (*milvus.values(), *postgres.values()) for value in values}
    unknown = sorted(value for value in all_categories if value and value not in KNOWN_CATEGORIES)
    overlap = sorted(
        source for source in set(milvus) | set(postgres)
        if len(set(milvus.get(source, [])) | set(postgres.get(source, []))) > 1
    )
    payload = {
        "known_categories": list(KNOWN_CATEGORIES),
        "milvus": milvus,
        "postgres": postgres,
        "unknown": unknown,
        "uncategorized": sorted(source for source, values in milvus.items() if values == [""]),
        "overlap": overlap,
        "milvus_only": sorted(set(milvus) - set(postgres)),
        "postgres_only": sorted(set(postgres) - set(milvus)),
    }
    (directory / "sources_by_category.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backup(directory: str | Path, *, collection: str = COLLECTION,
           allow_extra_sources: bool = False) -> dict:
    from sqlalchemy import create_engine

    from src.core.config import DATABASE_URL
    from src.index.milvus_kb import get_knowledge_base

    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("backup output must be empty; existing snapshots are immutable")
    directory.mkdir(parents=True, exist_ok=True)
    kb = get_knowledge_base(collection)
    client = kb._get_client()
    schema = client.describe_collection(collection)
    fields = snapshot_fields(schema)
    iterator = client.query_iterator(collection_name=collection, filter="", output_fields=fields, batch_size=256)
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    count = int(client.get_collection_stats(collection)["row_count"])
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError(f"Milvus snapshot count mismatch: {len(rows)} != {count}")
    if not all(len(row["dense_vector"]) == 1024 for row in rows):
        raise ValueError("snapshot must contain all 1024-dimensional dense vectors")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    sql_rows = {}
    pg_statuses = {}
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                for table in ("parent_chunks", "table_chunks"):
                    sql_rows[table] = [dict(row) for row in connection.execute(
                        text_select(table, collection), {"collection": collection}
                    ).mappings()]
                pg_artifacts, pg_statuses = _export_postgres(connection, directory)
    finally:
        engine.dispose()
    fingerprints = json.loads(fingerprint_path_for_snapshot(collection).read_text(encoding="utf-8"))
    sources = {row["source"] for row in rows}
    expected = {path.name for path in (ROOT / "assets/reports").glob("*.pdf")}
    if not allow_extra_sources and (sources != expected or set(fingerprints) != expected):
        raise ValueError("snapshot must contain exactly the 18 report sources and fingerprints")
    if not sources <= set(fingerprints):
        raise ValueError("snapshot sources missing fingerprints")
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
        _write_rows(path, items, compressed=True)
        artifacts[name] = {"sha256": _digest(path), "count": len(items)}
    artifacts.update(pg_artifacts)
    pg_documents = _read_rows(directory / "documents.jsonl") if "documents.jsonl" in artifacts else []
    _write_source_categories(directory, rows, pg_documents)
    artifacts["sources_by_category.json"] = {"sha256": _digest(directory / "sources_by_category.json")}
    for name, payload in [("schema.json", schema), ("fingerprints.json", fingerprints),
                          ("pdf_manifest.json", {path.name: _digest(path) for path in sorted((ROOT / "assets/reports").glob("*.pdf"))})]:
        path = directory / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        artifacts[name] = {"sha256": _digest(path)}
    manifest = {"collection": collection, "created_at": datetime.now(UTC).isoformat(),
                "source_count": len(sources), "artifacts": artifacts,
                "allow_extra_sources": allow_extra_sources,
                "postgres_tables": pg_statuses,
                "generated_fields": "BM25 sparse_vector is regenerated from preserved content on restore"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_snapshot(directory, collection=collection)
    return manifest


def text_select(table: str, collection: str):
    from sqlalchemy import text
    return text(f"SELECT * FROM {table} WHERE collection=:collection")


def _coerce_rows(table, rows: list[dict]) -> list[dict]:
    """JSON 里的时间戳是字符串；按反射列类型还原为 datetime 再插入。"""
    from sqlalchemy import DateTime
    datetime_columns = {column.name for column in table.columns if isinstance(column.type, DateTime)}
    coerced = []
    for row in rows:
        item = dict(row)
        for name in datetime_columns & item.keys():
            value = item[name]
            if isinstance(value, str) and value:
                item[name] = datetime.fromisoformat(value)
        coerced.append(item)
    return coerced


def _restore_postgres(directory: Path, artifacts: dict) -> None:
    from sqlalchemy import MetaData, Table, create_engine, inspect, tuple_

    from src.core.config import DATABASE_URL

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        inspector = inspect(engine)
        for name in PG_TABLES:
            artifact = f"{name}.jsonl"
            path = directory / artifact
            if artifact not in artifacts or not path.exists():
                continue
            if not inspector.has_table(name):
                # knowledge_bases 已随知识库 CRUD 下线；新环境没有该表时跳过，
                # 其余表缺失说明目标库未初始化，必须显式失败。
                if name == "knowledge_bases":
                    continue
                raise ValueError(f"cannot restore {name}: table does not exist")
            rows = _read_rows(path)
            if not rows:
                continue
            table = Table(name, MetaData(), autoload_with=engine)
            keys = [column.name for column in table.primary_key.columns]
            if not keys:
                raise ValueError(f"cannot restore {name}: table has no primary key")
            prepared = _coerce_rows(table, rows)
            with engine.begin() as connection:
                for start in range(0, len(prepared), 500):
                    batch = prepared[start:start + 500]
                    connection.execute(table.delete().where(
                        tuple_(*(table.c[key] for key in keys)).in_(
                            [tuple(row[key] for key in keys) for row in batch])))
                    connection.execute(table.insert(), batch)
    finally:
        engine.dispose()


def restore(directory: str | Path, confirmation: str, *, collection: str | None = None) -> dict:
    from src.index.fingerprint_store import FingerprintStore
    from src.index.milvus_kb import get_knowledge_base

    directory = Path(directory).resolve()
    manifest = verify_snapshot(directory, collection=collection or COLLECTION)
    target = manifest["collection"]
    if confirmation != target:
        raise ValueError(f"restore requires --confirm-collection {target}")
    required = {"vectors.jsonl.gz", "parent_chunks.jsonl.gz", "table_chunks.jsonl.gz", "schema.json", "fingerprints.json", "pdf_manifest.json"}
    if not required <= manifest["artifacts"].keys():
        raise ValueError("incomplete restore snapshot")
    rows = _read_rows(directory / "vectors.jsonl.gz")
    if not rows or not all(len(row.get("dense_vector", [])) == 1024 for row in rows):
        raise ValueError("invalid restore vectors")
    kb = get_knowledge_base(target)
    schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    current = kb._get_client().describe_collection(target)
    if schema["fields"] != current["fields"]:
        raise ValueError("restore schema differs from current schema")
    kb.rebuild_collection()
    for start in range(0, len(rows), 128):
        kb._get_client().upsert(collection_name=target, data=rows[start:start + 128])
    kb._get_parent_store().store_batch(_read_rows(directory / "parent_chunks.jsonl.gz"))
    from src.index.table_store import TableStore
    TableStore(target).store_batch(_read_rows(directory / "table_chunks.jsonl.gz"))
    store = FingerprintStore(str(fingerprint_path_for_snapshot(target)))
    expected_fingerprints = json.loads((directory / "fingerprints.json").read_text(encoding="utf-8"))
    with store._lock:
        store._data = expected_fingerprints
        store._save()
    kb._get_client().flush(target)
    if int(kb._get_client().get_collection_stats(target)["row_count"]) != len(rows):
        raise ValueError("restore count mismatch")
    if store._data != expected_fingerprints:
        raise ValueError('restore fingerprint mismatch')
    from sqlalchemy import create_engine, text

    from src.core.config import DATABASE_URL
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            for table, name in [('parent_chunks', 'parent_chunks.jsonl.gz'), ('table_chunks', 'table_chunks.jsonl.gz')]:
                actual = [dict(row) for row in connection.execute(
                    text(f'SELECT * FROM {table} WHERE collection=:collection'), {'collection': target}).mappings()]
                expected = _read_rows(directory / name)
                fields = ('id', 'source', 'content', 'heading_path') if table == 'parent_chunks' else ('id', 'source', 'markdown', 'payload', 'row_count', 'heading_path')
                def canonical(items, fields=fields):
                    return sorted(json.dumps({key: row.get(key) for key in fields}, ensure_ascii=False, sort_keys=True, default=str) for row in items)
                if canonical(actual) != canonical(expected):
                    raise ValueError(f'restore derived contents mismatch: {table}')
    finally:
        engine.dispose()
    _restore_postgres(directory, manifest.get("artifacts", {}))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "verify", "restore"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--allow-extra-sources", action="store_true")
    parser.add_argument("--confirm-collection", default="")
    args = parser.parse_args()
    if args.action == "backup":
        result = backup(args.snapshot, collection=args.collection,
                        allow_extra_sources=args.allow_extra_sources)
    elif args.action == "restore":
        result = restore(args.snapshot, args.confirm_collection, collection=args.collection)
    else:
        result = verify_snapshot(args.snapshot, collection=args.collection)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
