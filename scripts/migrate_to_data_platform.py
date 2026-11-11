"""把单集合 ``finance_kb`` 迁移为 ``finance_report`` + ``finance_policy``。

分阶段执行，默认 dry-run；只有显式 ``--apply`` 才写入。每个阶段可重复执行：
复制按主键幂等，删除与 retire/drop 需要显式确认参数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_COLLECTION = "finance_kb"
RETIRED_PREFIX = "finance_kb_retired_"
REPORT_CATEGORY = "investment_research"
DELETE_CATEGORIES = {"compliance_risk", "operations", "management"}
PHASES = ("preflight", "snapshot", "build", "copy", "fingerprints", "verify",
          "delete", "retire", "drop")

PARENT_COLUMNS = ("collection", "id", "content", "heading", "heading_path", "source")
TABLE_COLUMNS = ("collection", "id", "parent_id", "payload", "markdown", "row_count",
                 "source", "heading_path", "start_page")


@dataclass(frozen=True)
class MigrationPlan:
    report_sources: list[str]
    delete_sources: list[str]
    uncategorized: list[str]
    categories: dict[str, list[str]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _quoted_sources(sources: Iterable[str]) -> str:
    return ", ".join(f'"{_escape(source)}"' for source in sources)


def merge_categories(categories: dict[str, Any]) -> dict[str, list[str]]:
    """合并 Milvus 与 PG 两类来源；同一 source 出现多个类别即视为重叠。"""
    merged: dict[str, set[str]] = {}
    for key in ("milvus", "postgres"):
        for source, values in (categories.get(key) or {}).items():
            merged.setdefault(str(source), set()).update(str(value) for value in values)
    return {source: sorted(values) for source, values in merged.items()}


def partition_sources(categories: dict[str, Any], *, include_uncategorized: bool = False) -> MigrationPlan:
    unknown = sorted(categories.get("unknown") or [])
    if unknown:
        raise ValueError(f"存在未知类别：{unknown}")
    merged = merge_categories(categories)
    overlap = sorted(source for source, values in merged.items() if len(values) > 1)
    if overlap:
        raise ValueError(f"同一来源跨多个类别，拒绝迁移：{overlap}")
    report, delete, uncategorized = [], [], []
    for source, values in sorted(merged.items()):
        current = set(values)
        if current == {REPORT_CATEGORY}:
            report.append(source)
        elif current and current <= DELETE_CATEGORIES:
            delete.append(source)
        elif current == {""}:
            uncategorized.append(source)
        else:
            raise ValueError(f"来源 {source} 的类别无法归类：{sorted(current)}")
    if uncategorized and not include_uncategorized:
        raise ValueError(f"存在未分类来源：{uncategorized}；确认按删除处理请加 --include-uncategorized")
    delete.extend(uncategorized)
    return MigrationPlan(report, delete, uncategorized, merged)


def load_plan(snapshot: Path, *, include_uncategorized: bool = False) -> MigrationPlan:
    path = snapshot / "sources_by_category.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少快照来源清单：{path}（先执行 --phase snapshot）")
    return partition_sources(_read_json(path), include_uncategorized=include_uncategorized)


def snapshot_source_counts(snapshot: Path) -> dict[str, int]:
    from scripts.kb_snapshot import _read_rows

    counts: dict[str, int] = {}
    for row in _read_rows(snapshot / "vectors.jsonl.gz"):
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return counts


def _save_state(snapshot: Path, phase: str, payload: dict[str, Any]) -> None:
    path = snapshot / "state.json"
    state = _read_json(path) if path.exists() else {"phases": {}}
    state.setdefault("phases", {})[phase] = {
        "at": datetime.now(UTC).isoformat(),
        **payload,
    }
    _write_json_atomic(path, state)


def _client():
    from src.index.milvus_kb import get_knowledge_base

    return get_knowledge_base(SOURCE_COLLECTION)._get_client()


def _row_count(client, collection: str) -> int:
    return int(client.get_collection_stats(collection).get("row_count", 0))


def _iter_rows(client, collection: str, expression: str, fields: list[str],
               batch_size: int = 200):
    iterator = client.query_iterator(collection_name=collection, filter=expression,
                                     output_fields=fields, batch_size=batch_size)
    try:
        while batch := iterator.next():
            yield from batch
    finally:
        iterator.close()


def _schema_signature(schema: dict) -> dict[str, str]:
    from scripts.kb_snapshot import snapshot_fields

    allowed = set(snapshot_fields(schema))
    return {field["name"]: str(field.get("type")) for field in schema["fields"]
            if field["name"] in allowed}


def _pending_publishes() -> list[str]:
    from scripts.kb_snapshot import fingerprint_path_for_snapshot
    from src.index.fingerprint_store import FingerprintStore

    path = fingerprint_path_for_snapshot(SOURCE_COLLECTION)
    if not path.exists():
        return []
    return FingerprintStore(str(path)).pending_sources()


def preflight(snapshot: Path, *, include_uncategorized: bool = False,
              client=None, engine=None) -> dict[str, Any]:
    from sqlalchemy import create_engine, text

    from src.core.config import DATABASE_URL, POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME

    client = client or _client()
    owns_engine = engine is None
    engine = engine or create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        plan = load_plan(snapshot, include_uncategorized=include_uncategorized)
        if not client.has_collection(SOURCE_COLLECTION):
            raise ValueError(f"源集合不存在：{SOURCE_COLLECTION}")
        pending = _pending_publishes()
        if pending:
            raise ValueError(f"存在未完成的指纹发布，先停止写入并处理：{pending}")
        targets = {}
        for name in (REPORT_COLLECTION_NAME, POLICY_COLLECTION_NAME):
            exists = bool(client.has_collection(name))
            count = _row_count(client, name) if exists else 0
            if count:
                raise ValueError(f"目标集合非空：{name}（{count} 行）；迁移只允许写入空集合")
            targets[name] = {"exists": exists, "row_count": count}
        source_schema = client.describe_collection(SOURCE_COLLECTION)
        if targets[REPORT_COLLECTION_NAME]["exists"]:
            target_schema = client.describe_collection(REPORT_COLLECTION_NAME)
            if _schema_signature(source_schema) != _schema_signature(target_schema):
                raise ValueError("finance_report 字段集与源集合不一致，停止迁移")
        return {
            "source_rows": _row_count(client, SOURCE_COLLECTION),
            "report_sources": plan.report_sources,
            "delete_sources": plan.delete_sources,
            "uncategorized": plan.uncategorized,
            "targets": targets,
            "source_schema": _schema_signature(source_schema),
        }
    finally:
        if owns_engine:
            engine.dispose()


def run_snapshot(snapshot: Path, *, apply: bool) -> dict[str, Any]:
    if not apply:
        return {"phase": "snapshot", "dry_run": True, "snapshot": str(snapshot)}
    from scripts.kb_snapshot import backup

    return backup(snapshot, collection=SOURCE_COLLECTION, allow_extra_sources=True)


def build_collections(*, apply: bool) -> dict[str, Any]:
    from src.core.config import POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
    from src.index.milvus_kb import get_knowledge_base

    if not apply:
        return {"phase": "build", "dry_run": True,
                "collections": [REPORT_COLLECTION_NAME, POLICY_COLLECTION_NAME]}
    for name in (REPORT_COLLECTION_NAME, POLICY_COLLECTION_NAME):
        get_knowledge_base(name).ensure_collection()
    return {"phase": "build", "collections": [REPORT_COLLECTION_NAME, POLICY_COLLECTION_NAME]}


def copy_milvus(snapshot: Path, *, apply: bool, batch_size: int = 200,
                client=None, plan: MigrationPlan | None = None) -> dict[str, Any]:
    from scripts.kb_snapshot import snapshot_fields
    from src.core.config import REPORT_COLLECTION_NAME

    plan = plan or load_plan(snapshot)
    client = client or _client()
    fields = snapshot_fields(client.describe_collection(SOURCE_COLLECTION))
    expression = f'category == "{REPORT_CATEGORY}"'
    if not plan.report_sources:
        return {"phase": "copy", "copied": 0, "skipped_existing": 0, "report_sources": [],
                "dry_run": not apply}
    if not apply:
        count = sum(1 for _ in _iter_rows(client, SOURCE_COLLECTION, expression, ["id"], batch_size))
        return {"phase": "copy", "dry_run": True, "would_copy": count,
                "report_sources": plan.report_sources}
    existing = {row["id"] for row in _iter_rows(client, REPORT_COLLECTION_NAME, "", ["id"], 1000)}
    copied = skipped = 0
    batch: list[dict[str, Any]] = []
    for row in _iter_rows(client, SOURCE_COLLECTION, expression, fields, batch_size):
        if row["id"] in existing:
            skipped += 1
            continue
        batch.append({key: row.get(key) for key in fields})
        if len(batch) >= batch_size:
            client.upsert(collection_name=REPORT_COLLECTION_NAME, data=batch)
            copied += len(batch)
            batch.clear()
    if batch:
        client.upsert(collection_name=REPORT_COLLECTION_NAME, data=batch)
        copied += len(batch)
    client.flush(REPORT_COLLECTION_NAME)
    return {"phase": "copy", "copied": copied, "skipped_existing": skipped,
            "report_sources": plan.report_sources}


def _copy_table(connection, table: str, columns: tuple[str, ...], target: str,
                source_collection: str, source: str) -> int:
    from sqlalchemy import text

    select_columns = ", ".join(columns[1:])
    update_columns = ", ".join(f"{name} = EXCLUDED.{name}" for name in columns[2:])
    sql = text(
        f'INSERT INTO {table} ({", ".join(columns)}) '
        f"SELECT :target, {select_columns} FROM {table} "
        f"WHERE collection = :source_collection AND source = :source "
        f"ON CONFLICT (collection, id) DO UPDATE SET {update_columns}"
    )
    result = connection.execute(sql, {"target": target, "source_collection": source_collection,
                                      "source": source})
    return int(result.rowcount or 0)


def copy_postgres(snapshot: Path, *, apply: bool, plan: MigrationPlan | None = None,
                  engine=None) -> dict[str, Any]:
    from sqlalchemy import create_engine, inspect, text

    from src.core.config import DATABASE_URL, REPORT_COLLECTION_NAME

    plan = plan or load_plan(snapshot)
    engine = engine or create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            tables = {name for name in ("parent_chunks", "table_chunks")
                      if inspect(connection).has_table(name)}
        counts = {}
        for table, columns in (("parent_chunks", PARENT_COLUMNS), ("table_chunks", TABLE_COLUMNS)):
            if table not in tables:
                counts[table] = {"dry_run": True, "rows": 0}
                continue
            if not apply:
                with engine.connect() as connection:
                    rows = 0
                    for source in plan.report_sources:
                        rows += int(connection.execute(
                            text(f"SELECT COUNT(*) FROM {table} WHERE collection=:c AND source=:s"),
                            {"c": SOURCE_COLLECTION, "s": source}).scalar() or 0)
                counts[table] = {"dry_run": True, "would_copy": rows}
                continue
            with engine.begin() as connection:
                rows = sum(_copy_table(connection, table, columns, REPORT_COLLECTION_NAME,
                                       SOURCE_COLLECTION, source)
                           for source in plan.report_sources)
            counts[table] = {"copied": rows}
        return {"phase": "copy_postgres", "report_sources": plan.report_sources, "tables": counts}
    finally:
        engine.dispose()


def rewrite_fingerprints(snapshot: Path, *, apply: bool, plan: MigrationPlan | None = None) -> dict[str, Any]:
    from scripts.kb_snapshot import fingerprint_path_for_snapshot

    plan = plan or load_plan(snapshot)
    path = fingerprint_path_for_snapshot(SOURCE_COLLECTION)
    if not path.exists():
        raise FileNotFoundError(f"指纹文件不存在：{path}")
    data = _read_json(path)
    report_sources = set(plan.report_sources)
    removed = sorted(key for key in data if key not in report_sources)
    if not apply:
        return {"phase": "fingerprints", "dry_run": True, "path": str(path),
                "kept": len(report_sources), "removed": len(removed)}
    kept = {key: value for key, value in data.items() if key in report_sources}
    _write_json_atomic(path, kept)
    return {"phase": "fingerprints", "path": str(path),
            "kept": sorted(kept), "removed": removed}


def _milvus_source_counts(client, collection: str, fields: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _iter_rows(client, collection, "", fields or ["source"], 1000):
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    return counts


def verify(snapshot: Path, *, plan: MigrationPlan | None = None, client=None,
           engine=None) -> dict[str, Any]:
    from sqlalchemy import create_engine, inspect, text

    from src.core.config import DATABASE_URL, POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
    from src.index.milvus_kb import get_knowledge_base

    plan = plan or load_plan(snapshot)
    client = client or _client()
    engine = engine or create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        expected = {source: count for source, count in snapshot_source_counts(snapshot).items()
                    if source in set(plan.report_sources)}
        actual = _milvus_source_counts(client, REPORT_COLLECTION_NAME)
        actual = {source: actual.get(source, 0) for source in expected}
        if actual != expected:
            raise ValueError(f"finance_report 行数不一致：{actual} != {expected}")
        forbidden = [row for row in _iter_rows(
            client, REPORT_COLLECTION_NAME, f'category in [{_quoted_sources(DELETE_CATEGORIES)}]',
            ["source", "category"], 500)]
        if forbidden:
            raise ValueError(f"finance_report 仍包含待删除类别：{forbidden[:5]}")
        if _row_count(client, POLICY_COLLECTION_NAME):
            raise ValueError("finance_policy 必须为空集合")
        pg = {}
        with engine.connect() as connection:
            inspector = inspect(connection)
            for table in ("parent_chunks", "table_chunks"):
                if not inspector.has_table(table):
                    pg[table] = "missing"
                    continue
                if not all(connection.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE collection=:c AND source=:s"),
                        {"c": REPORT_COLLECTION_NAME, "s": source}).scalar() for source in plan.report_sources):
                    raise ValueError(f"{table} 缺少部分来源")
                pg[table] = "ok"
        kb = get_knowledge_base(REPORT_COLLECTION_NAME)
        hits = kb.hybrid_search("新能源汽车", k=3)
        from src.api.schemas.search import SearchRequest
        from src.retrieve.search_service import run_search
        policy_empty = run_search("policy", SearchRequest(query="政策", page=1, page_size=5))
        if policy_empty.items:
            raise ValueError("finance_policy 检索应返回空 items")
        return {"phase": "verify", "report_rows": actual, "postgres": pg,
                "smoke_hits": len(hits), "policy_items": 0}
    finally:
        engine.dispose()


def delete_sources(snapshot: Path, *, apply: bool, plan: MigrationPlan | None = None,
                   client=None, engine=None, batch_size: int = 200) -> dict[str, Any]:
    from sqlalchemy import bindparam, create_engine, inspect, text

    from src.core.config import DATABASE_URL

    plan = plan or load_plan(snapshot)
    if not plan.delete_sources:
        return {"phase": "delete", "deleted_sources": [], "dry_run": not apply}
    if not apply:
        return {"phase": "delete", "dry_run": True, "delete_sources": plan.delete_sources}
    client = client or _client()
    engine = engine or create_engine(DATABASE_URL, pool_pre_ping=True)
    removed = {}
    try:
        for start in range(0, len(plan.delete_sources), batch_size):
            batch = plan.delete_sources[start:start + batch_size]
            client.delete(collection_name=SOURCE_COLLECTION,
                          filter=f"source in [{_quoted_sources(batch)}]")
            client.flush(SOURCE_COLLECTION)
            removed.update({source: "milvus" for source in batch})
        with engine.begin() as connection:
            inspector = inspect(connection)
            for source in plan.delete_sources:
                document_ids = [row[0] for row in connection.execute(
                    text("SELECT document_id FROM documents WHERE source=:source"), {"source": source})] \
                    if inspector.has_table("documents") else []
                if document_ids:
                    for table in ("document_chunks", "document_versions"):
                        if inspector.has_table(table):
                            key = "version_id" if table == "document_versions" else "document_id"
                            connection.execute(
                                text(f"DELETE FROM {table} WHERE {key} IN :ids").bindparams(
                                    bindparam("ids", expanding=True)),
                                {"ids": document_ids})
                    connection.execute(
                        text("DELETE FROM documents WHERE document_id IN :ids").bindparams(
                            bindparam("ids", expanding=True)), {"ids": document_ids})
                for table in ("parent_chunks", "table_chunks"):
                    if inspector.has_table(table):
                        connection.execute(
                            text(f"DELETE FROM {table} WHERE collection=:c AND source=:s"),
                            {"c": SOURCE_COLLECTION, "s": source})
        report = {"phase": "delete", "deleted_sources": plan.delete_sources,
                  "categories": {source: plan.categories.get(source, []) for source in plan.delete_sources},
                  "original_files": "retained"}
        _write_json_atomic(snapshot / "deleted_sources.json", report)
        return report
    finally:
        engine.dispose()


def retire(*, apply: bool, confirm: bool, client=None) -> dict[str, Any]:
    client = client or _client()
    if not apply or not confirm:
        return {"phase": "retire", "dry_run": True, "confirm_required": "--confirm-retire"}
    if not client.has_collection(SOURCE_COLLECTION):
        raise ValueError(f"源集合不存在或已 retire：{SOURCE_COLLECTION}")
    retired = f"{RETIRED_PREFIX}{datetime.now(UTC).strftime('%Y%m%d')}"
    if client.has_collection(retired):
        raise ValueError(f"retired 集合已存在：{retired}")
    client.rename_collection(old_name=SOURCE_COLLECTION, new_name=retired)
    return {"phase": "retire", "from": SOURCE_COLLECTION, "to": retired}


def drop(*, apply: bool, confirmation: str, client=None) -> dict[str, Any]:
    client = client or _client()
    if not apply or confirmation != SOURCE_COLLECTION:
        return {"phase": "drop", "dry_run": True, "confirm_required": f"--confirm-drop {SOURCE_COLLECTION}"}
    if client.has_collection(SOURCE_COLLECTION):
        client.drop_collection(collection_name=SOURCE_COLLECTION)
        return {"phase": "drop", "dropped": SOURCE_COLLECTION}
    # retire 之后源集合已被重命名；验收通过后再删除那份回滚副本。
    retired = sorted(
        name for name in client.list_collections()
        if name.startswith(RETIRED_PREFIX)
    )
    if not retired:
        return {"phase": "drop", "dropped": None, "reason": "source collection not found"}
    if len(retired) > 1:
        raise ValueError(f"存在多个 retired 集合，需人工确认后逐个删除：{retired}")
    client.drop_collection(collection_name=retired[0])
    return {"phase": "drop", "dropped": retired[0], "from": SOURCE_COLLECTION}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--apply", action="store_true", help="默认 dry-run，不加此参数不写库")
    parser.add_argument("--resume", action="store_true", help="允许重复执行已完成阶段（操作本身幂等）")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--include-uncategorized", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--confirm-retire", action="store_true")
    parser.add_argument("--confirm-drop", default="")
    args = parser.parse_args(argv)
    snapshot = Path(args.snapshot).resolve()
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须 >= 1")
    if args.phase in ("delete", "retire", "drop") and not args.apply:
        raise SystemExit(f"阶段 {args.phase} 必须显式 --apply")
    plan = load_plan(snapshot, include_uncategorized=args.include_uncategorized) \
        if args.phase not in ("snapshot",) else None
    if args.phase == "preflight":
        result = preflight(snapshot, include_uncategorized=args.include_uncategorized)
    elif args.phase == "snapshot":
        result = run_snapshot(snapshot, apply=args.apply)
    elif args.phase == "build":
        result = build_collections(apply=args.apply)
    elif args.phase == "copy":
        result = {"milvus": copy_milvus(snapshot, apply=args.apply, batch_size=args.batch_size, plan=plan),
                  "postgres": copy_postgres(snapshot, apply=args.apply, plan=plan)}
    elif args.phase == "fingerprints":
        result = rewrite_fingerprints(snapshot, apply=args.apply, plan=plan)
    elif args.phase == "verify":
        result = verify(snapshot, plan=plan)
    elif args.phase == "delete":
        if not args.confirm_delete:
            raise SystemExit("删除三类数据不可逆，必须加 --confirm-delete")
        result = delete_sources(snapshot, apply=args.apply, plan=plan, batch_size=args.batch_size)
    elif args.phase == "retire":
        if not args.confirm_retire:
            raise SystemExit("retire 会重命名源集合，必须加 --confirm-retire")
        result = retire(apply=args.apply, confirm=args.confirm_retire)
    else:
        result = drop(apply=args.apply, confirmation=args.confirm_drop)
    if args.apply:
        _save_state(snapshot, args.phase, {"apply": args.apply, "result": result})
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
