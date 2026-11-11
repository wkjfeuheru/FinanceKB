"""集合迁移的真实 Milvus 演练（默认跳过）。

在临时集合上验证「复制 → 重跑幂等 → retire」，不触碰 ``finance_kb`` 与
生产目标集合。执行::

    FINANCEKB_LIVE_MIGRATION=1 pytest tests/live/test_data_platform_migration.py -m live
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from scripts import migrate_to_data_platform as migration
from src.core import config as core_config

pytestmark = pytest.mark.live

if os.getenv("FINANCEKB_LIVE_MIGRATION") != "1":
    pytest.skip(
        "设置 FINANCEKB_LIVE_MIGRATION=1 后才在真实 Milvus 上演练迁移",
        allow_module_level=True,
    )

SOURCE = "finance_kb_pytest_migration"
REPORT = "finance_report_pytest_migration"
POLICY = "finance_policy_pytest_migration"
SOURCE_FILE = "pytest-migration.pdf"


def _client():
    from src.index.milvus_kb import get_knowledge_base

    return get_knowledge_base(migration.SOURCE_COLLECTION)._get_client()


def _insert_synthetic_row(client) -> None:
    from pymilvus import DataType

    from src.core.config import EMBEDDING_DIM

    schema = client.describe_collection(SOURCE)
    row: dict[str, object] = {}
    for field in schema["fields"]:
        name = field["name"]
        kind = field["type"]
        if name == "id":
            row[name] = "pytest-migration-1"
        elif name == "dense_vector":
            row[name] = [0.01] * EMBEDDING_DIM
        elif kind == DataType.VARCHAR:
            row[name] = ""
        elif kind == DataType.INT64:
            row[name] = 0
        elif kind == DataType.BOOL:
            row[name] = True
        elif kind == DataType.FLOAT:
            row[name] = 0.0
        elif kind == DataType.DOUBLE:
            row[name] = 0.0
        # sparse_vector 由 BM25 Function 生成，不能手动写入
    row.update(
        {
            "content": "新能源汽车行业迁移演练",
            "source": SOURCE_FILE,
            "title": "迁移演练",
            "category": migration.REPORT_CATEGORY,
            "is_current": True,
        }
    )
    client.insert(collection_name=SOURCE, data=[row])
    client.flush(SOURCE)


def _snapshot(tmp_path: Path) -> Path:
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "sources_by_category.json").write_text(
        json.dumps(
            {
                "milvus": {SOURCE_FILE: [migration.REPORT_CATEGORY]},
                "postgres": {SOURCE_FILE: [migration.REPORT_CATEGORY]},
                "unknown": [],
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(directory / "vectors.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "pytest-migration-1", "source": SOURCE_FILE}) + "\n")
    return directory


@pytest.fixture
def temp_collections(monkeypatch: pytest.MonkeyPatch):
    from src.index.milvus_kb import get_knowledge_base

    monkeypatch.setattr(migration, "SOURCE_COLLECTION", SOURCE)
    monkeypatch.setattr(core_config, "REPORT_COLLECTION_NAME", REPORT)
    monkeypatch.setattr(core_config, "POLICY_COLLECTION_NAME", POLICY)

    for name in (SOURCE, REPORT, POLICY):
        if _client().has_collection(name):
            _client().drop_collection(collection_name=name)

    get_knowledge_base(SOURCE).ensure_collection()
    _insert_synthetic_row(_client())
    get_knowledge_base(REPORT).ensure_collection()
    get_knowledge_base(POLICY).ensure_collection()
    try:
        yield
    finally:
        client = _client()
        for name in (SOURCE, REPORT, POLICY):
            if client.has_collection(name):
                client.drop_collection(collection_name=name)
        retired_prefix = migration.RETIRED_PREFIX
        for name in list(getattr(client, "list_collections", lambda: [])()):
            if name.startswith(retired_prefix) and "pytest" in name:
                client.drop_collection(collection_name=name)


def test_copy_is_idempotent_and_retire_renames(
    tmp_path: Path, temp_collections, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    plan = migration.load_plan(snapshot)
    client = _client()

    first = migration.copy_milvus(snapshot, apply=True, client=client, plan=plan)
    assert first["copied"] == 1
    second = migration.copy_milvus(snapshot, apply=True, client=client, plan=plan)
    assert second["copied"] == 0
    assert second["skipped_existing"] == 1

    retired = migration.retire(apply=True, confirm=True, client=client)
    assert retired["to"].startswith(migration.RETIRED_PREFIX)
    assert client.has_collection(retired["to"])
    # 夹具清理时 SOURCE 已不存在，改名回源集合以便统一 drop
    client.rename_collection(old_name=retired["to"], new_name=SOURCE)
