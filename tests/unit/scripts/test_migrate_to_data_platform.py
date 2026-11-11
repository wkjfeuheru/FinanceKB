from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import migrate_to_data_platform as migration
from src.core import config as core_config

REPORT = migration.REPORT_CATEGORY
DELETE = sorted(migration.DELETE_CATEGORIES)


def _snapshot(tmp_path: Path, categories: dict[str, Any]) -> Path:
    directory = tmp_path / "snap"
    directory.mkdir()
    (directory / "sources_by_category.json").write_text(
        json.dumps(categories, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _plan(tmp_path: Path) -> migration.MigrationPlan:
    snapshot = _snapshot(
        tmp_path,
        {
            "milvus": {"report.pdf": [REPORT]},
            "postgres": {"report.pdf": [REPORT], "old.pdf": [DELETE[-1]]},
        },
    )
    return migration.load_plan(snapshot)


def _write_fingerprints(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "fingerprints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------- partitioning


def test_partition_splits_report_and_delete_sources(tmp_path: Path) -> None:
    categories = {
        "milvus": {
            "report.pdf": [REPORT],
            "compliance.pdf": ["compliance_risk"],
            "ops.pdf": ["operations"],
        },
        "postgres": {"management.pdf": ["management"], "report.pdf": [REPORT]},
    }
    plan = migration.partition_sources(categories)
    assert plan.report_sources == ["report.pdf"]
    assert plan.delete_sources == ["compliance.pdf", "management.pdf", "ops.pdf"]


def test_same_source_must_not_cross_categories() -> None:
    with pytest.raises(ValueError, match="跨多个类别"):
        migration.partition_sources(
            {"milvus": {"a.pdf": [REPORT]}, "postgres": {"a.pdf": ["operations"]}}
        )
    with pytest.raises(ValueError, match="跨多个类别"):
        migration.partition_sources({"milvus": {"a.pdf": [REPORT, "operations"]}})


def test_unknown_category_aborts() -> None:
    with pytest.raises(ValueError, match="未知类别"):
        migration.partition_sources({"unknown": ["a.pdf"]})


def test_uncategorized_requires_explicit_flag() -> None:
    categories = {"milvus": {"a.pdf": [""]}}
    with pytest.raises(ValueError, match="未分类"):
        migration.partition_sources(categories)
    plan = migration.partition_sources(categories, include_uncategorized=True)
    assert plan.uncategorized == ["a.pdf"]
    assert plan.delete_sources == ["a.pdf"]


def test_unrecognized_category_values_abort() -> None:
    with pytest.raises(ValueError, match="无法归类"):
        migration.partition_sources({"milvus": {"a.pdf": ["weird"]}})


# --------------------------------------------------------------------- snapshot


def test_snapshot_source_counts_reads_gzip_jsonl(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    rows = [{"id": "1", "source": "a.pdf"}, {"id": "2", "source": "a.pdf"}, {"id": 3, "source": "b.pdf"}]
    with gzip.open(snapshot / "vectors.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert migration.snapshot_source_counts(snapshot) == {"a.pdf": 2, "b.pdf": 1}


def test_missing_source_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sources_by_category.json"):
        migration.load_plan(tmp_path)


# ------------------------------------------------------------------ dry runs


def test_snapshot_dry_run_writes_nothing(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap"
    result = migration.run_snapshot(snapshot, apply=False)
    assert result["dry_run"] is True
    assert not snapshot.exists()


def test_build_dry_run_does_not_touch_milvus(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(
        "src.index.milvus_kb.get_knowledge_base",
        lambda name: called.append(name),
    )
    result = migration.build_collections(apply=False)
    assert result["dry_run"] is True
    assert called == []


def test_delete_dry_run_lists_sources_without_writing(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = migration.delete_sources(tmp_path / "snap", apply=False, plan=plan)
    assert result == {"phase": "delete", "dry_run": True, "delete_sources": ["old.pdf"]}
    assert not (tmp_path / "snap" / "deleted_sources.json").exists()


def test_copy_dry_run_counts_without_upsert(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    client = FakeMilvus(rows=[_row(1, REPORT), _row(2, REPORT)])
    result = migration.copy_milvus(tmp_path / "snap", apply=False, client=client, plan=plan)
    assert result["dry_run"] is True
    assert result["would_copy"] == 2
    assert client.upserts == []


# --------------------------------------------------------------------- copy


class FakeIterator:
    def __init__(self, rows: list[dict[str, Any]], batch_size: int) -> None:
        self._rows = rows
        self._batch = batch_size

    def next(self) -> list[dict[str, Any]]:
        if not self._rows:
            return []
        batch, self._rows = self._rows[: self._batch], self._rows[self._batch:]
        return batch

    def close(self) -> None:
        pass


class FakeMilvus:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.upserts: list[dict[str, Any]] = []
        self.flushes: list[str] = []

    def describe_collection(self, collection: str) -> dict[str, Any]:
        return {
            "fields": [
                {"name": name, "type": 5}
                for name in (
                    "id",
                    "source",
                    "category",
                    "content",
                    "dense_vector",
                )
            ]
        }

    def query_iterator(self, *, collection_name, filter, output_fields, batch_size):
        rows = self.rows if "REPORT" in filter or collection_name == "finance_report" else self.rows
        if collection_name == "finance_report":
            rows = []
        return FakeIterator([dict(row) for row in rows], batch_size)

    def upsert(self, *, collection_name, data):
        self.upserts.append({"collection": collection_name, "data": [dict(row) for row in data]})

    def flush(self, collection_name):
        self.flushes.append(collection_name)


def _row(index: int, category: str) -> dict[str, Any]:
    return {
        "id": f"chunk-{index}",
        "source": "report.pdf",
        "category": category,
        "content": f"内容 {index}",
        "dense_vector": [0.1] * 1024,
    }


def test_copy_is_idempotent_by_primary_key(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    client = FakeMilvus(rows=[_row(1, REPORT), _row(2, REPORT)])
    first = migration.copy_milvus(
        tmp_path / "snap", apply=True, batch_size=1, client=client, plan=plan
    )
    assert first["copied"] == 2
    assert first["skipped_existing"] == 0
    assert [batch["collection"] for batch in client.upserts] == [
        "finance_report",
        "finance_report",
    ]
    assert client.flushes == ["finance_report"]

    # 再次执行时目标集合已包含这些主键 -> 全部跳过
    client.query_iterator = lambda **kwargs: (  # type: ignore[method-assign]
        FakeIterator([{"id": "chunk-1"}, {"id": "chunk-2"}], 1000)
        if kwargs["collection_name"] == "finance_report"
        else FakeIterator([_row(1, REPORT), _row(2, REPORT)], 1000)
    )
    client.upserts.clear()
    second = migration.copy_milvus(
        tmp_path / "snap", apply=True, client=client, plan=plan
    )
    assert second["copied"] == 0
    assert second["skipped_existing"] == 2
    assert client.upserts == []


# --------------------------------------------------------------- fingerprints


def test_fingerprint_rewrite_keeps_only_report_sources_on_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "fingerprints.json"
    monkeypatch.setattr(core_config, "MILVUS_FINGERPRINT_PATH", str(base))
    _write_fingerprints(
        tmp_path,
        {
            "report.pdf": {"sha256": "a"},
            "old.pdf": {"sha256": "b"},
            "__pending_publish__": {"old.pdf": {}},
        },
    )
    plan = _plan(tmp_path)

    dry = migration.rewrite_fingerprints(tmp_path / "snap", apply=False, plan=plan)
    assert dry["dry_run"] is True
    assert sorted(json.loads(base.read_text(encoding="utf-8"))) == [
        "__pending_publish__",
        "old.pdf",
        "report.pdf",
    ]

    applied = migration.rewrite_fingerprints(tmp_path / "snap", apply=True, plan=plan)
    assert applied["kept"] == ["report.pdf"]
    assert sorted(applied["removed"]) == ["__pending_publish__", "old.pdf"]
    assert json.loads(base.read_text(encoding="utf-8")) == {"report.pdf": {"sha256": "a"}}


# ------------------------------------------------------------ confirmations


class FakeAdminClient:
    def __init__(self, collections: set[str]) -> None:
        self.collections = set(collections)
        self.renamed: list[tuple[str, str]] = []
        self.dropped: list[str] = []

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def list_collections(self) -> list[str]:
        return sorted(self.collections)

    def rename_collection(self, *, old_name: str, new_name: str) -> None:
        self.collections.remove(old_name)
        self.collections.add(new_name)
        self.renamed.append((old_name, new_name))

    def drop_collection(self, *, collection_name: str) -> None:
        self.collections.discard(collection_name)
        self.dropped.append(collection_name)


def test_retire_requires_apply_and_confirmation() -> None:
    client = FakeAdminClient({"finance_kb"})
    assert migration.retire(apply=True, confirm=False, client=client)["dry_run"] is True
    assert migration.retire(apply=False, confirm=True, client=client)["dry_run"] is True
    assert client.renamed == []

    result = migration.retire(apply=True, confirm=True, client=client)
    assert result["to"].startswith("finance_kb_retired_")
    assert client.renamed and client.renamed[0][0] == "finance_kb"


def test_drop_requires_matching_confirmation() -> None:
    client = FakeAdminClient({"finance_kb"})
    assert migration.drop(apply=True, confirmation="wrong", client=client)["dry_run"] is True
    assert client.dropped == []
    migration.drop(apply=True, confirmation="finance_kb", client=client)
    assert client.dropped == ["finance_kb"]


def test_drop_targets_retired_copy_after_retire() -> None:
    client = FakeAdminClient({"finance_kb_retired_20261120"})
    result = migration.drop(apply=True, confirmation="finance_kb", client=client)
    assert result["dropped"] == "finance_kb_retired_20261120"
    assert client.dropped == ["finance_kb_retired_20261120"]


def test_cli_rejects_destructive_phase_without_apply(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, {"milvus": {"report.pdf": [REPORT]}})
    with pytest.raises(SystemExit, match="--apply"):
        migration.main(["--snapshot", str(snapshot), "--phase", "delete"])
    with pytest.raises(SystemExit, match="--confirm-delete"):
        migration.main(["--snapshot", str(snapshot), "--phase", "delete", "--apply"])
    with pytest.raises(SystemExit, match="--confirm-retire"):
        migration.main(["--snapshot", str(snapshot), "--phase", "retire", "--apply"])


def test_cli_dry_run_does_not_write_state(tmp_path: Path, capsys) -> None:
    snapshot = _snapshot(tmp_path, {"milvus": {"report.pdf": [REPORT]}})
    assert migration.main(["--snapshot", str(snapshot), "--phase", "build"]) == 0
    assert not (snapshot / "state.json").exists()
    assert '"dry_run": true' in capsys.readouterr().out
