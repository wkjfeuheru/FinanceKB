import hashlib
import json

import pytest

from scripts.kb_snapshot import verify_snapshot


def test_snapshot_reads_dense_and_scalars_not_generated_sparse():
    from scripts.kb_snapshot import snapshot_fields
    schema = {"fields": [{"name": name} for name in ("id", "content", "dense_vector", "sparse_vector")],
              "functions": [{"output_field_names": ["sparse_vector"]}]}
    assert snapshot_fields(schema) == ["id", "content", "dense_vector"]


def test_snapshot_rejects_modified_artifact(tmp_path):
    artifact = tmp_path / "rows.jsonl"
    artifact.write_text('{"id":"a"}\n', encoding="utf-8")
    manifest = {"collection": "finance_kb", "artifacts": {
        "rows.jsonl": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "count": 1}
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifact.write_text('{"id":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_snapshot_rejects_unsafe_paths_and_wrong_collection(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"collection": "other", "artifacts": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="finance_kb"):
        verify_snapshot(tmp_path)
    path.write_text(json.dumps({"collection": "finance_kb", "artifacts": {"../escape": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        verify_snapshot(tmp_path)


def test_snapshot_verifies_jsonl_count(tmp_path):
    artifact = tmp_path / "rows.jsonl"
    artifact.write_text('{"id":"a"}\n', encoding="utf-8")
    manifest = {"collection": "finance_kb", "artifacts": {
        "rows.jsonl": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "count": 2}
    }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        verify_snapshot(tmp_path)


def test_snapshot_rejects_digest_valid_but_incomplete_backup(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'collection': 'finance_kb', 'artifacts': {}}), encoding='utf8')
    with pytest.raises(ValueError, match='incomplete'):
        verify_snapshot(tmp_path)


def test_restore_skips_removed_knowledge_bases_table(tmp_path, monkeypatch):
    from src.core import config as core_config
    from scripts.kb_snapshot import _restore_postgres

    monkeypatch.setattr(core_config, "DATABASE_URL", f"sqlite:///{tmp_path / 'restore.db'}")
    (tmp_path / "knowledge_bases.jsonl").write_text('{"id":"kb"}\n', encoding="utf-8")

    _restore_postgres(
        tmp_path, {"knowledge_bases.jsonl": {"sha256": "ignored", "count": 1}}
    )


def test_restore_rejects_missing_core_table(tmp_path, monkeypatch):
    from src.core import config as core_config
    from scripts.kb_snapshot import _restore_postgres

    monkeypatch.setattr(core_config, "DATABASE_URL", f"sqlite:///{tmp_path / 'restore.db'}")
    (tmp_path / "documents.jsonl").write_text('{"document_id":"doc"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist"):
        _restore_postgres(tmp_path, {"documents.jsonl": {"sha256": "ignored", "count": 1}})
