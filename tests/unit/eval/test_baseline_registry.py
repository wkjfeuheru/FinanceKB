"""冻结基线注册表：加载、校验、漂移与噪声下限。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.baseline import (
    list_baselines,
    load_baseline,
    measure_noise_floor,
    resolve_profile,
    verify_baseline,
)
from src.eval.manifest import PROTOCOL_FINGERPRINT


class _OfflineKB:
    """离线假 KB：任何需要连库的调用都失败，验证降级为 unavailable。"""

    collection_name = "finance_kb"

    def _get_client(self):
        raise RuntimeError("offline")

    def list_documents(self):
        raise RuntimeError("offline")

    def get_fingerprint_store(self):
        raise RuntimeError("offline")


def _write_baseline(root, *, dataset_path: Path, dataset_sha: str) -> Path:
    version_dir = root / "b1" / "v1"
    version_dir.mkdir(parents=True)
    manifest = {
        "profile": "b1",
        "label": "生产默认",
        "version": "v1",
        "strategy": "PRODUCTION",
        "policy": {},
        "frozen_at": "2026-10-25T22:00:00+08:00",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha,
            "fingerprint": "dataset-fp",
            "case_count": 2,
            "tier_counts": {"gold": 2},
        },
        "kb": {
            "kb_fingerprint": "unavailable",
            "pdf_set_fingerprint": "unavailable",
            "chunk_content_fingerprint": "unavailable",
            "collection": "finance_kb",
        },
        "code": {"worktree_sha256": "unavailable"},
        "config": {},
        "judge": {"model": "deepseek:deepseek-chat", "fingerprint": "unavailable"},
        "protocol": {"version": "agentic-v2", "fingerprint": PROTOCOL_FINGERPRINT},
        "metrics_version": "2.0",
        "noise_floor": {"mrr": 0.02},
        "summary": {"n": 2},
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        {"case_id": "g1", "tier": "gold", "stratum": "explicit", "metrics": {"mrr": 0.2}},
        {"case_id": "g2", "tier": "gold", "stratum": "cross_doc", "metrics": {"mrr": 0.4}},
    ]
    (version_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (version_dir / "summary.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    return version_dir


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="未知基线 profile"):
        resolve_profile("b9")


def test_load_and_list_frozen_baseline(tmp_path, monkeypatch):
    root = tmp_path / "baselines"
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text('{"id": "g1"}\n', encoding="utf-8")
    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    _write_baseline(root, dataset_path=dataset, dataset_sha=sha)
    monkeypatch.setenv("EVAL_BASELINE_DIR", str(root))

    rows = list_baselines()
    assert len(rows) == 1 and rows[0]["profile"] == "b1"

    data = load_baseline("b1")
    assert len(data["records"]) == 2
    assert data["manifest"]["noise_floor"] == {"mrr": 0.02}


def test_verify_reports_dataset_drift(tmp_path, monkeypatch):
    root = tmp_path / "baselines"
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text('{"id": "g1"}\n', encoding="utf-8")
    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    _write_baseline(root, dataset_path=dataset, dataset_sha=sha)
    monkeypatch.setenv("EVAL_BASELINE_DIR", str(root))

    ok = verify_baseline("b1", kb=_OfflineKB())
    assert ok["ok"] is True
    assert ok["drift"] == []

    dataset.write_text('{"id": "g1", "changed": true}\n', encoding="utf-8")
    drifted = verify_baseline("b1", kb=_OfflineKB())
    assert drifted["ok"] is False
    assert drifted["drift"][0]["field"] == "dataset.sha256"


def test_measure_noise_floor_uses_paired_absolute_delta():
    left = [{"case_id": f"c{i}", "metrics": {"mrr": 0.2, "recall_at_5": 0.5}} for i in range(4)]
    right = [
        {"case_id": f"c{i}", "metrics": {"mrr": 0.25 if i % 2 else 0.15, "recall_at_5": 0.5}}
        for i in range(4)
    ]
    floor = measure_noise_floor(left, right, metrics=("mrr", "recall_at_5"))
    assert floor["mrr"] == pytest.approx(0.05)
    assert floor["recall_at_5"] == 0.0
