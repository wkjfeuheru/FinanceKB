from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
from src.index import fingerprint_store
from src.index.fingerprint_store import FingerprintStore, fingerprint_path_for


@pytest.fixture
def base_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "fingerprints.json"
    monkeypatch.setattr(fingerprint_store, "MILVUS_FINGERPRINT_PATH", str(base))
    return base


def test_default_collection_reuses_base_file(base_path: Path) -> None:
    assert fingerprint_path_for(REPORT_COLLECTION_NAME) == str(base_path)


def test_non_default_collections_get_isolated_files(base_path: Path) -> None:
    policy = fingerprint_path_for(POLICY_COLLECTION_NAME)
    legacy = fingerprint_path_for("finance_kb")
    assert policy == f"{base_path}_{POLICY_COLLECTION_NAME}"
    assert legacy == f"{base_path}_finance_kb"
    assert len({str(base_path), policy, legacy}) == 3


def test_report_and_policy_stores_do_not_read_each_other(base_path: Path) -> None:
    report_path = fingerprint_path_for(REPORT_COLLECTION_NAME)
    policy_path = fingerprint_path_for(POLICY_COLLECTION_NAME)
    report = FingerprintStore(report_path)
    policy = FingerprintStore(policy_path)

    report.update("a.pdf", "hash-a", "a.pdf", 3)
    policy.update("b.pdf", "hash-b", "b.pdf", 2)

    assert report.is_unchanged("a.pdf", "hash-a") is True
    assert report.is_unchanged("b.pdf", "hash-b") is False
    assert policy.is_unchanged("b.pdf", "hash-b") is True
    assert policy.is_unchanged("a.pdf", "hash-a") is False

    # 重新打开后仍各自独立
    assert FingerprintStore(report_path).get("b.pdf") is None
    assert FingerprintStore(policy_path).get("a.pdf") is None
    assert FingerprintStore(report_path).get("a.pdf")["file_path"] == "a.pdf"


def test_pending_publish_is_scoped_to_its_collection(base_path: Path) -> None:
    report = FingerprintStore(fingerprint_path_for(REPORT_COLLECTION_NAME))
    policy = FingerprintStore(fingerprint_path_for(POLICY_COLLECTION_NAME))
    report.begin_publish("a.pdf", sha256="hash-a", chunk_ids=["1"], previous_ids=[])

    assert report.has_pending_publish("a.pdf") is True
    assert policy.has_pending_publish("a.pdf") is False
    assert report.is_unchanged("a.pdf", "hash-a") is False
