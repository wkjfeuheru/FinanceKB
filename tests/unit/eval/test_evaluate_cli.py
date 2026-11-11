"""统一评测 CLI 的离线命令与报告协议。"""

from __future__ import annotations

import json

from src.eval.cli import main
from src.eval.dataset import EvaluationDataset


def _row(case_id="case-1", *, tier="silver", status="auto_pass"):
    return {
        "id": case_id, "query": f"问题 {case_id}？", "question_type": "single_hop",
        "difficulty": "medium", "answer_type": "factual", "reference_answer": "答案",
        "required_facts": ["事实"],
        "evidence": [{"source": "a.pdf", "document_version": "v1",
                      "chunk_id": f"{case_id}-chunk", "quote": "证据"}],
        "expected_behavior": "", "tier": tier, "review_status": status,
        "reviewer": "", "review_note": "", "provenance": {"kb_fingerprint": "kb-v1"},
    }


def test_dataset_validate_and_export_review_commands(tmp_path):
    dataset_path = tmp_path / "cases.jsonl"
    review_path = tmp_path / "review.csv"
    EvaluationDataset.from_dicts([_row()]).save(dataset_path)

    assert main(["dataset", "validate", "--dataset", str(dataset_path)]) == 0
    assert main([
        "dataset", "export-review", "--dataset", str(dataset_path),
        "--out", str(review_path),
    ]) == 0
    assert review_path.exists()


def test_ab_run_compares_existing_checkpoint_records(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    out = tmp_path / "comparison"
    baseline.mkdir()
    candidate.mkdir()
    base_record = {
        "case_id": "g1", "tier": "gold",
        "metrics": {"recall_at_5": 0.0, "precision_at_5": 0.0, "mrr": 0.0},
    }
    candidate_record = {
        "case_id": "g1", "tier": "gold",
        "metrics": {"recall_at_5": 1.0, "precision_at_5": 0.2, "mrr": 1.0},
    }
    (baseline / "records.jsonl").write_text(json.dumps(base_record) + "\n", encoding="utf-8")
    (candidate / "records.jsonl").write_text(json.dumps(candidate_record) + "\n", encoding="utf-8")

    result = main([
        "ab", "run", "--baseline", str(baseline), "--candidate", str(candidate),
        "--experiment", "hybrid_vs_dense", "--out", str(out), "--bootstrap", "100",
    ])

    assert result == 0
    report = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert report["decision"] == "pass"
    assert (out / "comparison.md").exists()
