"""统一 JSONL 评测集的数据契约与冻结门禁。"""

from __future__ import annotations

import csv
import json

import pytest

from src.eval.dataset import (
    DatasetValidationError,
    EvaluationDataset,
)
from src.eval.test_set import TestSetLoader
from src.eval.migration import migrate_markdown_dataset


def _case(
    case_id: str,
    *,
    question_type: str = "single_hop",
    tier: str = "gold",
    review_status: str = "approved",
    sources: tuple[str, ...] = ("含，逗号的完整文件名.pdf",),
) -> dict:
    evidence = [
        {
            "source": source,
            "document_version": "v1",
            "chunk_id": f"{case_id}-chunk-{index}",
            "quote": f"证据 {index}",
        }
        for index, source in enumerate(sources, 1)
    ]
    return {
        "id": case_id,
        "query": f"问题 {case_id}？",
        "question_type": question_type,
        "difficulty": "medium",
        "answer_type": "factual",
        "reference_answer": "参考答案",
        "required_facts": ["事实一"],
        "evidence": evidence,
        "expected_behavior": "",
        "tier": tier,
        "review_status": review_status,
        "reviewer": "reviewer" if review_status == "approved" else "",
        "review_note": "",
        "provenance": {
            "generator_model": "test",
            "generated_at": "2026-09-19T00:00:00+08:00",
            "kb_fingerprint": "kb-v1",
        },
    }


def test_jsonl_preserves_source_names_containing_commas(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(_case("eval-0001"), ensure_ascii=False) + "\n", encoding="utf-8")

    dataset = EvaluationDataset.load(path)

    assert dataset.cases[0].evidence[0].source == "含，逗号的完整文件名.pdf"


def test_load_filters_tier_and_review_status(tmp_path):
    path = tmp_path / "dataset.jsonl"
    rows = [
        _case("gold-approved"),
        _case("gold-pending", review_status="pending"),
        _case("silver-auto", tier="silver", review_status="auto_pass"),
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    dataset = EvaluationDataset.load(path, tier="gold", review_status="approved")

    assert [case.id for case in dataset.cases] == ["gold-approved"]


def test_validate_rejects_multihop_evidence_from_one_document():
    row = _case(
        "multi-1",
        question_type="multi_hop",
        sources=("same.pdf", "same.pdf"),
    )
    dataset = EvaluationDataset.from_dicts([row])

    with pytest.raises(DatasetValidationError, match="不同文档"):
        dataset.validate()


def test_online_validation_checks_chunk_source_and_quote():
    dataset = EvaluationDataset.from_dicts([_case("eval-0001")])

    audit = dataset.validate(
        evidence_resolver=lambda chunk_id: {
            "id": chunk_id,
            "source": "含，逗号的完整文件名.pdf",
            "content": "这里包含证据 1的完整内容",
        }
    )

    assert audit.valid is True
    assert audit.checked_evidence == 1


def test_freeze_rejects_pending_gold(tmp_path):
    dataset = EvaluationDataset.from_dicts([
        _case("gold-pending", review_status="pending"),
        _case("silver", tier="silver", review_status="auto_pass"),
    ])

    with pytest.raises(DatasetValidationError, match="Gold.*approved"):
        dataset.freeze(
            tmp_path / "frozen.jsonl",
            quotas={"gold": {"single_hop": 1}, "silver": {"single_hop": 1}},
            kb_fingerprint="kb-v1",
            max_source_share=1.0,
        )


def test_freeze_writes_exact_quotas_and_manifest(tmp_path):
    rows = [
        _case("g-single", sources=("a.pdf",)),
        _case("g-multi", question_type="multi_hop", sources=("a.pdf", "b.pdf")),
        _case("g-neg", question_type="negative", sources=()),
        _case("s-single", tier="silver", review_status="auto_pass", sources=("b.pdf",)),
    ]
    rows[2]["reference_answer"] = ""
    rows[2]["required_facts"] = []
    rows[2]["expected_behavior"] = "应拒绝回答"
    dataset = EvaluationDataset.from_dicts(rows)
    output = tmp_path / "frozen.jsonl"

    manifest = dataset.freeze(
        output,
        quotas={
            "gold": {"single_hop": 1, "multi_hop": 1, "negative": 1},
            "silver": {"single_hop": 1},
        },
        kb_fingerprint="kb-v1",
        dataset_version="2026-09-test",
        max_source_share=1.0,
    )

    assert len(EvaluationDataset.load(output).cases) == 4
    assert manifest["dataset_version"] == "2026-09-test"
    assert manifest["counts"]["gold"]["multi_hop"] == 1
    assert json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8")) == manifest


def test_review_round_trip_updates_only_review_fields(tmp_path):
    dataset = EvaluationDataset.from_dicts([_case("eval-1", review_status="pending")])
    review_path = tmp_path / "review.csv"
    dataset.export_review(review_path)
    rows = list(csv.DictReader(review_path.open(encoding="utf-8-sig", newline="")))
    rows[0]["review_status"] = "approved"
    rows[0]["reviewer"] = "alice"
    rows[0]["review_note"] = "已核对"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    updated = dataset.import_review(review_path)

    assert updated.cases[0].review_status == "approved"
    assert updated.cases[0].reviewer == "alice"
    assert updated.cases[0].query == dataset.cases[0].query


def test_legacy_loader_reads_jsonl_without_splitting_filename_commas(tmp_path):
    path = tmp_path / "dataset.jsonl"
    EvaluationDataset.from_dicts([_case("eval-comma")]).save(path)

    entry = TestSetLoader(path).load_test_set()[0]

    assert entry.related_docs == "含，逗号的完整文件名.pdf"
    assert entry.chunk_ids == ("eval-comma-chunk-1",)
    assert entry.tier == "gold"


def test_legacy_markdown_loader_emits_deprecation_warning(tmp_path):
    path = tmp_path / "legacy.md"
    path.write_text(
        "## 1. 问题是什么？\n\n- 问题类型：negative\n"
        "- 期望行为：拒绝回答\n\n### 标准答案\n\n无法回答\n\n---\n",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="Markdown"):
        entries = TestSetLoader(path).load_test_set()

    assert len(entries) == 1


def test_markdown_migration_uses_chunk_source_instead_of_splitting_filename(tmp_path):
    path = tmp_path / "legacy.md"
    path.write_text(
        "## 1. 资本要求是什么？\n\n- 问题类型：single_hop\n"
        "- 相关文档：关于证券公司，风险控制指标的规定.pdf\n"
        "- 证据chunk：chunk-1\n\n### 标准答案\n\n资本要求答案。\n\n---\n",
        encoding="utf-8",
    )
    resolver = lambda _chunk_id: {
        "source": "关于证券公司，风险控制指标的规定.pdf",
        "version": "v2",
        "content": "资本要求答案。更多证据内容。",
    }

    dataset = migrate_markdown_dataset(path, resolver, kb_fingerprint="kb-v2")

    assert dataset.cases[0].evidence[0].source == "关于证券公司，风险控制指标的规定.pdf"
    assert dataset.cases[0].evidence[0].document_version == "v2"
