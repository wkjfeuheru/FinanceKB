"""agentic-v2 清单与独立样本统计协议（全程离线）。"""

import json

import pytest

from finance_rag.src.eval.statistics import ExperimentSpec, compare_runs


def _case(case_id, query, **extra):
    return {"id": case_id, "query": query, "required_facts": ["事实一"],
            "tier": "silver", "review_status": "pending", **extra}


def _config():
    return dict(protocol_version="agentic-v2", dataset_fingerprint="d",
                kb_fingerprint="k", judge_fingerprint="j", protocol_fingerprint="p")


def _record(case_id, group, primary, guard=1.0):
    return {"case_id": case_id, "sample_group_id": group, "tier": "gold",
            "metrics": {"mrr": primary, "recall": guard}}


def test_manifest_deduplicates_exact_and_groups_variants_without_promoting_gold(tmp_path):
    from finance_rag.src.eval.manifest import build_manifest

    rows = [_case("a", "Ａ 问题"), _case("b", "a问题"),
            _case("v", "换一种问法", provenance={"paired_case_id": "a"}),
            _case("c", "完全不同的问题", tier="gold", review_status="approved", reviewer="张三")]
    (tmp_path / "gold100.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    result = build_manifest(tmp_path)
    assert result == build_manifest(tmp_path)
    assert result["raw_case_count"] == 4
    assert result["exact_unique_case_count"] == 3
    assert result["independent_sample_count"] == 2
    by_id = {row["case_id"]: row for row in result["cases"]}
    assert by_id["a"]["sample_group_id"] == by_id["v"]["sample_group_id"] == by_id["b"]["sample_group_id"]
    assert by_id["a"]["tier"] == "silver"
    assert by_id["c"]["reviewer"] == "张三"
    assert by_id["a"]["source_file"] == "gold100.jsonl"


def test_manifest_separates_anchors_regression_and_flags_near_duplicates(tmp_path):
    from finance_rag.src.eval.manifest import build_manifest

    rows = [_case("a", "请问公司今年研发投入是多少元"), _case("b", "请问公司今年研发投入是多少万元")]
    (tmp_path / "main.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "eval_anchors.jsonl").write_text(json.dumps({"anchor_id": "anchor", "user_input": "q"}), encoding="utf-8")
    (tmp_path / "research_report_regression4.jsonl").write_text(json.dumps(_case("r", "回归")), encoding="utf-8")
    result = build_manifest(tmp_path)
    assert result["independent_sample_count"] == 2
    assert result["anchor_count"] == result["regression_case_count"] == 1
    assert all(row["needs_review"] for row in result["cases"] if row["pool"] == "evaluation")


def test_group_bootstrap_averages_members_before_resampling():
    before = [_record("a", "g1", 0), _record("v", "g1", 0), _record("b", "g2", 0)]
    after = [_record("a", "g1", 1), _record("v", "g1", 1), _record("b", "g2", 0)]
    report = compare_runs(before, after, ExperimentSpec("x", "mrr", ()),
                          baseline_config=_config(), candidate_config=_config(), bootstrap_samples=100)
    metric = report["tiers"]["gold"]["metrics"]["mrr"]
    assert metric["n"] == 2
    assert metric["delta"] == 0.5
    assert metric["paired_case_count"] == 3


@pytest.mark.parametrize("key", ["dataset_fingerprint", "kb_fingerprint", "judge_fingerprint", "protocol_fingerprint", "protocol_version"])
def test_v2_fingerprint_mismatch_cannot_pass(key):
    before, after = _config(), _config()
    after[key] = "different"
    report = compare_runs([_record("a", "g", 0)], [_record("a", "g", 1)], ExperimentSpec("x", "mrr", ()),
                          baseline_config=before, candidate_config=after, bootstrap_samples=30)
    assert report["decision"] == "no_decision"
    assert key in report["comparability_reasons"]


def test_v2_missing_fingerprints_and_group_conflicts_cannot_claim_improvement():
    report = compare_runs([_record("a", "g", 0)], [_record("a", "other", 1)], ExperimentSpec("x", "mrr", ()),
                          baseline_config={"protocol_version": "agentic-v2"}, candidate_config={"protocol_version": "agentic-v2"}, bootstrap_samples=30)
    assert report["decision"] == "no_decision"
    assert "sample_group_conflict:a" in report["comparability_reasons"]
    assert report["can_claim_improvement"] is False


def test_v2_guardrail_negative_mean_blocks_pass_even_if_ci_crosses_zero():
    before = [_record(str(i), str(i), 0) for i in range(10)]
    after = [_record(str(i), str(i), 1, 0.99 if i == 0 else 1.0) for i in range(10)]
    report = compare_runs(before, after, ExperimentSpec("x", "mrr", ("recall",)),
                          baseline_config=_config(), candidate_config=_config(), bootstrap_samples=100)
    assert report["decision"] == "fail"
    assert report["guardrail_failures"] == ["recall"]


def test_runner_carries_protocol_judge_and_variant_grouping(tmp_path):
    from finance_rag.src.eval.dataset import EvaluationDataset
    from finance_rag.src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput

    dataset = EvaluationDataset.from_dicts([
        _case("a", "主问题", question_type="single_hop", reference_answer="事实一", evidence=[{"source": "d.pdf", "document_version": "v1", "chunk_id": "c", "quote": "事实一"}]),
        _case("v", "另一问法", question_type="single_hop", reference_answer="事实一", evidence=[{"source": "d.pdf", "document_version": "v1", "chunk_id": "c", "quote": "事实一"}], provenance={"paired_case_id": "a"}),
    ])
    strategy = EvaluationStrategy("offline", lambda query: RetrievalOutput([], []))
    run = EvaluationRunner().run(dataset, strategy, "retrieval", tmp_path,
                                 metadata={"protocol_version": "agentic-v2", "judge_fingerprint": "judge", "kb_fingerprint": "kb"})
    assert run.records[0]["sample_group_id"] == run.records[1]["sample_group_id"]
    assert run.records[0]["run_fingerprints"]["protocol_version"] == "agentic-v2"
    assert run.records[0]["run_fingerprints"]["judge_fingerprint"] == "judge"
    assert run.summary["protocol_fingerprint"]


def test_manifest_reads_legacy_markdown_without_network_or_source_writes(tmp_path):
    from finance_rag.src.eval.manifest import build_manifest

    path = tmp_path / "evaluation_qa_generated.md"
    content = "## 1. 问题是什么？\n\n- 问题类型：single_hop\n- 相关文档：doc.pdf\n- 证据chunk：abc\n\n### 标准答案\n\n事实一。\n"
    path.write_text(content, encoding="utf-8")
    result = build_manifest(tmp_path)
    assert path.read_text(encoding="utf-8") == content
    assert result["raw_case_count"] == 1
    assert result["cases"][0]["tier"] == "silver"
    assert result["cases"][0]["review_status"] == "pending"
    assert result["cases"][0]["evidence_chunk_ids"] == ["abc"]


@pytest.mark.parametrize("mutation", ["missing_protocol", "missing_group", "unavailable_judge", "record_conflict"])
def test_v2_partial_or_inconsistent_provenance_cannot_pass(mutation):
    before_config, after_config = _config(), _config()
    before, after = _record("a", "g", 0), _record("a", "g", 1)
    if mutation == "missing_protocol":
        after_config.pop("protocol_version")
    elif mutation == "missing_group":
        after.pop("sample_group_id")
    elif mutation == "unavailable_judge":
        before_config["judge_fingerprint"] = after_config["judge_fingerprint"] = "unavailable"
    else:
        before["run_fingerprints"] = {**before_config, "kb_fingerprint": "other"}
    report = compare_runs([before], [after], ExperimentSpec("x", "mrr", ()),
                          baseline_config=before_config, candidate_config=after_config, bootstrap_samples=30)
    assert report["decision"] == "no_decision"


def test_v2_tiny_guardrail_regression_is_not_hidden_by_rounding():
    before = [_record(str(i), str(i), 0) for i in range(10)]
    after = [_record(str(i), str(i), 1, 0.9999999 if i == 0 else 1) for i in range(10)]
    report = compare_runs(before, after,
                          ExperimentSpec("x", "mrr", ("recall",)), baseline_config=_config(),
                          candidate_config=_config(), bootstrap_samples=30)
    assert report["decision"] == "fail"
