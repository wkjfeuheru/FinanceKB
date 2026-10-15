"""离线验证证据覆盖、候选阶段与运行可比性。"""

import csv
import json

import pytest

from finance_rag.src.eval import metrics as metrics_module
from finance_rag.src.eval.cli import main
from finance_rag.src.eval.dataset import EvaluationDataset
from finance_rag.src.eval.retrieval import build_retrieval_strategy
from finance_rag.src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput


def _case(case_id="explicit", stratum="explicit"):
    return {
        "id": case_id, "query": f"问题{case_id}", "question_type": "single_hop",
        "reference_answer": "收入100亿元", "required_facts": ["收入100亿元", "增长20%"],
        "evidence": [
            {"chunk_id": "gold-a", "source": "a.pdf", "document_version": "v1", "quote": "收入：100 亿元。"},
            {"chunk_id": "gold-b", "source": "b.pdf", "document_version": "v1", "quote": "增长20%"},
        ],
        "tier": "gold", "review_status": "approved", "provenance": {"stratum": stratum},
    }


class DiagnosticKB:
    embedding_model = "fixture/model"

    def hybrid_search(self, _query, **kwargs):
        self.last_retrieval_diagnostics = {
            "fusion_mode": "dense" if kwargs["use_dense_only"] else "weighted",
            "cliff_enabled": kwargs["cliff_detection"],
            "coarse_candidate_count": 50, "coarse_source_count": 2,
            "coarse_duplicate_source_slots": 48,
            "coarse_candidate_chunk_ids": ["gold-a"] + [f"x{i}" for i in range(20)] + ["gold-b"] + [f"y{i}" for i in range(28)],
            "rerank_candidate_count": 2, "rerank_source_count": 2,
            "rerank_duplicate_source_slots": 0,
            "rerank_candidate_chunk_ids": ["gold-b", "wrong"],
        }
        return [{"id": "wrong", "source": "a.pdf", "content": "收入100亿元！"}]


def test_text_coverage_normalizes_punctuation_without_cross_chunk_false_hit():
    compute = getattr(metrics_module, "compute_text_coverage", None)
    assert callable(compute), "缺少确定性文本覆盖指标"
    assert compute(["收入：100 亿元！", "增长20％"], ["收入100亿元。", "增长 20%", "不存在"]) == 0.6667
    assert compute(["收入", "100亿元"], ["收入100亿元"]) == 0.0
    assert compute(["内容"], ["", "  ", "。！"]) is None


def test_document_recall_deduplicates_sources():
    compute = getattr(metrics_module, "compute_document_recall", None)
    assert callable(compute), "缺少文档召回指标"
    assert compute(["a.pdf", "a.pdf", "wrong.pdf"], ["a.pdf", "b.pdf"], k=5) == 0.5
    assert compute([], [], k=5) is None


def test_coverage_preserves_numeric_decimal_sign_and_unit():
    compute = getattr(metrics_module, "compute_text_coverage", None)
    assert callable(compute)
    assert compute(["利润15亿元，增长20，增速5%"], ["利润1.5亿元", "增长20%", "增速-5%"] ) == 0.0


def test_direct_comparator_uses_record_fingerprints():
    from finance_rag.src.eval.statistics import EXPERIMENT_SPECS, compare_runs

    baseline = {"case_id": "g", "tier": "gold", "metrics": {"recall_at_5": 0, "precision_at_5": 0, "mrr": 0},
                "run_fingerprints": {"dataset_fingerprint": "dataset-a", "kb_fingerprint": "kb-a"}}
    candidate = {**baseline, "metrics": {"recall_at_5": 1, "precision_at_5": 1, "mrr": 1},
                 "run_fingerprints": {"dataset_fingerprint": "dataset-a", "kb_fingerprint": "kb-b"}}
    report = compare_runs([baseline], [candidate], EXPERIMENT_SPECS["hybrid_vs_dense"], bootstrap_samples=20)
    assert report["comparable"] is False
    assert report["decision"] == "no_decision"


def test_runner_uses_coarse_pool_and_exposes_stage_survival(tmp_path):
    run = EvaluationRunner().run(
        EvaluationDataset.from_dicts([_case()]), build_retrieval_strategy(DiagnosticKB(), "L5"),
        "retrieval", tmp_path, metadata={"kb_fingerprint": "kb-fixture"},
    )
    row = run.records[0]
    assert row["status"] == "completed"
    assert row["metrics"]["document_recall_at_5"] == 0.5
    assert row["metrics"]["exact_candidate_recall_at_20"] == 0.5
    assert row["metrics"]["exact_candidate_recall_at_50"] == 1.0
    assert row["metrics"]["quote_coverage_at_5"] == 0.5
    assert row["metrics"]["required_fact_coverage_at_5"] == 0.5
    diag = row["diagnostics"]
    assert diag["candidate_stage"] == "fusion"
    assert diag["resolved_fusion_mode"] == "weighted"
    assert diag["coarse_candidate_count"] == 50
    assert diag["rerank_candidate_chunk_ids"] == ["gold-b", "wrong"]
    assert diag["final_candidate_count"] == 1
    assert diag["final_source_count"] == 1
    assert diag["final_duplicate_source_slots"] == 0
    assert diag["evidence_survival"] == {
        "fusion": ["gold-a", "gold-b"], "rerank": ["gold-b"], "final_top5": [],
        "sparse": "unavailable",
    }


def test_dense_stage_is_not_labelled_fusion(tmp_path):
    run = EvaluationRunner().run(
        EvaluationDataset.from_dicts([_case()]), build_retrieval_strategy(DiagnosticKB(), "L0"),
        "retrieval", tmp_path,
    )
    diag = run.records[0]["diagnostics"]
    assert diag["candidate_stage"] == "dense"
    assert "fusion" not in diag["evidence_survival"]


def test_production_multiple_filter_attempts_have_no_invented_global_coarse_rank(monkeypatch):
    from finance_rag.src.services import retrieval_service

    async def search(_request, _queries, _keywords, _filter_plan):
        return {
            "docs": [{"id": "final", "source": "a.pdf", "content": "证据"}],
            "filters": {"broker": ["东吴证券"]}, "rewritten": "原问题", "k": 50,
            "rerank_top_n": 5,
            "diagnostics": {"filter_attempts": [
                {"reason": "tentative_priority", "filters": {"broker": ["东吴证券"], "report_type": ["个股"]}, "candidate_count": 1, "error": None},
                {"reason": "hard_only", "filters": {"broker": ["东吴证券"]}, "candidate_count": 1, "error": None},
            ], "retrieval_calls": [
                {"reason": "tentative_priority", "coarse_candidate_chunk_ids": ["first"], "filters": {"broker": ["东吴证券"], "report_type": ["个股"]}},
                {"reason": "hard_only", "coarse_candidate_chunk_ids": ["second"], "filters": {"broker": ["东吴证券"]}},
            ]},
        }

    # 让真实共享执行器聚合分支诊断，替换存储/上下文依赖以确保离线。
    policy = retrieval_service.RetrievalPolicy(route=False, grade=False)
    dependencies = retrieval_service.RetrievalDependencies(
        search=search,
        prepare_context=lambda docs: {"context_docs": docs, "context_text": "[1]证据", "sources": []},
    )
    execute = retrieval_service.execute_retrieval
    kb = object()

    async def retrieve(request, selected_policy, _dependencies=None, **kwargs):
        assert request.query == "原问题"
        assert selected_policy is policy
        assert kwargs["kb"] is kb
        return await execute(request, selected_policy, dependencies, kb=kb)

    monkeypatch.setattr(retrieval_service, "resolve_policy", lambda **kwargs: policy)
    monkeypatch.setattr(retrieval_service, "execute_retrieval", retrieve)
    output = build_retrieval_strategy(kb, "PRODUCTION").retrieve("原问题")
    assert output.candidates == []
    assert output.diagnostics["unified_coarse_stage"] == "unavailable"
    assert "coarse_candidate_chunk_ids" not in output.diagnostics
    assert output.diagnostics["filter_attempts"][1]["filters"] == {"broker": ["东吴证券"]}


def test_outputs_include_stratum_coverage_and_reproducibility(tmp_path):
    run = EvaluationRunner().run(
        EvaluationDataset.from_dicts([_case(), _case("cross", "cross_doc")]),
        build_retrieval_strategy(DiagnosticKB(), "L5"), "retrieval", tmp_path,
        metadata={"kb_fingerprint": "kb-fixture"},
    )
    for bucket in [run.summary["independent_positive"], *run.summary["strata"].values()]:
        assert bucket["metric_coverage"]["quote_coverage_at_5"] == 1.0
    config = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    for output in [config, run.summary]:
        assert len(output["dataset_fingerprint"]) == 64
        assert output["kb_fingerprint"] == "kb-fixture"
        assert output["embedding_model"] == "fixture/model"
        assert output["embedding_version"]
        assert output["embedding_dim"] > 0
        assert output["embedding_pooling"] in {"cls", "mean"}
        assert output["chunk_max_tokens"] > 0
        assert output["chunk_overlap_tokens"] >= 0
        assert output["fusion_mode"] == "weighted"
        assert output["dense_weight"] > 0
        assert output["sparse_weight"] > 0
        assert len(output["git_head"]) == 40
        assert isinstance(output["git_dirty"], bool)
    with (tmp_path / "per_query.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert row["document_recall_at_5"] == "0.5"
    assert row["coarse_candidate_count"] == "50"
    assert row["candidate_stage"] == "fusion"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "exact_candidate_recall_at_50" in report
    assert "unavailable" in report
    assert "kb-fixture" in report


def test_summary_explains_modes_cliff_and_missing_stage_observations(tmp_path):
    run = EvaluationRunner().run(
        EvaluationDataset.from_dicts([_case()]), build_retrieval_strategy(DiagnosticKB(), "L5"),
        "retrieval", tmp_path,
    )
    for bucket in [run.summary, run.summary["independent_positive"], run.summary["strata"]["explicit"]]:
        diag = bucket["retrieval_diagnostics"]
        assert diag.get("resolved_fusion_modes") == {"weighted": 1}
        assert diag.get("cliff_enabled_count") == 1
        assert diag.get("coverage", {}).get("coarse_candidate_count") == 1.0
        assert diag.get("survival_recall", {}).get("fusion") == 1.0
        assert diag.get("survival_recall", {}).get("rerank") == 0.5
        assert diag.get("survival_recall", {}).get("final_top5") == 0.0


def test_run_config_preserves_zero_weight(tmp_path):
    strategy = build_retrieval_strategy(DiagnosticKB(), "L1", dense_weight=1.0, sparse_weight=0.0)
    run = EvaluationRunner().run(EvaluationDataset.from_dicts([_case()]), strategy, "retrieval", tmp_path)
    assert run.summary["sparse_weight"] == 0.0


def test_csv_preserves_record_identity_status_latency_and_error(tmp_path):
    def retrieve(query):
        if query == "问题failed":
            raise RuntimeError("fixture retrieval failed")
        return RetrievalOutput([], [{"id": "gold-a", "source": "a.pdf", "content": "收入100亿元"}])

    run = EvaluationRunner().run(
        EvaluationDataset.from_dicts([_case("completed"), _case("failed")]),
        EvaluationStrategy(name="csv", retrieve=retrieve), "retrieval", tmp_path,
    )
    with (tmp_path / "per_query.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row, record in zip(rows, run.records, strict=True):
        for key in ("case_id", "tier", "question_type", "stratum", "status", "query", "latency_ms", "error"):
            assert row[key] == ("" if record[key] is None else str(record[key]))
    assert rows[0]["recall_at_5"] == "0.5"
    assert rows[1]["recall_at_5"] == ""


@pytest.mark.parametrize("content,target", [
    ("营业收入1153.62亿元", "153.62亿元"),
    ("同比-15%", "5%"),
    ("同比-15%", "15%"),
    ("同比+5%", "5%"),
    ("营业收入9.76亿元", "76亿元"),
    ("营业收入15.3亿元", "15"),
    ("同比15%", "15"),
    ("业务AIDC增长", "IDC"),
])
def test_coverage_rejects_partial_numeric_or_latin_tokens(content, target):
    assert metrics_module.compute_text_coverage([content], [target]) == 0.0


@pytest.mark.parametrize("content,target", [
    ("营业收入：１５３．６２ 亿元。", "153.62亿元"),
    ("同比（-15％）。", "-15%"),
    ("营业收入9.76 亿，利润稳健增长。", "9.76亿"),
    ("公司利润保持稳健增长", "稳健增长"),
    ("业务 IDC 增长", "IDC"),
    ("The IDC industry", "IDC"),
    ("The  IDC  industry", "IDC"),
    ("The\t IDC industry", "IDC"),
    ("The,IDC,industry", "IDC"),
    ("前次1153.62亿元，本次153.62亿元", "153.62亿元"),
])
def test_coverage_keeps_normalized_financial_values_and_chinese_phrases(content, target):
    assert metrics_module.compute_text_coverage([content], [target]) == 1.0


@pytest.mark.parametrize("field", ["dataset_fingerprint", "kb_fingerprint"])
def test_comparison_refuses_mismatching_fingerprint(tmp_path, field):
    paths = []
    for name in ["baseline", "candidate"]:
        folder = tmp_path / name
        folder.mkdir()
        config = {"dataset_fingerprint": "dataset-a", "kb_fingerprint": "kb-a"}
        if name == "candidate":
            config[field] = "different"
        (folder / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        record = {"case_id": "same", "tier": "gold", "metrics": {"recall_at_5": 1, "precision_at_5": 1, "mrr": 1}}
        (folder / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        paths.append(str(folder))
    assert main(["ab", "run", "--baseline", paths[0], "--candidate", paths[1],
                 "--experiment", "hybrid_vs_dense", "--out", str(tmp_path / "comparison"),
                 "--bootstrap", "20"]) == 0
    report = json.loads((tmp_path / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    assert report.get("comparable") is False
    assert report["decision"] == "no_decision"
    assert field in report["comparability_reasons"]
