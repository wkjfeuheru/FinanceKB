"""30 题分层集、父块还原、Agentic 开关和报告聚合。"""

from __future__ import annotations

from src.eval.ab_catalog import select_experiments
from src.eval.agentic_retrieval import retrieve_agentic
from src.eval.dataset import EvaluationDataset
from src.eval.interview_dataset import (
    INTERVIEW30_PATH,
    VAGUE_PAIRS,
    build_interview_cases,
)
from src.eval.interview_report import render_interview_report
from src.eval.metrics import assess_parent_restore
from src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput


def test_interview30_has_four_strata_and_shared_vague_evidence():
    cases = build_interview_cases()
    EvaluationDataset.from_dicts(cases).validate()
    counts: dict[str, int] = {}
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        stratum = case["provenance"]["stratum"]
        counts[stratum] = counts.get(stratum, 0) + 1
    assert counts == {"explicit": 8, "cross_doc": 10, "negative": 2, "vague": 10}
    assert len(by_id) == 30

    for spec in VAGUE_PAIRS:
        vague = by_id[spec["id"]]
        paired = by_id[spec["paired_case_id"]]
        assert vague["provenance"]["paired_case_id"] == paired["id"]
        assert vague["evidence"] == paired["evidence"]
        assert vague["reference_answer"] == paired["reference_answer"]
        assert vague["query"] != paired["query"]
        assert vague['provenance']['query_style'] == 'standalone_paired'
        assert vague['provenance']['independent_sample'] is False
        assert vague['query'].endswith(paired['query'])

    stored = EvaluationDataset.load(INTERVIEW30_PATH)
    assert [case.provenance["stratum"] for case in stored.cases].count("vague") == 10


def test_parent_restore_respects_production_size_window():
    quote = "营收 153.62亿"
    assert assess_parent_restore(quote, None) == "no_parent"
    assert assess_parent_restore(quote, "太短") == "not_replaced"
    assert assess_parent_restore(quote, "x" * 3001) == "not_replaced"
    assert assess_parent_restore(quote, ("前言。" * 20) + quote) == "restored"
    assert assess_parent_restore(quote, "这段父块足够长，但没有目标引文。" * 4) == "missing"


def test_runner_reports_independent_recall_and_parent_restore(tmp_path):
    explicit = _case("explicit-1", "explicit", "e1", "证据原文")
    vague = _case("vague-1", "vague", "e1", "证据原文", paired="explicit-1", query="模糊问法？")
    dataset = EvaluationDataset.from_dicts([explicit, vague])

    def retrieve(query: str) -> RetrievalOutput:
        hit = query == "显式问题？"
        return RetrievalOutput(
            candidates=[],
            results=[{
                "id": "e1" if hit else "other",
                "parent_id": "p1",
                "content": "子块",
            }],
        )

    strategy = EvaluationStrategy(
        name="L5",
        retrieve=retrieve,
        resolve_parent=lambda _parent_id: "填充。" * 20 + "证据原文",
    )
    result = EvaluationRunner().run(dataset, strategy, "retrieval", tmp_path)

    assert result.summary["independent_positive"]["n"] == 1
    assert result.summary["independent_positive"]["metrics"]["recall_at_5"] == 1.0
    assert result.summary["strata"]["vague"]["metrics"]["recall_at_5"] == 0.0
    assert result.summary["vague_pairs"]["mean_delta"]["recall_at_5"] == -1.0
    assert result.summary["independent_positive"]["parent_restore"]["rate"] == 1.0
    assert "不含 vague" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_agentic_switches_do_not_call_each_other():
    calls = {"rewrite": 0, "route": 0, "reflect": 0, "search": []}

    def rewrite(_query: str) -> dict:
        calls["rewrite"] += 1
        return {"rewritten": "改写", "keywords": ["词"], "sub_queries": ["子查询"]}

    def route(_query: str) -> dict:
        calls["route"] += 1
        return {"category": ["macro"]}

    def reflect(state: dict) -> dict:
        calls["reflect"] += 1
        return {**state, "reflection_round": 99}

    def search(queries, *, filters, keywords):
        calls["search"].append({
            "queries": list(queries),
            "filters": filters,
            "keywords": list(keywords),
        })
        return [{"id": "c1", "content": "证据"}]

    retrieve_agentic(
        "原问题",
        rewrite=False,
        route=False,
        reflect=False,
        search=search,
        rewrite_query_fn=rewrite,
        infer_filters_fn=route,
        reflect_fn=reflect,
        normalize_fn=lambda value: value or {},
    )
    assert calls == {"rewrite": 0, "route": 0, "reflect": 0, "search": [
        {"queries": ["原问题"], "filters": None, "keywords": []},
    ]}

    retrieve_agentic(
        "原问题",
        rewrite=True,
        route=True,
        reflect=True,
        search=search,
        rewrite_query_fn=rewrite,
        infer_filters_fn=route,
        reflect_fn=reflect,
        normalize_fn=lambda value: value or {},
        max_reflect_rounds=2,
    )
    assert calls["rewrite"] == 1
    assert calls["route"] == 1
    assert calls["reflect"] == 1
    assert calls["search"][-1]["queries"] == ["原问题", "改写", "子查询"]
    assert calls["search"][-1]["filters"] == {"category": ["macro"]}
    assert calls["search"][-1]["keywords"] == ["词"]


def test_component_experiments_toggle_one_switch():
    route = select_experiments(["route_on_off"])[0]
    rewrite = select_experiments(["rewrite_on_off"])[0]
    reflect = select_experiments(["reflect_on_off"])[0]
    agentic = select_experiments(["agentic_on_off"])[0]

    assert route.arm_a.strategy_overrides["route"] is False
    assert route.arm_b.strategy_overrides == {
        "rewrite": False, "route": True, "reflect": False, "retrieval_only": True,
    }
    assert rewrite.arm_b.strategy_overrides["rewrite"] is True
    assert rewrite.arm_b.strategy_overrides["route"] is False
    assert rewrite.arm_b.strategy_overrides["reflect"] is False
    assert reflect.arm_b.strategy_overrides["reflect"] is True
    assert reflect.arm_b.strategy_overrides["rewrite"] is False
    assert agentic.arm_a.strategy_overrides == {
        "rewrite": False, "route": False, "reflect": False,
    }
    assert agentic.arm_b.strategy_overrides == {
        "rewrite": True, "route": True, "reflect": True,
    }
    assert "retrieval_only" not in agentic.arm_b.strategy_overrides


def test_interview_report_uses_measured_numbers_only():
    text = render_interview_report({
        "run_id": "demo",
        "dataset": "research_report_interview30.jsonl",
        "layers": [{
            "strategy": "L5",
            "independent_positive": {
                "n": 18,
                "metrics": {"recall_at_5": 0.25, "mrr": 0.2},
                "parent_restore": {"rate": 0.5, "not_replaced": 1},
            },
            "strata": {"explicit": {"n": 8, "metrics": {"recall_at_5": 0.3, "mrr": 0.25}}},
            "vague_pairs": {"mean_delta": {"recall_at_5": -0.1}},
        }],
        "agentic": {"comparison": {"metrics": {
            "context_precision": {"arm_a": 0.4, "arm_b": 0.55, "delta": 0.15},
        }}},
        "ablations": [{"name": "route_on_off", "comparison": {"metrics": {
            "recall_at_5": {"arm_a": 0.2, "arm_b": 0.3, "delta": 0.1},
        }}}],
        "commands": ["python scripts/run_interview30.py"],
    })
    assert "0.2500" in text
    assert "context_precision" in text
    assert "Recall@5=90%" not in text
    assert "父块还原" in text


def _case(case_id: str, stratum: str, chunk_id: str, quote: str, paired: str = "", query: str = "显式问题？"):
    return {
        "id": case_id,
        "query": query,
        "question_type": "single_hop",
        "difficulty": "medium",
        "answer_type": "factual",
        "reference_answer": "答案",
        "required_facts": ["事实"],
        "evidence": [{
            "source": "a.pdf",
            "document_version": "a.pdf",
            "chunk_id": chunk_id,
            "quote": quote,
        }],
        "expected_behavior": "",
        "tier": "gold",
        "review_status": "approved",
        "reviewer": "test",
        "review_note": "",
        "provenance": {"stratum": stratum, "paired_case_id": paired},
    }
