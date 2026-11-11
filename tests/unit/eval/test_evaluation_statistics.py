"""配对统计、置信区间与护栏判定。"""

from src.eval.statistics import ExperimentSpec, compare_runs


def _record(case_id, tier, **metrics):
    return {"case_id": case_id, "tier": tier, "metrics": metrics}


def test_compare_runs_reports_paired_bootstrap_improvement_deterministically():
    baseline = [_record(f"g{i}", "gold", mrr=0.2) for i in range(20)]
    candidate = [_record(f"g{i}", "gold", mrr=0.5) for i in range(20)]
    spec = ExperimentSpec("rerank", primary_metric="mrr", guardrails=())

    first = compare_runs(baseline, candidate, spec, bootstrap_samples=1000, seed=7)
    second = compare_runs(baseline, candidate, spec, bootstrap_samples=1000, seed=7)

    assert first == second
    assert first["tiers"]["gold"]["metrics"]["mrr"]["delta"] == 0.3
    assert first["tiers"]["gold"]["metrics"]["mrr"]["verdict"] == "improvement"
    assert first["decision"] == "pass"


def test_guardrail_regression_blocks_primary_improvement():
    baseline = [
        _record(f"g{i}", "gold", mrr=0.2, recall_at_5=0.9)
        for i in range(20)
    ]
    candidate = [
        _record(f"g{i}", "gold", mrr=0.5, recall_at_5=0.5)
        for i in range(20)
    ]
    spec = ExperimentSpec("rerank", "mrr", ("recall_at_5",))

    report = compare_runs(baseline, candidate, spec, bootstrap_samples=500, seed=7)

    assert report["decision"] == "fail"
    assert report["guardrail_failures"] == ["recall_at_5"]


def test_gold_metric_coverage_below_95_percent_has_no_decision():
    baseline = [_record(f"g{i}", "gold", mrr=0.2) for i in range(20)]
    candidate = [
        _record(f"g{i}", "gold", mrr=0.5 if i < 18 else None)
        for i in range(20)
    ]

    report = compare_runs(
        baseline,
        candidate,
        ExperimentSpec("rerank", "mrr", ()),
        bootstrap_samples=200,
        seed=7,
    )

    assert report["tiers"]["gold"]["metrics"]["mrr"]["coverage"] == 0.9
    assert report["decision"] == "no_decision"


def test_gold_guardrail_coverage_below_95_percent_has_no_decision():
    baseline = [
        _record(f"g{i}", "gold", recall_at_5=0.2, precision_at_5=0.2, mrr=0.2)
        for i in range(20)
    ]
    candidate = [
        _record(
            f"g{i}", "gold", recall_at_5=0.5,
            precision_at_5=0.2 if i < 18 else None, mrr=0.5,
        )
        for i in range(20)
    ]
    spec = ExperimentSpec("hybrid", "recall_at_5", ("precision_at_5", "mrr"))

    report = compare_runs(baseline, candidate, spec, bootstrap_samples=100, seed=7)

    assert report["decision"] == "no_decision"


def test_gold_and_silver_are_never_merged():
    baseline = [
        _record("gold-1", "gold", mrr=0.2),
        _record("silver-1", "silver", mrr=0.9),
    ]
    candidate = [
        _record("gold-1", "gold", mrr=0.4),
        _record("silver-1", "silver", mrr=0.1),
    ]

    report = compare_runs(
        baseline,
        candidate,
        ExperimentSpec("rerank", "mrr", ()),
        bootstrap_samples=100,
        seed=7,
    )

    assert report["tiers"]["gold"]["metrics"]["mrr"]["delta"] == 0.2
    assert report["tiers"]["silver"]["metrics"]["mrr"]["delta"] == -0.8


# ---------------------------------------------------------------------------
# 生成侧指标的 Bootstrap 置信区间
# ---------------------------------------------------------------------------


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_mean():
    from src.eval.statistics import bootstrap_mean_ci

    values = [0.1, 0.4, 0.5, 0.6, 0.9]
    first = bootstrap_mean_ci(values, samples=500, seed=7)
    second = bootstrap_mean_ci(values, samples=500, seed=7)
    assert first == second
    assert first is not None
    low, high = first
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_mean_ci_returns_none_for_degenerate_input():
    from src.eval.statistics import bootstrap_mean_ci

    assert bootstrap_mean_ci([0.5]) is None
    assert bootstrap_mean_ci([None, None]) is None
    assert bootstrap_mean_ci([0.1, 0.2], samples=0) is None


def test_generation_spec_keeps_retrieval_guardrails():
    """生成侧 A/B 的护栏必须包含 recall@5 / mrr（保留检索侧评测的要求）。"""
    from src.eval.statistics import EXPERIMENT_SPECS

    spec = EXPERIMENT_SPECS["generation_ab"]
    assert spec.primary_metric == "faithfulness"
    assert {"recall_at_5", "mrr"} <= set(spec.guardrails)
    # 已有三个检索实验的主指标/护栏不得被改动
    assert EXPERIMENT_SPECS["hybrid_vs_dense"].guardrails == ("precision_at_5", "mrr")
    assert EXPERIMENT_SPECS["rerank_on_off"].guardrails == ("recall_at_5", "precision_at_5")
    assert EXPERIMENT_SPECS["cliff_on_off"].guardrails == ("recall_at_5", "mrr")

