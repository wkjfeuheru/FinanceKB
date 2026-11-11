"""判定口径：Gold/Silver 证据等级、population 过滤与噪声下限。"""

from __future__ import annotations

from src.eval.statistics import ExperimentSpec, compare_runs

_FINGERPRINTS = {
    "dataset_fingerprint": "d",
    "kb_fingerprint": "k",
    "protocol_version": "agentic-v2",
    "protocol_fingerprint": "p",
    "judge_fingerprint": "j",
}


def _rec(case_id: str, tier: str, stratum: str = "explicit", **metrics) -> dict:
    return {
        "case_id": case_id,
        "tier": tier,
        "stratum": stratum,
        "metrics": metrics,
        "sample_group_id": case_id,
    }


def test_silver_only_run_still_decides_but_cannot_claim_improvement():
    baseline = [_rec(f"s{i}", "silver", mrr=0.2) for i in range(20)]
    candidate = [_rec(f"s{i}", "silver", mrr=0.5) for i in range(20)]

    report = compare_runs(
        baseline, candidate, ExperimentSpec("x", "mrr", ()),
        bootstrap_samples=200, seed=7,
    )

    assert report["decision"] == "pass"
    assert report["decision_tier"] == "silver"
    assert report["evidence_grade"] == "silver"
    assert report["can_claim_improvement"] is False


def test_gold_pass_can_claim_improvement():
    baseline = [_rec(f"g{i}", "gold", mrr=0.2) for i in range(20)]
    candidate = [_rec(f"g{i}", "gold", mrr=0.5) for i in range(20)]

    report = compare_runs(
        baseline, candidate, ExperimentSpec("x", "mrr", ()),
        bootstrap_samples=200, seed=7,
        baseline_config=dict(_FINGERPRINTS), candidate_config=dict(_FINGERPRINTS),
    )

    assert report["decision_tier"] == "gold"
    assert report["comparable"] is True
    assert report["can_claim_improvement"] is True


def test_population_restricts_decision_to_one_stratum():
    baseline = [
        _rec("e1", "silver", "explicit", mrr=0.1),
        _rec("c1", "silver", "cross_doc", mrr=0.2),
    ]
    candidate = [
        _rec("e1", "silver", "explicit", mrr=0.9),
        _rec("c1", "silver", "cross_doc", mrr=0.25),
    ]

    report = compare_runs(
        baseline, candidate,
        ExperimentSpec("x", "mrr", (), population="cross_doc"),
        bootstrap_samples=200, seed=7,
    )

    section = report["tiers"]["silver"]
    assert section["case_count"] == 1
    assert section["metrics"]["mrr"]["delta"] == 0.05
    assert report["population"] == "cross_doc"


def test_noise_floor_downgrades_marginal_pass():
    baseline = [_rec(f"s{i}", "silver", mrr=0.2) for i in range(20)]
    candidate = [_rec(f"s{i}", "silver", mrr=0.22) for i in range(20)]

    report = compare_runs(
        baseline, candidate,
        ExperimentSpec("x", "mrr", (), noise_floor={"mrr": 0.05}),
        bootstrap_samples=200, seed=7,
    )

    assert report["noise_suppressed"] is True
    assert report["decision"] == "inconclusive"
