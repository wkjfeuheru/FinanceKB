"""回归门禁：三态退出码、证据等级与延迟预算。"""

from __future__ import annotations

from src.eval.gate import EXIT_BLOCK, EXIT_PASS, EXIT_WARN, evaluate_gate


def _report(**overrides) -> dict:
    report = {
        "experiment": {"arm_a": {"name": "a"}, "arm_b": {"name": "b"}},
        "primary_metric": "mrr",
        "decision": "pass",
        "decision_tier": "gold",
        "evidence_grade": "gold",
        "comparability_status": "verified",
        "tiers": {"gold": {"metrics": {"mrr": {"delta": 0.05, "verdict": "improvement"}}}},
        "latency": {
            "a": {"retrieval": {"p95_ms": 100.0}},
            "b": {"retrieval": {"p95_ms": 110.0}},
        },
    }
    report.update(overrides)
    return report


def test_pass_report_exits_zero():
    result = evaluate_gate(_report(), policy="smoke")
    assert result["status"] == "pass"
    assert result["exit_code"] == EXIT_PASS


def test_primary_regression_blocks():
    report = _report(
        decision="fail",
        tiers={"gold": {"metrics": {"mrr": {"delta": -0.08, "verdict": "regression"}}}},
    )
    result = evaluate_gate(report, policy="smoke")
    assert result["status"] == "block"
    assert result["exit_code"] == EXIT_BLOCK
    assert any("regression" in reason for reason in result["reasons"])


def test_inconclusive_is_warning_not_block():
    result = evaluate_gate(_report(decision="inconclusive"), policy="smoke")
    assert result["status"] == "warn"
    assert result["exit_code"] == EXIT_WARN


def test_release_requires_gold_evidence():
    report = _report(decision="pass", decision_tier="silver", evidence_grade="silver")
    result = evaluate_gate(report, policy="release")
    assert result["status"] == "block"
    assert any("Gold" in reason for reason in result["reasons"])


def test_latency_budget_breach_blocks():
    report = _report(
        latency={
            "a": {"retrieval": {"p95_ms": 100.0}},
            "b": {"retrieval": {"p95_ms": 180.0}},
        }
    )
    result = evaluate_gate(report, policy="release", latency_ratio=1.25)
    assert result["status"] == "block"
    assert any("p95" in reason for reason in result["reasons"])


def test_unverified_comparability_blocks():
    result = evaluate_gate(_report(comparability_status="mismatch"), policy="smoke")
    assert result["status"] == "block"
    assert any("可比性" in reason for reason in result["reasons"])
