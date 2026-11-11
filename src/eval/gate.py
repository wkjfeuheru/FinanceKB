"""回归门禁：把「这次改动能不能合」写成退出码。

输入是 A/B 报告（``report.json`` 或 ``comparison.json``），不重新计算指标——
门禁只做判定聚合，保证「报告上写的结论」与「CI 拦不拦」是同一份事实。

策略
----
``smoke``：只看主指标回退与可比性（每次改检索/重排/规划都跑）。
``release``：在 smoke 之上加延迟预算、锚点越界与证据等级要求。

退出码：0 通过；1 阻断；2 结论不足（inconclusive，告警不阻断）。
"""

from __future__ import annotations

from typing import Any

POLICIES: dict[str, dict[str, Any]] = {
    "smoke": {"latency_ratio": None, "require_anchor_check": False, "allow_silver": True},
    "release": {"latency_ratio": 1.25, "require_anchor_check": True, "allow_silver": False},
}

EXIT_PASS = 0
EXIT_BLOCK = 1
EXIT_WARN = 2


def _primary_metric(report: dict[str, Any]) -> dict[str, Any]:
    tier = str(report.get("decision_tier") or "gold")
    primary = str(report.get("primary_metric") or "")
    metrics = ((report.get("tiers") or {}).get(tier) or {}).get("metrics") or {}
    return metrics.get(primary) or {}


def _anchor_failed(report: dict[str, Any]) -> str | None:
    anchors = report.get("anchors") or {}
    if not anchors:
        return None
    status = str(anchors.get("status") or "")
    if anchors.get("passed") is False or status in {"failed", "out_of_band", "blocked"}:
        return "锚点样本越界"
    return None


def evaluate_gate(
    report: dict[str, Any],
    *,
    policy: str = "smoke",
    max_regression: float = 0.0,
    latency_ratio: float | None = None,
) -> dict[str, Any]:
    """按策略聚合阻断理由；返回 ``status`` / ``exit_code`` / ``reasons``。"""
    if policy not in POLICIES:
        raise ValueError(f"未知门禁策略：{policy}；可选 {', '.join(sorted(POLICIES))}")
    config = POLICIES[policy]
    ratio_budget = latency_ratio if latency_ratio is not None else config["latency_ratio"]

    reasons: list[str] = []
    decision = str(report.get("decision") or "no_decision")
    comparability = str(report.get("comparability_status") or "unavailable")
    if comparability != "verified":
        reasons.append(f"可比性未验证（{comparability}）")
    if decision in {"fail", "no_decision"}:
        reasons.append(f"判定为 {decision}")

    metric = _primary_metric(report)
    delta = metric.get("delta")
    if isinstance(delta, (int, float)) and delta < -abs(max_regression):
        reasons.append(
            f"主指标 {report.get('primary_metric')} 回退 {delta:.4f}（阈值 -{abs(max_regression):.4f}）"
        )
    if metric.get("verdict") == "regression":
        reasons.append(f"主指标 {report.get('primary_metric')} 判定为 regression")

    noisy = report.get("noise_floor") or {}
    if report.get("noise_suppressed"):
        reasons.append("提升未超过噪声下限")

    grade = str(report.get("evidence_grade") or report.get("decision_tier") or "gold")
    if policy == "release" and grade != "gold" and not config["allow_silver"]:
        reasons.append(f"发布门禁要求 Gold 证据，当前为 {grade}")

    if config["require_anchor_check"]:
        failed = _anchor_failed(report)
        if failed:
            reasons.append(failed)

    if ratio_budget:
        latency = report.get("latency") or {}
        experiment = report.get("experiment") or {}
        arm_a = (experiment.get("arm_a") or {}).get("name")
        arm_b = (experiment.get("arm_b") or {}).get("name")
        for stage in ("retrieval", "generation"):
            before = ((latency.get(arm_a) or {}).get(stage) or {}).get("p95_ms")
            after = ((latency.get(arm_b) or {}).get(stage) or {}).get("p95_ms")
            if not before or not after:
                continue
            if after / before > ratio_budget:
                reasons.append(
                    f"{stage} p95 延迟 {after:.0f}ms 超过基线 {before:.0f}ms 的 "
                    f"{ratio_budget:.2f}× 预算"
                )

    if reasons:
        status, code = "block", EXIT_BLOCK
    elif decision == "inconclusive" or noisy:
        status, code = "warn", EXIT_WARN
    else:
        status, code = "pass", EXIT_PASS
    return {
        "status": status,
        "exit_code": code,
        "policy": policy,
        "decision": decision,
        "evidence_grade": grade,
        "comparability": comparability,
        "reasons": reasons,
    }


def render_gate(result: dict[str, Any]) -> str:
    lines = [
        f"门禁结果：{result['status']}（策略 {result['policy']}，判定 {result['decision']}，"
        f"证据 {result['evidence_grade']}）"
    ]
    lines.extend(f"  - {reason}" for reason in result.get("reasons") or [])
    if not result.get("reasons"):
        lines.append("  - 无阻断项")
    return "\n".join(lines)
