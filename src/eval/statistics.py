"""配对 Bootstrap、指标覆盖率与实验护栏判定。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    primary_metric: str
    guardrails: tuple[str, ...]
    #: 判定用哪一层样本：``auto`` 有 Gold 用 Gold、否则用 Silver（并标证据等级）；
    #: 也可显式写 ``gold`` / ``silver``。silver 结论不允许对外声称提升。
    decision_tier: str = "auto"
    #: 只在该分层上判定（例如 ``cross_doc``）；None 表示全量。
    population: str | None = None
    #: 逐指标噪声下限（同配置重跑的平均绝对偏移），低于它的提升按 inconclusive 处理。
    noise_floor: dict[str, float] = field(default_factory=dict)


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "hybrid_vs_dense": ExperimentSpec(
        "hybrid_vs_dense", "recall_at_5", ("precision_at_5", "mrr")
    ),
    "rerank_on_off": ExperimentSpec(
        "rerank_on_off", "mrr", ("recall_at_5", "precision_at_5")
    ),
    "cliff_on_off": ExperimentSpec(
        "cliff_on_off", "precision_at_5", ("recall_at_5", "mrr")
    ),
    # 生成侧 A/B：主指标是「防幻觉」的 faithfulness；护栏里同时保留检索侧客观
    # 指标 recall@5 / mrr —— 生成质量不能靠牺牲检索能力换。
    "generation_ab": ExperimentSpec(
        "generation_ab", "faithfulness",
        ("answer_relevancy", "context_recall", "recall_at_5", "mrr"),
    ),
    # 只比检索的臂（retrieval_only）：不生成答案时 faithfulness 等恒为 n/a。
    "retrieval_only_ab": ExperimentSpec(
        "retrieval_only_ab", "recall_at_5", ("precision_at_5", "chunk_mrr")
    ),
    # 四个优化点的组件实验：都只比检索，主指标是块级 Recall@5，护栏看噪声与排序。
    "agentic_on_off": ExperimentSpec(
        "agentic_on_off", "recall_at_5", ("precision_at_5", "chunk_mrr")
    ),
    # 查询分解只在跨文档题上有预期收益，预注册限定 population，避免被单跳题稀释。
    "rewrite_on_off": ExperimentSpec(
        "rewrite_on_off", "recall_at_5", ("precision_at_5", "chunk_mrr"),
        population="cross_doc",
    ),
    "route_on_off": ExperimentSpec(
        "route_on_off", "recall_at_5", ("precision_at_5", "chunk_mrr")
    ),
    "reflect_on_off": ExperimentSpec(
        "reflect_on_off", "recall_at_5", ("precision_at_5", "chunk_mrr")
    ),
}


def select_experiment_spec(
    experiment_name: str,
    *,
    retrieval_only: bool = False,
) -> ExperimentSpec:
    """为一次 A/B 选择判定 spec。

    已有同名 spec（检索三件套 / 显式指定的生成侧 spec）优先；否则按 ``retrieval_only``
    落到 ``retrieval_only_ab`` 或 ``generation_ab``。
    """
    if experiment_name in EXPERIMENT_SPECS:
        return EXPERIMENT_SPECS[experiment_name]
    return EXPERIMENT_SPECS["retrieval_only_ab" if retrieval_only else "generation_ab"]



def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    differences: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("differences must not be empty")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    size = len(differences)
    means = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _metric_verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "improvement"
    if ci_high < 0:
        return "regression"
    return "inconclusive"


def bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    samples: int = 2000,
    seed: int = 7,
) -> tuple[float, float] | None:
    """单个指标均值的 Bootstrap 95% 置信区间（离线、确定性）。

    生成侧指标（faithfulness 等）是有限的逐题分数：给出 CI 才能区分「真实差异」
    与「小样本噪声」。样本少于 2 条或 ``samples <= 0`` 时返回 ``None``。
    """
    finite = [float(value) for value in values if _finite(value)]
    if len(finite) < 2 or samples <= 0:
        return None
    rng = random.Random(f"{seed}:bootstrap_mean")
    size = len(finite)
    means = [
        sum(finite[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _index(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("run record missing case_id")
        if case_id in indexed:
            raise ValueError(f"duplicate case_id: {case_id}")
        indexed[case_id] = record
    return indexed


def compare_runs(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
    experiment_spec: ExperimentSpec,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 7,
    minimum_gold_coverage: float = 0.95,
    baseline_config: dict[str, Any] | None = None,
    candidate_config: dict[str, Any] | None = None,
    decision_tier: str | None = None,
    noise_floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    """先按 case_id 配对，再按独立样本组均值 Bootstrap；Gold/Silver 分开。"""
    baseline = list(baseline)
    candidate = list(candidate)
    population = experiment_spec.population
    if population:
        baseline = [row for row in baseline if str(row.get("stratum") or "") == population]
        candidate = [row for row in candidate if str(row.get("stratum") or "") == population]
    baseline_by_id = _index(baseline)
    candidate_by_id = _index(candidate)
    # 新运行由 run_config 提供指纹；直接调用者也可使用 records 上保存的指纹。
    def fingerprints(config: dict[str, Any] | None, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        values = [row.get("run_fingerprints") for row in records.values() if row.get("run_fingerprints")]
        if values and any(value != values[0] for value in values[1:]):
            return {**(config or {}), "inconsistent_run_fingerprints": True}
        if config is not None:
            merged = {**(config.get("metadata") or {}), **config}
            if values and any(key in merged and merged[key] != value for key, value in values[0].items()):
                merged["inconsistent_run_fingerprints"] = True
            return {**(values[0] if values else {}), **merged}
        return values[0] if values else {}

    before_config = fingerprints(baseline_config, baseline_by_id)
    after_config = fingerprints(candidate_config, candidate_by_id)
    strict = any(config.get("protocol_version") == "agentic-v2" for config in (before_config, after_config))
    comparability_reasons = []
    fingerprint_keys = ("dataset_fingerprint", "kb_fingerprint", "protocol_fingerprint", "judge_fingerprint")
    for key in (*fingerprint_keys, "protocol_version"):
        before_value, after_value = before_config.get(key), after_config.get(key)
        if before_value is not None and after_value is not None and before_value != after_value:
            comparability_reasons.append(key)
    if before_config.get("inconsistent_run_fingerprints") or after_config.get("inconsistent_run_fingerprints"):
        comparability_reasons.append("inconsistent_run_fingerprints")
    if before_config.get("comparable") is False or after_config.get("comparable") is False:
        comparability_reasons.append("run_marked_non_comparable")
    if strict:
        for key in (*fingerprint_keys, "protocol_version"):
            if any(config.get(key) in (None, "", "unavailable", "unknown") for config in (before_config, after_config)):
                comparability_reasons.append(f"missing_{key}")
    all_ids = sorted(set(baseline_by_id) | set(candidate_by_id))
    groups: dict[str, str] = {}
    group_tiers: dict[str, set[str]] = {}
    for case_id in all_ids:
        before, after = baseline_by_id.get(case_id, {}), candidate_by_id.get(case_id, {})
        if strict and any(row and not row.get("sample_group_id") for row in (before, after)):
            comparability_reasons.append(f"missing_sample_group_id:{case_id}")
        before_group = str(before.get("sample_group_id") or before.get("paired_case_id") or case_id)
        after_group = str(after.get("sample_group_id") or after.get("paired_case_id") or case_id)
        if before and after and before_group != after_group:
            comparability_reasons.append(f"sample_group_conflict:{case_id}")
        group = before_group if before else after_group
        groups[case_id] = group
        for row in (before, after):
            if row:
                group_tiers.setdefault(group, set()).add(str(row.get("tier")))
    for group, tier_values in group_tiers.items():
        if len(tier_values) > 1:
            comparability_reasons.append(f"sample_group_tier_conflict:{group}")
    complete_fingerprints = all(before_config.get(key) and after_config.get(key) for key in (fingerprint_keys if strict else ("dataset_fingerprint", "kb_fingerprint")))
    comparable = False if comparability_reasons else (True if complete_fingerprints else None)
    tiers: dict[str, Any] = {}
    negative_gold_means: set[str] = set()

    for tier in ("gold", "silver"):
        tier_ids = [
            case_id for case_id in all_ids
            if (baseline_by_id.get(case_id) or candidate_by_id.get(case_id) or {}).get("tier") == tier
            and baseline_by_id.get(case_id, {}).get("tier", tier) == tier
            and candidate_by_id.get(case_id, {}).get("tier", tier) == tier
        ]
        metric_names: set[str] = set()
        for case_id in tier_ids:
            metric_names.update((baseline_by_id.get(case_id, {}).get("metrics") or {}).keys())
            metric_names.update((candidate_by_id.get(case_id, {}).get("metrics") or {}).keys())

        metric_results: dict[str, Any] = {}
        tier_groups: dict[str, list[str]] = {}
        for case_id in tier_ids:
            tier_groups.setdefault(groups[case_id], []).append(case_id)
        for metric_name in sorted(metric_names):
            pairs: list[tuple[str, float, float]] = []
            paired_case_count = 0
            for group, member_ids in sorted(tier_groups.items()):
                members = []
                for case_id in member_ids:
                    before = (baseline_by_id.get(case_id, {}).get("metrics") or {}).get(metric_name)
                    after = (candidate_by_id.get(case_id, {}).get("metrics") or {}).get(metric_name)
                    if _finite(before) and _finite(after):
                        members.append((float(before), float(after)))
                # 缺失某个成员时整组无覆盖，避免用两臂不同成员代表同一事实。
                if len(members) == len(member_ids):
                    paired_case_count += len(members)
                    pairs.append((group, sum(a for a, _ in members) / len(members), sum(b for _, b in members) / len(members)))
            coverage = len(pairs) / len(tier_groups) if tier_groups else 0.0
            if not pairs:
                metric_results[metric_name] = {
                    "n": 0,
                    "paired_case_count": 0,
                    "coverage": round(coverage, 4),
                    "baseline": None,
                    "candidate": None,
                    "delta": None,
                    "ci95": None,
                    "verdict": "no_data",
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "largest_changes": [],
                }
                continue
            differences = [after - before for _, before, after in pairs]
            if tier == "gold" and sum(differences) < 0:
                negative_gold_means.add(metric_name)
            metric_rng = random.Random(f"{seed}:{tier}:{metric_name}")
            ci_low, ci_high = _bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                rng=metric_rng,
            )
            changes = sorted(
                (
                    {"case_id": case_id, "delta": round(after - before, 4)}
                    for case_id, before, after in pairs
                ),
                key=lambda item: (-abs(item["delta"]), item["case_id"]),
            )[:5]
            metric_results[metric_name] = {
                "n": len(pairs),
                "paired_case_count": paired_case_count,
                "coverage": round(coverage, 4),
                "baseline": round(sum(before for _, before, _ in pairs) / len(pairs), 4),
                "candidate": round(sum(after for _, _, after in pairs) / len(pairs), 4),
                "delta": round(sum(differences) / len(differences), 4),
                "ci95": [round(ci_low, 4), round(ci_high, 4)],
                "verdict": _metric_verdict(ci_low, ci_high),
                "wins": sum(diff > 0 for diff in differences),
                "losses": sum(diff < 0 for diff in differences),
                "ties": sum(diff == 0 for diff in differences),
                "largest_changes": changes,
            }
        tiers[tier] = {"n": len(tier_groups), "case_count": len(tier_ids), "metrics": metric_results}

    # 判定层：auto = 有 Gold 用 Gold，否则降级到 Silver 并标证据等级。
    requested_tier = str(decision_tier or experiment_spec.decision_tier or "auto")
    if requested_tier not in {"auto", "gold", "silver"}:
        raise ValueError(f"未知 decision_tier：{requested_tier}")
    if requested_tier == "auto":
        resolved_tier = "gold" if tiers["gold"]["n"] > 0 else "silver"
    else:
        resolved_tier = requested_tier
    decision_metrics = tiers[resolved_tier]["metrics"]
    primary = decision_metrics.get(experiment_spec.primary_metric)
    required_metrics = (experiment_spec.primary_metric, *experiment_spec.guardrails)
    insufficient_coverage = [
        metric_name for metric_name in required_metrics
        if (decision_metrics.get(metric_name) or {}).get("coverage", 0.0) < minimum_gold_coverage
    ]
    guardrail_failures = [
        metric_name
        for metric_name in experiment_spec.guardrails
        if (decision_metrics.get(metric_name) or {}).get("verdict") == "regression"
        or (strict and resolved_tier == "gold" and metric_name in negative_gold_means)
    ]
    if (
        not primary
        or insufficient_coverage
        or primary.get("verdict") == "no_data"
    ):
        decision = "no_decision"
    elif primary["verdict"] == "regression" or guardrail_failures:
        decision = "fail"
    elif primary["verdict"] == "improvement":
        decision = "pass"
    else:
        decision = "inconclusive"

    if comparable is False:
        decision = "no_decision"

    floor = {**(experiment_spec.noise_floor or {}), **(noise_floor or {})}
    primary_floor = floor.get(experiment_spec.primary_metric)
    noise_suppressed = bool(
        decision == "pass"
        and primary
        and primary_floor is not None
        and abs(float(primary.get("delta") or 0.0)) <= float(primary_floor)
    )
    if noise_suppressed:
        decision = "inconclusive"

    return {
        "experiment": experiment_spec.name,
        "primary_metric": experiment_spec.primary_metric,
        "guardrails": list(experiment_spec.guardrails),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "minimum_gold_coverage": minimum_gold_coverage,
        "tiers": tiers,
        "guardrail_failures": guardrail_failures,
        "insufficient_coverage": insufficient_coverage,
        "decision": decision,
        "decision_tier": resolved_tier,
        "evidence_grade": resolved_tier,
        "population": experiment_spec.population,
        "noise_floor": floor,
        "noise_suppressed": noise_suppressed,
        "comparable": comparable,
        "protocol_version": "agentic-v2" if strict else before_config.get("protocol_version"),
        "bootstrap_unit": "sample_group_id",
        # 只有 Gold 证据等级 + 可比 + pass 才允许对外声称提升。
        "can_claim_improvement": decision == "pass" and comparable is True and resolved_tier == "gold",
        "comparability_reasons": comparability_reasons,
        "comparability_status": "mismatch" if comparable is False else ("verified" if comparable else "unavailable"),
    }
