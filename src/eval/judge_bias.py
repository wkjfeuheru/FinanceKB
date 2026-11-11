"""LLM-as-Judge 的三类典型偏差统计：位置 / 长度 / 自我强化。

三类偏差的来源与防护（与本项目评测链路的对应关系）：

* **位置偏差**：裁判偏好先出现（或后出现）的候选。防护 = 成对裁判顺序随机 +
  换序复评（见 :mod:`pairwise_judge`），本模块负责量化：「换序后结论翻转率」
  ``order_flip_rate`` 与「先出现者获胜率」``first_position_win_rate``。
* **长度偏差**：裁判倾向给更长的回答更高分。防护 = 裁判 Prompt 明确
  「简洁准确优于详细冗长」+ 本模块的「分数~长度」秩相关与长度匹配子集对照
  （``length_matched_deltas``：只在两臂回答字数接近的配对上重算 A/B 差值）。
* **自我强化偏差**：生成模型与裁判模型同族时系统性给自己加分。防护 = 第二模型族
  交叉校验（``self_preference_report``：偏置、秩相关、A/B 方向一致性）。

本模块只做纯统计，不发起任何 LLM 调用、不读配置：阈值一律由调用方传入，
因此可以完全离线单测。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import median
from typing import Any

#: 判定「长度偏差疑似」时的默认样本量下限：小样本的秩相关不可信。
MIN_CORRELATION_SAMPLES = 10

#: A/B 差值被认定为「无变化」的容差（用于方向一致性判定）。
DELTA_EPSILON = 1e-9


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metric(row: dict[str, Any], name: str) -> float | None:
    value = (row.get("metrics") or {}).get(name)
    return float(value) if _finite(value) else None


def answer_chars(row: dict[str, Any]) -> int | None:
    """取回答字数：优先显式字段 ``answer_chars``，否则按 ``response`` 现算。"""
    value = row.get("answer_chars")
    if _finite(value):
        return int(value)
    response = row.get("response")
    return len(response) if isinstance(response, str) and response else None


def _rank(values: Sequence[float]) -> list[float]:
    """秩（并列取平均秩）。"""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(xs: Sequence[Any], ys: Sequence[Any]) -> float | None:
    """Spearman 秩相关；样本不足或任一侧无方差时返回 ``None``。

    用秩相关而不是皮尔逊相关：分数与长度都不是正态分布，且裁判打分常大量并列。
    """
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=False)
        if _finite(x) and _finite(y)
    ]
    size = len(pairs)
    if size < 2:
        return None
    left = _rank([pair[0] for pair in pairs])
    right = _rank([pair[1] for pair in pairs])
    mean_left = sum(left) / size
    mean_right = sum(right) / size
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left, right, strict=False)
    )
    norm_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    norm_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if norm_left == 0 or norm_right == 0:
        return None
    return round(numerator / (norm_left * norm_right), 4)


def mean_abs_diff(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    """平均绝对差（只统计两侧都是有限数的配对）。"""
    values = [
        abs(float(a) - float(b))
        for a, b in pairs
        if _finite(a) and _finite(b)
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


# ---------------------------------------------------------------------------
# 长度偏差
# ---------------------------------------------------------------------------

def length_report(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    metrics: Sequence[str],
    *,
    corr_threshold: float = 0.3,
    min_samples: int = MIN_CORRELATION_SAMPLES,
) -> dict[str, Any]:
    """回答长度遥测 + 「分数 ~ 长度」秩相关。

    长度偏差无法用单侧统计消除：这里只回答「本臂内长回答是否系统性得分更高」，
    多长算长、分数差别是否真是长度造成的，交给 ``length_matched_deltas`` 的
    配对对照去回答。
    """
    arms: dict[str, Any] = {}
    correlations: dict[str, Any] = {}
    suspects: list[dict[str, Any]] = []

    for arm, rows in rows_by_arm.items():
        chars = [value for value in (answer_chars(row) for row in rows) if value is not None]
        arms[arm] = {
            "n": len(rows),
            "n_with_chars": len(chars),
            "mean_chars": round(sum(chars) / len(chars), 1) if chars else None,
            "median_chars": float(median(chars)) if chars else None,
        }
        per_metric: dict[str, Any] = {}
        for name in metrics:
            pairs = [
                (_metric(row, name), answer_chars(row))
                for row in rows
            ]
            pairs = [(value, size) for value, size in pairs if value is not None and size is not None]
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            per_metric[name] = {"n": len(pairs), "spearman": rho}
            if rho is not None and abs(rho) >= corr_threshold and len(pairs) >= min_samples:
                suspects.append(
                    {"arm": arm, "metric": name, "spearman": rho, "n": len(pairs)}
                )
        correlations[arm] = per_metric

    return {
        "arms": arms,
        "metric_length_correlation": correlations,
        "suspects": suspects,
        "length_bias_suspect": bool(suspects),
        "corr_threshold": corr_threshold,
        "min_samples": min_samples,
    }


def length_matched_deltas(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    max_ratio: float = 1.25,
    min_pairs: int = 8,
) -> dict[str, Any]:
    """只在「两臂回答字数接近」的配对上重算 A/B 差值，剥离长度混杂。

    判定：``max(chars_a, chars_b) / min(chars_a, chars_b) <= max_ratio``。
    配对不足 ``min_pairs`` 时 ``delta_matched`` 记 ``None``——不给近似结论。
    ``sign_flip`` 为真表示原始差值与长度匹配后差值方向相反，说明结论很可能是
    长度混杂造成的。
    """
    by_id_a = {str(row.get("case_id")): row for row in rows_a}
    by_id_b = {str(row.get("case_id")): row for row in rows_b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    matched_ids: list[str] = []
    unknown_length = 0
    for case_id in shared:
        chars_a = answer_chars(by_id_a[case_id])
        chars_b = answer_chars(by_id_b[case_id])
        if chars_a is None or chars_b is None or chars_a == 0 or chars_b == 0:
            unknown_length += 1
            continue
        if max(chars_a, chars_b) / min(chars_a, chars_b) <= max_ratio:
            matched_ids.append(case_id)

    per_metric: dict[str, Any] = {}
    confounded: list[str] = []
    for name in metrics:
        raw_pairs = [
            (_metric(by_id_a[case_id], name), _metric(by_id_b[case_id], name))
            for case_id in shared
        ]
        raw_pairs = [(a, b) for a, b in raw_pairs if a is not None and b is not None]
        matched_pairs = [
            (a, b)
            for case_id in matched_ids
            for a, b in [(_metric(by_id_a[case_id], name), _metric(by_id_b[case_id], name))]
            if a is not None and b is not None
        ]
        delta_raw = (
            round(sum(b - a for a, b in raw_pairs) / len(raw_pairs), 4)
            if raw_pairs
            else None
        )
        delta_matched = (
            round(sum(b - a for a, b in matched_pairs) / len(matched_pairs), 4)
            if len(matched_pairs) >= min_pairs
            else None
        )
        sign_flip = (
            delta_raw is not None
            and delta_matched is not None
            and delta_raw * delta_matched < 0
            and abs(delta_raw) > DELTA_EPSILON
            and abs(delta_matched) > DELTA_EPSILON
        )
        if sign_flip:
            confounded.append(name)
        per_metric[name] = {
            "n_raw": len(raw_pairs),
            "delta_raw": delta_raw,
            "n_matched": len(matched_pairs),
            "delta_matched": delta_matched,
            "sign_flip": sign_flip,
        }

    return {
        "n_paired": len(shared),
        "n_matched": len(matched_ids),
        "n_unknown_length": unknown_length,
        "max_ratio": max_ratio,
        "min_pairs": min_pairs,
        "metrics": per_metric,
        "length_confounded_metrics": confounded,
    }


# ---------------------------------------------------------------------------
# 位置偏差
# ---------------------------------------------------------------------------

def position_report(
    records: Sequence[dict[str, Any]],
    *,
    flip_threshold: float = 0.3,
) -> dict[str, Any]:
    """成对裁判结果的位置偏差统计。

    输入为 :func:`pairwise_judge.run_pairwise` 的逐条记录：

    * ``presented_order``：``"a_first"`` / ``"b_first"``，记录实际呈现顺序；
    * ``winner``：换序合并后的结论（``"a"`` / ``"b"`` / ``"tie"``）；
    * ``order_flip``：两次换序结论是否冲突；
    * ``status``：``"evaluated"`` / ``"skipped_empty"`` / ``"judge_error"``。
    """
    evaluated = [row for row in records if row.get("status") == "evaluated"]
    skipped = sum(row.get("status") == "skipped_empty" for row in records)
    errors = sum(row.get("status") == "judge_error" for row in records)

    wins = {"a": 0, "b": 0, "tie": 0}
    for row in evaluated:
        winner = str(row.get("winner") or "")
        if winner in wins:
            wins[winner] += 1

    decided = wins["a"] + wins["b"]
    first_wins = 0
    for row in evaluated:
        order = str(row.get("presented_order") or "")
        winner = str(row.get("winner") or "")
        if winner not in {"a", "b"}:
            continue
        first = "a" if order == "a_first" else "b"
        if winner == first:
            first_wins += 1

    swapped = [row for row in evaluated if row.get("swapped")]
    flips = sum(bool(row.get("order_flip")) for row in swapped)
    flip_rate = round(flips / len(swapped), 4) if swapped else None

    a_first = sum(str(row.get("presented_order")) == "a_first" for row in evaluated)
    b_first = len(evaluated) - a_first

    return {
        "n": len(evaluated),
        "skipped_empty": skipped,
        "judge_errors": errors,
        "win_rate": {
            "a": round(wins["a"] / len(evaluated), 4) if evaluated else None,
            "b": round(wins["b"] / len(evaluated), 4) if evaluated else None,
            "tie": round(wins["tie"] / len(evaluated), 4) if evaluated else None,
        },
        "decided": decided,
        "first_position_win_rate": (
            round(first_wins / decided, 4) if decided else None
        ),
        "order_flip_rate": flip_rate,
        "order_flip_count": flips,
        "swapped_count": len(swapped),
        "presentation_balance": {
            "a_first": a_first,
            "b_first": b_first,
            "a_first_share": round(a_first / len(evaluated), 4) if evaluated else None,
        },
        "position_bias_suspect": bool(flip_rate is not None and flip_rate > flip_threshold),
        "flip_threshold": flip_threshold,
    }


# ---------------------------------------------------------------------------
# 自我强化偏差（跨模型校验）
# ---------------------------------------------------------------------------

def _delta(values: dict[str, dict[str, float]]) -> float | None:
    if "a" not in values or "b" not in values:
        return None
    return round(values["b"] - values["a"], 4)


def self_preference_report(
    primary_by_arm: dict[str, Sequence[dict[str, Any]]],
    secondary_by_arm: dict[str, Sequence[dict[str, Any]]],
    metrics: Sequence[str],
    *,
    generator_family: str = "",
    primary_judge_family: str = "",
    bias_threshold: float = 0.1,
) -> dict[str, Any]:
    """主裁判 vs 第二裁判族的一致性、偏置与 A/B 方向一致性。

    ``primary_by_arm`` / ``secondary_by_arm`` 为 ``{"a": rows, "b": rows}``，
    且两侧必须在**同一批 case_id** 上评分，否则方向一致性没有意义。
    """
    per_metric: dict[str, Any] = {}
    suspects: list[dict[str, Any]] = []
    paired_total = 0

    for name in metrics:
        pairs = []
        means: dict[str, dict[str, float]] = {}
        for arm in ("a", "b"):
            primary = {str(row.get("case_id")): row for row in primary_by_arm.get(arm, [])}
            secondary = {
                str(row.get("case_id")): row for row in secondary_by_arm.get(arm, [])
            }
            values: list[float] = []
            for case_id in sorted(set(primary) & set(secondary)):
                left = _metric(primary[case_id], name)
                right = _metric(secondary[case_id], name)
                if left is None or right is None:
                    continue
                pairs.append((left, right))
                values.append(right)
            if values:
                means[arm] = {"primary": 0.0, "secondary": 0.0}
                primary_values = [
                    _metric(primary[case_id], name)
                    for case_id in sorted(set(primary) & set(secondary))
                ]
                primary_values = [value for value in primary_values if value is not None]
                means[arm]["primary"] = round(sum(primary_values) / len(primary_values), 4)
                means[arm]["secondary"] = round(sum(values) / len(values), 4)

        paired_total += len(pairs)
        bias = None
        if pairs:
            bias = round(
                sum(right - left for left, right in pairs) / len(pairs), 4
            )
        primary_delta = _delta({arm: bucket["primary"] for arm, bucket in means.items()})
        secondary_delta = _delta(
            {arm: bucket["secondary"] for arm, bucket in means.items()}
        )
        direction_agree = None
        if primary_delta is not None and secondary_delta is not None:
            direction_agree = (
                (primary_delta > DELTA_EPSILON and secondary_delta > DELTA_EPSILON)
                or (primary_delta < -DELTA_EPSILON and secondary_delta < -DELTA_EPSILON)
                or (
                    abs(primary_delta) <= DELTA_EPSILON
                    and abs(secondary_delta) <= DELTA_EPSILON
                )
            )
        if bias is not None and abs(bias) >= bias_threshold:
            suspects.append({"metric": name, "bias": bias})
        per_metric[name] = {
            "n_pairs": len(pairs),
            "mean_abs_diff": mean_abs_diff(pairs),
            "bias": bias,
            "spearman": spearman(
                [left for left, _ in pairs], [right for _, right in pairs]
            ),
            "primary_delta": primary_delta,
            "secondary_delta": secondary_delta,
            "delta_direction_agree": direction_agree,
        }

    same_family = bool(
        generator_family and primary_judge_family
        and generator_family.lower() == primary_judge_family.lower()
    )
    return {
        "generator_family": generator_family,
        "primary_judge_family": primary_judge_family,
        "self_judged": same_family,
        "metrics": per_metric,
        "paired_total": paired_total,
        "judge_bias_suspects": suspects,
        "self_preference_suspect": bool(suspects),
        "self_preference_unverified": paired_total == 0,
        "bias_threshold": bias_threshold,
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def render_bias_markdown(report: dict[str, Any]) -> list[str]:
    """把三类偏差的报告渲染成 Markdown 小节（缺项写"未运行"，不编数字）。"""
    lines: list[str] = ["", "## Judge 偏差（位置 / 长度 / 自我强化）", ""]

    length = report.get("length")
    lines.append("### 长度偏差")
    lines.append("")
    if not length:
        lines.append("未运行。")
    else:
        lines.append("| 臂 | n | 平均回答字数 | 中位字数 |")
        lines.append("|---|---:|---:|---:|")
        for arm, bucket in (length.get("arms") or {}).items():
            lines.append(
                f"| {arm} | {bucket.get('n', 0)} | "
                f"{_fmt(bucket.get('mean_chars'), 1)} | {_fmt(bucket.get('median_chars'), 1)} |"
            )
        lines.append("")
        suspects = length.get("suspects") or []
        if suspects:
            lines.append(
                f"- 疑似长度偏差（|Spearman| ≥ {length.get('corr_threshold')}，"
                f"n ≥ {length.get('min_samples')}）："
            )
            for item in suspects:
                lines.append(
                    f"  - 臂 `{item['arm']}` 指标 `{item['metric']}`：ρ={_fmt(item['spearman'])}"
                    f"（n={item['n']}）"
                )
        else:
            lines.append("- 未检测到显著的长度相关（样本不足时结论不成立）。")
        matched = report.get("length_matched")
        if matched:
            lines.append("")
            lines.append(
                f"- 长度匹配对照：配对 {matched.get('n_matched', 0)}/{matched.get('n_paired', 0)} "
                f"（字数比 ≤ {matched.get('max_ratio')}）"
            )
            for name, item in (matched.get("metrics") or {}).items():
                flag = " ⚠ 方向翻转" if item.get("sign_flip") else ""
                lines.append(
                    f"  - `{name}`：Δ原始={_fmt(item.get('delta_raw'))}，"
                    f"Δ匹配后={_fmt(item.get('delta_matched'))}{flag}"
                )

    lines.extend(["", "### 位置偏差（成对裁判）", ""])
    position = report.get("position")
    if not position or position.get("status") != "ok":
        status = (position or {}).get("status")
        note = "" if status in {None, "disabled"} else f"（状态：{status}）"
        lines.append(f"未运行（用 `--pairwise N` 开启）。{note}")
    else:
        lines.append(
            f"- 评估 {position['n']} 题；跳过空答案 {position['skipped_empty']}；"
            f"裁判错误 {position['judge_errors']}"
        )
        win = position.get("win_rate") or {}
        lines.append(
            f"- 胜率：A={_fmt(win.get('a'))}，B={_fmt(win.get('b'))}，tie={_fmt(win.get('tie'))}"
        )
        lines.append(
            f"- 先出现者获胜率：{position.get('first_position_win_rate')}；"
            f"换序翻转率：{position.get('order_flip_rate')}"
            f"（{position.get('order_flip_count')}/{position.get('swapped_count')}）"
        )
        balance = position.get("presentation_balance") or {}
        lines.append(
            f"- 呈现顺序平衡：A 先 {balance.get('a_first', 0)} / B 先 {balance.get('b_first', 0)}"
        )
        if position.get("position_bias_suspect"):
            lines.append(
                f"- ⚠ 疑似位置偏差：换序翻转率超过阈值 {position.get('flip_threshold')}，"
                "成对结论需换序平均后再用。"
            )

    lines.extend(["", "### 自我强化偏差（跨模型校验）", ""])
    self_pref = report.get("self_preference")
    if not self_pref:
        lines.append("未运行（用 `--cross-check N` 开启，需要第二裁判模型）。")
    else:
        lines.append(
            f"- 生成模型族：`{self_pref.get('generator_family') or '未知'}`；"
            f"主裁判族：`{self_pref.get('primary_judge_family') or '未知'}`；"
            f"自评：{'是' if self_pref.get('self_judged') else '否'}"
        )
        if self_pref.get("self_preference_unverified"):
            lines.append("- ⚠ 同一批样本没有可配对的第二裁判分数，自我强化偏差未校验。")
        lines.append("")
        lines.append("| 指标 | 配对数 | 平均绝对差 | 偏置(第二-主) | Spearman | Δ方向一致 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, item in (self_pref.get("metrics") or {}).items():
            agree = item.get("delta_direction_agree")
            agree_text = "N/A" if agree is None else ("一致" if agree else "不一致")
            lines.append(
                f"| {name} | {item.get('n_pairs', 0)} | {_fmt(item.get('mean_abs_diff'))} | "
                f"{_fmt(item.get('bias'))} | {_fmt(item.get('spearman'))} | {agree_text} |"
            )
        if self_pref.get("self_preference_suspect"):
            lines.append("")
            lines.append(
                f"- ⚠ 疑似自我强化偏差：偏置绝对值 ≥ {self_pref.get('bias_threshold')} 的指标："
                + ", ".join(
                    f"{item['metric']}({item['bias']})"
                    for item in self_pref.get("judge_bias_suspects") or []
                )
            )
    return lines
