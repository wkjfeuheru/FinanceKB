"""锚点样本：分数已知的黄金用例，用于监控 judge 标准是否漂移。

思路（工程校准的第一条）：每批评估都掺入少量**人工构造、期望分数已知**的样本，
用它们的实测分数盯住裁判尺度；锚点集体偏离期望区间 = 裁判标准漂移或 Prompt 变动，
此时任何 A/B 差异都不能直接采信。

设计取舍：

* 期望值是**区间**（``{"min": 0.9}`` / ``{"max": 0.3}``）而不是点值：LLM judge
  本身有抖动，点值会制造假失败；
* 默认**只报告、不拦路**，``EVAL_ANCHOR_STRICT`` / ``--anchor-strict`` 才升级为
  硬失败（锚点期望由人写，硬门禁必须由使用方显式开启）；
* 锚点用独立批次评估，不混进题目均值，避免污染正式指标。

本模块不 import ragas：只依赖 :mod:`metrics` 里的指标名单，因此可离线单测。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .metrics import GENERATION_METRICS

logger = logging.getLogger(__name__)

ANCHOR_REQUIRED_KEYS = (
    "anchor_id",
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "expected",
)


class AnchorDataError(ValueError):
    """锚点文件结构非法。"""


def _validate_anchor(anchor: Any, line_number: int) -> dict[str, Any]:
    prefix = f"第 {line_number} 行"
    if not isinstance(anchor, dict):
        raise AnchorDataError(f"{prefix}：锚点必须是 JSON 对象")
    for key in ANCHOR_REQUIRED_KEYS:
        if key not in anchor:
            raise AnchorDataError(f"{prefix}：缺少字段 {key}")
    anchor_id = str(anchor["anchor_id"]).strip()
    if not anchor_id:
        raise AnchorDataError(f"{prefix}：anchor_id 不能为空")
    if not str(anchor["user_input"]).strip():
        raise AnchorDataError(f"{prefix}（{anchor_id}）：user_input 不能为空")
    if not isinstance(anchor["retrieved_contexts"], list):
        raise AnchorDataError(f"{prefix}（{anchor_id}）：retrieved_contexts 必须是数组")
    expected = anchor["expected"]
    if not isinstance(expected, dict) or not expected:
        raise AnchorDataError(f"{prefix}（{anchor_id}）：expected 必须是非空对象")
    for metric, band in expected.items():
        if metric not in GENERATION_METRICS:
            raise AnchorDataError(
                f"{prefix}（{anchor_id}）：expected 含未知指标 {metric!r}，"
                f"可用指标：{list(GENERATION_METRICS)}"
            )
        if not isinstance(band, dict) or not ({"min", "max"} & set(band)):
            raise AnchorDataError(
                f"{prefix}（{anchor_id}）：{metric} 的区间必须含 min 或 max"
            )
        for bound in ("min", "max"):
            if bound not in band:
                continue
            value = band[bound]
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise AnchorDataError(
                    f"{prefix}（{anchor_id}）：{metric}.{bound} 必须是 0~1 的数"
                )
        low, high = band.get("min"), band.get("max")
        if low is not None and high is not None and float(low) > float(high):
            raise AnchorDataError(
                f"{prefix}（{anchor_id}）：{metric} 的 min 大于 max"
            )
    normalized = dict(anchor)
    normalized["anchor_id"] = anchor_id
    normalized["rationale"] = str(anchor.get("rationale", ""))
    return normalized


def load_anchors(path: str | Path) -> list[dict[str, Any]]:
    """加载并校验锚点文件（JSONL）。

    Raises:
        AnchorDataError: 文件不存在、JSON 非法、字段缺失、指标名未知或区间越界。
    """
    anchor_path = Path(path)
    if not anchor_path.exists():
        raise AnchorDataError(f"锚点文件不存在：{anchor_path}")
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    with anchor_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnchorDataError(f"第 {line_number} 行不是有效 JSON：{exc}") from exc
            anchor = _validate_anchor(raw, line_number)
            if anchor["anchor_id"] in seen:
                raise AnchorDataError(f"anchor_id 重复：{anchor['anchor_id']}")
            seen.add(anchor["anchor_id"])
            anchors.append(anchor)
    if not anchors:
        raise AnchorDataError(f"锚点文件为空：{anchor_path}")
    return anchors


def _band_distance(value: float, band: dict[str, Any], slack: float) -> float:
    """分数离期望区间有多远（0 表示在区间内）。``slack`` 只放宽、不收紧。"""
    low = band.get("min")
    high = band.get("max")
    if low is not None and value < float(low) - slack:
        return round(float(low) - slack - value, 4)
    if high is not None and value > float(high) + slack:
        return round(value - (float(high) + slack), 4)
    return 0.0


def check_bands(
    observed: dict[str, dict[str, float | None]],
    anchors: Sequence[dict[str, Any]],
    *,
    slack: float = 0.0,
) -> dict[str, Any]:
    """逐锚点逐指标比对实测值与期望区间。

    Args:
        observed: ``{anchor_id: {metric: value}}``，缺失或 None 视为未测到。

    Returns:
        ``{"n_checked", "n_in_band", "n_missing", "pass_rate", "mean_abs_drift",
        "violations", "verdict"}``；``verdict`` 只可能是 ``ok`` / ``warn``，
        是否据此失败由调用方的 strict 开关决定。
    """
    checked = 0
    in_band = 0
    missing = 0
    drifts: list[float] = []
    violations: list[dict[str, Any]] = []

    for anchor in anchors:
        anchor_id = str(anchor["anchor_id"])
        values = observed.get(anchor_id) or {}
        for metric, band in (anchor.get("expected") or {}).items():
            checked += 1
            value = values.get(metric)
            if not isinstance(value, (int, float)):
                missing += 1
                violations.append(
                    {"anchor_id": anchor_id, "metric": metric, "observed": None,
                     "band": band, "drift": None, "reason": "未测到该指标"}
                )
                continue
            distance = _band_distance(float(value), band, slack)
            drifts.append(distance)
            if distance == 0.0:
                in_band += 1
            else:
                violations.append(
                    {"anchor_id": anchor_id, "metric": metric,
                     "observed": round(float(value), 4), "band": dict(band),
                     "drift": distance, "reason": "超出期望区间"}
                )

    return {
        "n_checked": checked,
        "n_in_band": in_band,
        "n_missing": missing,
        "pass_rate": round(in_band / checked, 4) if checked else None,
        "mean_abs_drift": (
            round(sum(drifts) / len(drifts), 4) if drifts else None
        ),
        "violations": violations,
        "slack": slack,
        "verdict": "ok" if checked and in_band == checked else "warn",
    }


def evaluate_anchors(
    evaluator: Any,
    anchors: Sequence[dict[str, Any]],
    *,
    fast: bool = False,
    judge_model: Any = None,
) -> dict[str, Any]:
    """用与正式评估同一条 RAGAS 链路给锚点打分（独立批次，不混进题目均值）。"""
    rows = [
        {
            "anchor_id": anchor["anchor_id"],
            "user_input": anchor["user_input"],
            "retrieved_contexts": list(anchor.get("retrieved_contexts") or []),
            "response": anchor.get("response") or "",
            "reference": anchor.get("reference") or "",
        }
        for anchor in anchors
    ]
    result = evaluator.evaluate_batch(rows, fast=fast, judge_model=judge_model)
    observed = {
        str(item.get("anchor_id")): dict(item.get("metrics") or {})
        for item in result.get("per_query") or []
    }
    return {
        "n": len(rows),
        "observed": observed,
        "metric_errors": result.get("metric_errors") or {},
        "fallback_metrics": result.get("fallback_metrics") or [],
    }


def summarize_anchors(
    anchors: Sequence[dict[str, Any]],
    observed: dict[str, dict[str, float | None]],
    *,
    slack: float = 0.0,
) -> dict[str, Any]:
    """把锚点实测值与期望区间合成一份可落盘的报告。"""
    bands = check_bands(observed, anchors, slack=slack)
    return {
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "rationale": anchor.get("rationale", ""),
                "expected": anchor.get("expected") or {},
                "observed": observed.get(str(anchor["anchor_id"])) or {},
            }
            for anchor in anchors
        ],
        **bands,
    }


def render_anchor_markdown(report: dict[str, Any] | None) -> list[str]:
    """锚点漂移小节（未运行时明确写"未运行"，不编数字）。"""
    lines = ["", "## 锚点漂移（judge 校准）", ""]
    if not report:
        lines.append("未运行（`--anchors` 需要锚点文件；`--no-anchors` 可显式关闭）。")
        return lines
    lines.append(
        f"- 锚点 {len(report.get('anchors') or [])} 条；检查 {report.get('n_checked', 0)} 项；"
        f"区间内 {report.get('n_in_band', 0)} 项；未测到 {report.get('n_missing', 0)} 项"
    )
    lines.append(
        f"- 判定：{report.get('verdict')}；通过率 {report.get('pass_rate')}；"
        f"平均越界幅度 {report.get('mean_abs_drift')}"
    )
    violations = report.get("violations") or []
    if violations:
        lines.append("")
        lines.append("| 锚点 | 指标 | 实测 | 期望区间 | 越界幅度 |")
        lines.append("|---|---|---:|---|---:|")
        for item in violations:
            band = item.get("band") or {}
            band_text = ", ".join(
                f"{key}={band[key]}" for key in ("min", "max") if key in band
            )
            observed = item.get("observed")
            lines.append(
                f"| {item.get('anchor_id')} | {item.get('metric')} | "
                f"{'N/A' if observed is None else f'{float(observed):.4f}'} | "
                f"{band_text} | {item.get('drift')} |"
            )
    return lines


def anchor_rows_preview(anchors: Iterable[dict[str, Any]]) -> str:
    """一行摘要（日志/终端用）。"""
    ids = [str(anchor["anchor_id"]) for anchor in anchors]
    return f"{len(ids)} 条：" + ", ".join(ids)
