"""人工标注校准：把 LLM Judge 的尺度对齐到人类判断。

思路（工程校准的第二条）：每周从评估产物里分层抽出 50–100 条，人工按同一套
0~1 口径打分，再计算自动分与人工分的一致性（MAE / 偏置 / 秩相关）。

* ``export_human_sample``：从**已落盘的运行产物**抽样导出待标注 CSV（不花 LLM 成本）；
* ``load_human_scores`` + ``calibration_report``：回填后给出每指标的一致性判定与
  建议偏移量（judge 分数加上该偏移即对齐人类均值）。

导出/回填解耦是刻意的：标注是人的时间，重跑评测是钱和时间，两者互不阻塞。
本模块不 import ragas，也不发起任何网络调用。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .judge_bias import spearman
from .metrics import GENERATION_METRICS

logger = logging.getLogger(__name__)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


#: 人工标注默认抽样量（图片里的工程建议：每周 50–100 条）。
DEFAULT_HUMAN_SAMPLE_SIZE = 60

#: 一致性判定阈值。``aligned`` 要求偏置小且排序一致；``divergent`` 任一越界即成立。
CALIBRATION_ALIGNED_BIAS = 0.1
CALIBRATION_DIVERGENT_BIAS = 0.2
CALIBRATION_MIN_SPEARMAN = 0.5
CALIBRATION_DIVERGENT_SPEARMAN = 0.2

#: 每条样本送进 CSV 的上下文预览长度上限（够人判断忠实度即可）。
CONTEXT_PREVIEW_LIMIT = 500


def row_key(row: dict[str, Any], *, arm: str | None = None) -> str:
    """样本稳定标识：优先 ``case_id``，其次 ``unique_id``，最后用「臂 + 问题」兜底。

    早期运行产物（例如 ``scripts/results/ab/rerank_on_off/20260825-201550``）
    的 ``per_query`` 没有 ``case_id`` 字段。抽样导出与回填必须能对齐这类产物，
    否则校准只能对新报告做，老报告永远进不了对齐流程；因此这里按
    「臂名 + 问题原文」生成稳定键（同一题在同一臂内唯一）。
    """
    explicit = str(row.get("case_id") or row.get("unique_id") or "").strip()
    if explicit:
        return explicit
    question = _query(row).strip()
    arm_name = str(arm or row.get("arm") or row.get("_arm") or "").strip()
    if not question:
        return ""
    return f"{arm_name}:{question}" if arm_name else question


def _query(row: dict[str, Any]) -> str:
    return str(row.get("user_input") or row.get("query") or "")


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("response") or row.get("answer") or "")


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        value = metrics.get(metric)
        return float(value) if _finite(value) else None
    value = row.get(f"auto_{metric}")
    return float(value) if _finite(value) else None


def _context_preview(row: dict[str, Any], *, limit: int = CONTEXT_PREVIEW_LIMIT) -> str:
    sources = row.get("sources")
    texts: list[str] = []
    if isinstance(sources, list):
        texts = [
            str(item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in sources
        ]
    else:
        raw = row.get("retrieved_contexts") or row.get("contexts") or []
        if isinstance(raw, list):
            texts = [str(item) for item in raw]
    joined = " | ".join(text for text in texts if text)
    return joined[:limit]


def stratified_pick(rows: Sequence[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    """按 ``question_type`` 分层抽样（与 ``ab_runner._stratified_sample`` 同策略）。

    小样本下保证每个题型都有代表；余量补给题量最多的题型；顺序受 ``seed`` 控制。
    成对裁判抽样也复用它，保证「题型覆盖」与可复现性口径一致。
    """
    if sample <= 0 or sample >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("question_type") or "single_hop"), []).append(row)

    types = sorted(buckets)
    quota = {
        qtype: min(sample // len(types), len(buckets[qtype])) for qtype in types
    }
    remaining = sample - sum(quota.values())
    while remaining > 0:
        candidates = [qtype for qtype in types if quota[qtype] < len(buckets[qtype])]
        if not candidates:
            break
        candidates.sort(key=lambda qtype: (-len(buckets[qtype]), qtype))
        quota[candidates[0]] += 1
        remaining -= 1

    picked: list[dict[str, Any]] = []
    for qtype in types:
        pool = list(buckets[qtype])
        rng.shuffle(pool)
        picked.extend(pool[: quota[qtype]])
    picked.sort(key=lambda row: (str(row.get("question_type") or ""), str(row.get("case_id") or "")))
    return picked


def human_sample_fields(metrics: Sequence[str]) -> list[str]:
    return [
        "case_id", "arm", "question_type", "stratum", "query", "response",
        "response_chars", "contexts_preview", "reference",
        *(f"auto_{metric}" for metric in metrics),
        *(f"human_{metric}" for metric in metrics),
        "human_note",
    ]


def export_human_sample(
    per_query_by_arm: dict[str, Sequence[dict[str, Any]]],
    *,
    out_csv: str | Path,
    n: int = DEFAULT_HUMAN_SAMPLE_SIZE,
    seed: int = 7,
    arm: str | None = None,
    metrics: Sequence[str] = GENERATION_METRICS,
) -> Path:
    """分层抽样导出待人工标注的 CSV；``human_*`` 列为空待填。

    Args:
        per_query_by_arm: ``{臂名: per_query 记录列表}``。
        arm: 只导出指定臂；``None`` 表示所有臂都导出（每臂各抽 ``n`` 条）。
    """
    arms = [arm] if arm else sorted(per_query_by_arm)
    rows: list[dict[str, Any]] = []
    for arm_name in arms:
        picked = stratified_pick(list(per_query_by_arm.get(arm_name) or []), n, seed)
        for row in picked:
            rows.append({"arm": arm_name, **row})

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = human_sample_fields(metrics)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = {
                # 稳定样本键：早期产物没有 case_id 时回落为「臂 + 问题」
                "case_id": row_key(row),
                "arm": row.get("arm", ""),
                "question_type": row.get("question_type", ""),
                "stratum": row.get("stratum", ""),
                "query": _query(row),
                "response": _answer(row),
                "response_chars": len(_answer(row)),
                "contexts_preview": _context_preview(row),
                "reference": str(row.get("reference") or row.get("ground_truth") or ""),
            }
            for metric in metrics:
                value = _metric_value(row, metric)
                record[f"auto_{metric}"] = "" if value is None else value
                record[f"human_{metric}"] = ""
            record["human_note"] = ""
            writer.writerow(record)
    logger.info("人工标注样本已导出 %d 条 → %s", len(rows), out_path)
    return out_path


def load_human_scores(
    path: str | Path,
    *,
    metrics: Sequence[str] = GENERATION_METRICS,
) -> dict[str, dict[str, Any]]:
    """读取人工标注 CSV，返回 ``{case_id: {"metrics": {...}, "note": str}}``。

    只保留至少填了一项 ``human_*`` 的行；分数必须在 0~1 之间，否则报错。
    同一 ``case_id`` 出现多行（多臂导出）时以第一次为准并记 ``duplicates``。
    """
    scores: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                continue
            values: dict[str, float] = {}
            for metric in metrics:
                raw = str(row.get(f"human_{metric}") or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"第 {line_number} 行 human_{metric} 不是数字：{raw!r}"
                    ) from exc
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"第 {line_number} 行 human_{metric} 超出 0~1：{value}"
                    )
                values[metric] = value
            if not values:
                continue
            if case_id in scores:
                duplicates.append(case_id)
                continue
            scores[case_id] = {
                "metrics": values,
                "note": str(row.get("human_note") or ""),
            }
    if duplicates:
        logger.warning("人工标注 CSV 出现重复 case_id（已取首次）：%s", duplicates[:10])
    return scores


def calibration_report(
    auto_rows: Iterable[dict[str, Any]],
    human_scores: dict[str, dict[str, Any]],
    *,
    metrics: Sequence[str] = GENERATION_METRICS,
) -> dict[str, Any]:
    """逐指标计算 judge 与人类的一致性、偏置与建议偏移量。

    ``bias = mean(human - auto)``；``recommended_offset`` 即该偏置——把 judge 分数
    加上它，均值就对齐人类。``verdict``：``aligned`` / ``offset`` / ``divergent`` /
    ``no_data``。
    """
    auto_by_id: dict[str, dict[str, Any]] = {}
    for row in auto_rows:
        key = row_key(row)
        if key and key not in auto_by_id:
            auto_by_id[key] = row

    per_metric: dict[str, Any] = {}
    matched_cases: set[str] = set()
    for metric in metrics:
        pairs: list[tuple[float, float]] = []
        for case_id, human in human_scores.items():
            if metric not in human["metrics"]:
                continue
            row = auto_by_id.get(case_id)
            if row is None:
                continue
            auto = _metric_value(row, metric)
            if auto is None:
                continue
            pairs.append((auto, human["metrics"][metric]))
            matched_cases.add(case_id)
        n = len(pairs)
        mae = (
            round(mean(abs(human - auto) for auto, human in pairs), 4) if pairs else None
        )
        bias = (
            round(mean(human - auto for auto, human in pairs), 4) if pairs else None
        )
        rho = spearman([auto for auto, _ in pairs], [human for _, human in pairs])
        agreement = (
            round(
                sum(abs(human - auto) <= 0.2 for auto, human in pairs) / n, 4
            )
            if n
            else None
        )
        if n == 0:
            verdict = "no_data"
        elif (
            (bias is not None and abs(bias) > CALIBRATION_DIVERGENT_BIAS)
            or (rho is not None and rho < CALIBRATION_DIVERGENT_SPEARMAN)
        ):
            verdict = "divergent"
        elif (
            bias is not None
            and abs(bias) <= CALIBRATION_ALIGNED_BIAS
            and (rho is None or rho >= CALIBRATION_MIN_SPEARMAN)
        ):
            verdict = "aligned"
        else:
            verdict = "offset"
        per_metric[metric] = {
            "n": n,
            "mae": mae,
            "bias": bias,
            "spearman": rho,
            "agreement_within_0.2": agreement,
            "recommended_offset": bias,
            "verdict": verdict,
        }

    human_count = len(human_scores)
    return {
        "n_human": human_count,
        "n_matched": len(matched_cases),
        "unreviewed_or_unmatched": human_count - len(matched_cases),
        "thresholds": {
            "aligned_bias": CALIBRATION_ALIGNED_BIAS,
            "divergent_bias": CALIBRATION_DIVERGENT_BIAS,
            "aligned_spearman": CALIBRATION_MIN_SPEARMAN,
            "divergent_spearman": CALIBRATION_DIVERGENT_SPEARMAN,
        },
        "metrics": per_metric,
    }


def render_calibration_markdown(report: dict[str, Any]) -> list[str]:
    lines = ["", "## 人工标注校准（judge vs 人类）", ""]
    lines.append(
        f"- 人工标注 {report.get('n_human', 0)} 条；可配对 {report.get('n_matched', 0)} 条；"
        f"未能配对 {report.get('unreviewed_or_unmatched', 0)} 条"
    )
    lines.append("")
    lines.append("| 指标 | n | MAE | 偏置(人-机) | Spearman | ≤0.2 一致率 | 建议偏移 | 判定 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for metric, item in (report.get("metrics") or {}).items():
        def _fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {metric} | {item.get('n', 0)} | {_fmt(item.get('mae'))} | "
            f"{_fmt(item.get('bias'))} | {_fmt(item.get('spearman'))} | "
            f"{_fmt(item.get('agreement_within_0.2'))} | "
            f"{_fmt(item.get('recommended_offset'))} | {item.get('verdict')} |"
        )
    return lines


def load_run_arms(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """读取一次运行的两类落盘布局：

    * A/B 产物：``<run_dir>/<arm>.json``（含 ``per_query``）；
    * 统一运行器产物：``<run_dir>/records.jsonl``（逐题记录）。

    Returns:
        ``{臂名: per_query 记录列表}``；两种布局都不存在时返回空字典。
        每行会带上 ``_arm`` 标记：没有 ``case_id`` 的早期产物靠「臂 + 问题」
        生成稳定键，抽样与回填才能对齐。
    """
    directory = Path(run_dir)
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in {"report.json", "comparison.json", "run_config.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("per_query"), list):
            arms[path.stem] = [
                {"_arm": path.stem, **row} for row in payload["per_query"]
            ]
    records_path = directory / "records.jsonl"
    if records_path.exists():
        rows: list[dict[str, Any]] = []
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append({"_arm": "records", **json.loads(line)})
        if rows:
            arms.setdefault("records", rows)
    return arms


def write_calibration_report(run_dir: str | Path, report: dict[str, Any]) -> Path:
    out_path = Path(run_dir) / "human_calibration.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return out_path
