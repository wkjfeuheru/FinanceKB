"""锚点样本：结构校验、区间判定、漂移汇总与 strict 门禁。

锚点是「分数已知的黄金用例」，用来盯住 judge 尺度漂移。全部用例离线。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finance_rag.src.eval.anchors import (
    AnchorDataError,
    check_bands,
    evaluate_anchors,
    load_anchors,
    render_anchor_markdown,
    summarize_anchors,
)

SHIPPED_ANCHORS = (
    Path(__file__).resolve().parents[2]
    / "finance_rag" / "src" / "eval" / "data" / "eval_anchors.jsonl"
)


def _anchor(anchor_id: str, metric: str, band: dict) -> dict:
    return {
        "anchor_id": anchor_id,
        "user_input": "问题？",
        "retrieved_contexts": ["上下文"],
        "response": "回答",
        "reference": "参考答案",
        "expected": {metric: band},
    }


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "anchors.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 随仓库发布的锚点数据
# ---------------------------------------------------------------------------


def test_shipped_anchors_are_structurally_valid_and_cover_both_ends():
    anchors = load_anchors(SHIPPED_ANCHORS)

    assert len(anchors) >= 5
    metrics = {
        metric for anchor in anchors for metric in anchor["expected"]
    }
    assert metrics <= {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    # 四个指标都要有锚点覆盖（否则漂移监控缺一条腿）
    assert metrics == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    # 既有"必须高"也有"必须低"的锚点，才能同时发现尺度上飘与下飘
    lows = [
        metric
        for anchor in anchors
        for metric, band in anchor["expected"].items()
        if "min" in band
    ]
    highs = [
        metric
        for anchor in anchors
        for metric, band in anchor["expected"].items()
        if "max" in band
    ]
    assert lows and highs
    assert all(anchor["rationale"] for anchor in anchors)


# ---------------------------------------------------------------------------
# 结构校验
# ---------------------------------------------------------------------------


def test_loader_rejects_duplicate_id(tmp_path):
    path = _write(tmp_path, [_anchor("a1", "faithfulness", {"min": 0.9})] * 2)
    with pytest.raises(AnchorDataError, match="重复"):
        load_anchors(path)


def test_loader_rejects_unknown_metric_and_bad_band(tmp_path):
    path = _write(tmp_path, [_anchor("a1", "answer_correctness", {"min": 0.9})])
    with pytest.raises(AnchorDataError, match="未知指标"):
        load_anchors(path)

    path = _write(tmp_path, [_anchor("a1", "faithfulness", {"min": 1.5})])
    with pytest.raises(AnchorDataError, match="0~1"):
        load_anchors(path)

    path = _write(tmp_path, [_anchor("a1", "faithfulness", {"min": 0.9, "max": 0.5})])
    with pytest.raises(AnchorDataError, match="min 大于 max"):
        load_anchors(path)

    path = _write(tmp_path, [_anchor("a1", "faithfulness", {})])
    with pytest.raises(AnchorDataError, match="min 或 max"):
        load_anchors(path)


def test_loader_rejects_missing_field_and_empty_file(tmp_path):
    row = _anchor("a1", "faithfulness", {"min": 0.9})
    row.pop("reference")
    path = _write(tmp_path, [row])
    with pytest.raises(AnchorDataError, match="缺少字段 reference"):
        load_anchors(path)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(AnchorDataError, match="为空"):
        load_anchors(empty)

    with pytest.raises(AnchorDataError, match="不存在"):
        load_anchors(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# 区间判定
# ---------------------------------------------------------------------------


def test_check_bands_boundary_and_slack_semantics():
    anchors = [_anchor("a1", "faithfulness", {"min": 0.9})]

    # 恰好等于下界算在区间内（>= 语义）
    report = check_bands({"a1": {"faithfulness": 0.9}}, anchors)
    assert report["verdict"] == "ok"
    assert report["n_in_band"] == 1

    # 越界 0.05：无 slack 判越界，slack=0.1 时放宽
    outside = check_bands({"a1": {"faithfulness": 0.85}}, anchors)
    assert outside["verdict"] == "warn"
    assert outside["violations"][0]["drift"] == pytest.approx(0.05)
    assert check_bands({"a1": {"faithfulness": 0.85}}, anchors, slack=0.1)["verdict"] == "ok"


def test_check_bands_marks_missing_metric_as_not_measured():
    anchors = [_anchor("a1", "context_recall", {"max": 0.5})]
    report = check_bands({"a1": {}}, anchors)
    assert report["n_missing"] == 1
    assert report["verdict"] == "warn"
    assert report["violations"][0]["reason"] == "未测到该指标"
    # 未测到不得被当成"合格"
    assert report["pass_rate"] == 0.0


def test_summarize_and_render_anchor_report():
    anchors = [_anchor("a1", "faithfulness", {"min": 0.9})]
    report = summarize_anchors(anchors, {"a1": {"faithfulness": 0.4}}, slack=0.1)

    assert report["verdict"] == "warn"
    assert report["pass_rate"] == 0.0
    assert report["mean_abs_drift"] == pytest.approx(0.4)

    lines = render_anchor_markdown(report)
    text = "\n".join(lines)
    assert "## 锚点漂移" in text
    assert "a1" in text and "faithfulness" in text
    # 未运行时明确写"未运行"，不编数字
    assert "未运行" in "\n".join(render_anchor_markdown(None))


# ---------------------------------------------------------------------------
# 与 RAGAS 链路的衔接
# ---------------------------------------------------------------------------


class _FakeEvaluator:
    def __init__(self, values):
        self.values = values
        self.calls: list[dict] = []

    def evaluate_batch(self, rows, *, fast=False, judge_model=None, judge_label="primary"):
        self.calls.append({"n": len(rows), "fast": fast, "judge_label": judge_label})
        return {
            "per_query": [
                {
                    "anchor_id": row["anchor_id"],
                    "metrics": {"faithfulness": self.values.get(row["anchor_id"])},
                }
                for row in rows
            ],
            "metric_errors": {},
            "fallback_metrics": [],
        }


def test_evaluate_anchors_runs_an_isolated_batch():
    anchors = [_anchor("a1", "faithfulness", {"min": 0.9}),
               _anchor("a2", "faithfulness", {"max": 0.3})]
    evaluator = _FakeEvaluator({"a1": 0.95, "a2": 0.1})

    result = evaluate_anchors(evaluator, anchors, fast=False, judge_model=object())

    assert result["n"] == 2
    assert result["observed"]["a1"] == {"faithfulness": 0.95}
    assert evaluator.calls[0]["n"] == 2  # 锚点独立成批，不混进题目均值
    report = summarize_anchors(anchors, result["observed"])
    assert report["verdict"] == "ok"


# ---------------------------------------------------------------------------
# 与 A/B 编排的接线：跳过 / 报告 / strict 门禁
# ---------------------------------------------------------------------------


def test_run_anchors_skips_when_file_missing(tmp_path, monkeypatch):
    from finance_rag.src.core import config
    from finance_rag.src.eval.ab_runner import _run_anchors

    monkeypatch.setattr(config, "EVAL_ANCHORS_PATH", str(tmp_path / "missing.jsonl"))
    report = _run_anchors(fast=True, strict=False)
    assert report["status"] == "skipped"


def test_run_anchors_strict_raises_on_band_violation(tmp_path, monkeypatch):
    from finance_rag.src.core import config
    from finance_rag.src.eval import ab_runner, ragas_eval
    from finance_rag.src.eval import anchors as anchors_module

    anchor_path = _write(tmp_path, [_anchor("a1", "faithfulness", {"min": 0.9})])
    monkeypatch.setattr(config, "EVAL_ANCHORS_PATH", str(anchor_path))
    monkeypatch.setattr(config, "get_model", lambda: object())
    monkeypatch.setattr(config, "get_judge_model", lambda: object())
    monkeypatch.setattr(ragas_eval, "RagasBatchEvaluator", lambda *a, **k: object())
    monkeypatch.setattr(
        anchors_module, "evaluate_anchors",
        lambda *a, **k: {
            "n": 1,
            "observed": {"a1": {"faithfulness": 0.1}},
            "metric_errors": {},
            "fallback_metrics": [],
        },
    )

    # 默认只报告：越界写进报告但不拦路
    report = ab_runner._run_anchors(fast=True, strict=False)
    assert report["status"] == "ok"
    assert report["verdict"] == "warn"
    assert report["violations"][0]["anchor_id"] == "a1"

    with pytest.raises(RuntimeError, match="锚点漂移"):
        ab_runner._run_anchors(fast=True, strict=True)

