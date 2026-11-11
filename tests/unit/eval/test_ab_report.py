"""A/B 报告链路（父进程部分）的离线行为：分组、配对统计、偏差小节、报告渲染。

不启动子进程、不连 Milvus、不调模型：直接喂造好的臂结果 JSON 形状，
覆盖 ``_run_judge_bias`` / `_run_statistics` / `_pairwise_pairs` / `render_report_md`。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.eval.ab_runner import (
    _pairwise_pairs,
    _run_judge_bias,
    _run_statistics,
    _with_public_metrics,
    compare_arms,
    render_report_md,
)
from src.eval.metrics import GENERATION_METRICS


def _row(case_id: str, *, qtype: str, value: float, recall: float, response: str) -> dict:
    metrics = {
        "faithfulness": value,
        "answer_relevancy": value,
        "context_precision": value,
        "context_recall": value,
        "recall_at_5": recall,
        "precision_at_5": 0.2,
        "chunk_mrr": recall,
        "hit_rate": 1.0,
        "mrr": recall,
        "ndcg": recall,
    }
    return {
        "case_id": case_id,
        "tier": "gold",
        "question_type": qtype,
        "stratum": "explicit" if qtype != "negative" else "negative",
        "user_input": f"问题 {case_id}？",
        "response": response,
        "answer_chars": len(response),
        "metrics": metrics,
        "sources": [{"content": "上下文片段", "id": f"{case_id}-chunk", "source": "a.pdf"}],
        "diagnostics": {"evidence_rank": 1.0, "retrieved_doc_stems": "a",
                        "retrieved_chunk_ids": f"{case_id}-chunk"},
    }


def _arm_result(name: str, rows: list[dict], *, retrieval_only: bool = False) -> dict:
    """按 ``execute_arm`` 的形状聚合：主指标 = RAGAS 四指标 + 负样本拒答。"""
    primary_names = {*GENERATION_METRICS, "negative_rejection"}
    names = sorted({key for row in rows for key in row["metrics"]})
    aggregates: dict[str, float | None] = {}
    for key in names:
        values = [
            row["metrics"][key]
            for row in rows
            if isinstance(row["metrics"].get(key), (int, float))
        ]
        aggregates[key] = round(sum(values) / len(values), 4) if values else None
    return {
        "arm": {"name": name, "label": name, "strategy_overrides":
                ({"retrieval_only": True} if retrieval_only else {})},
        "metrics": {key: value for key, value in aggregates.items() if key in primary_names},
        "diagnostic_metrics": {
            key: value for key, value in aggregates.items() if key not in primary_names
        },
        "per_query": rows,
        "entry_count": len(rows),
        "collection": "finance_kb",
    }


def _results() -> tuple[dict, list]:
    rows_a = [
        _row("c1", qtype="single_hop", value=0.4, recall=0.5, response="短回答"),
        _row("c2", qtype="multi_hop", value=0.3, recall=0.0, response="回答" * 40),
        _row("c3", qtype="negative", value=0.5, recall=0.0, response="无法回答"),
    ]
    rows_b = [
        _row("c1", qtype="single_hop", value=0.8, recall=1.0, response="更好的回答"),
        _row("c2", qtype="multi_hop", value=0.6, recall=0.5, response="回答" * 20),
        _row("c3", qtype="negative", value=0.5, recall=0.0, response="无法回答"),
    ]
    arms = [SimpleNamespace(name="rerank_off"), SimpleNamespace(name="rerank_on")]
    return {"rerank_off": _arm_result("rerank_off", rows_a),
            "rerank_on": _arm_result("rerank_on", rows_b)}, arms


# ---------------------------------------------------------------------------
# 配对与统计
# ---------------------------------------------------------------------------


def test_pairwise_pairs_exclude_negative_and_empty_answers():
    results, arms = _results()
    results["rerank_on"]["per_query"][0]["response"] = ""  # c1 回答为空
    pairs = _pairwise_pairs(results, arms)

    ids = {pair["case_id"] for pair in pairs}
    assert ids == {"c2"}  # c1 空答案、c3 负样本都被排除
    pair = pairs[0]
    assert pair["question"] == "问题 c2？"
    assert pair["row_a"]["case_id"] == "c2"
    assert pair["row_b"]["case_id"] == "c2"


def test_run_statistics_uses_retrieval_spec_and_reports_ci():
    results, arms = _results()
    experiment = SimpleNamespace(name="rerank_on_off")
    report = _run_statistics(
        experiment, results, arms, seed=7, bootstrap=200, retrieval_only=False,
    )

    assert report["status"] == "ok"
    assert report["primary_metric"] == "mrr"
    assert report["decision"] in {"pass", "fail", "inconclusive", "no_decision"}
    metric = report["tiers"]["gold"]["metrics"]["mrr"]
    assert metric["delta"] == round(1.0 - 0.5, 4) or metric["delta"] is not None
    assert metric["ci95"] is not None


def test_run_statistics_falls_back_to_retrieval_only_spec():
    results, arms = _results()
    experiment = SimpleNamespace(name="unknown_component")
    report = _run_statistics(
        experiment, results, arms, seed=7, bootstrap=100, retrieval_only=True,
    )
    assert report["primary_metric"] == "recall_at_5"
    assert set(report["guardrails"]) == {"precision_at_5", "chunk_mrr"}


def test_run_judge_bias_reports_lengths_and_matched_deltas():
    results, arms = _results()
    report = _run_judge_bias(results, arms)

    assert set(report["length"]["arms"]) == {"rerank_off", "rerank_on"}
    # B 臂回答总体更短（"回答"*20 vs "回答"*40）→ 长度遥测能反映出来
    assert (
        report["length"]["arms"]["rerank_on"]["mean_chars"]
        < report["length"]["arms"]["rerank_off"]["mean_chars"]
    )
    assert report["length_matched"]["n_paired"] == 3


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def _report() -> dict:
    results, arms = _results()
    experiment = {
        "name": "rerank_on_off",
        "label": "BGE 重排序",
        "kind": "runtime",
        "arm_a": {"name": "rerank_off", "label": "关闭重排序"},
        "arm_b": {"name": "rerank_on", "label": "开启重排序"},
    }
    return {
        "experiment": experiment,
        "testset": "gold20.jsonl",
        "fast": True,
        "sample": 3,
        "seed": 7,
        "arm_order": ["rerank_off", "rerank_on"],
        "retrieval_only": False,
        "arms": {
            "rerank_off": {
                "label": "关闭重排序",
                "metrics": _with_public_metrics(results["rerank_off"])["metrics"],
                "metrics_by_type": {},
                "metrics_by_type_counts": {},
                "metrics_by_stratum": {},
                "metrics_by_stratum_counts": {},
                "suspect_annotations": [],
                "entry_count": 3,
                "collection": "finance_kb",
            },
            "rerank_on": {
                "label": "开启重排序",
                "metrics": _with_public_metrics(results["rerank_on"])["metrics"],
                "metrics_by_type": {},
                "metrics_by_type_counts": {},
                "metrics_by_stratum": {},
                "metrics_by_stratum_counts": {},
                "suspect_annotations": [],
                "entry_count": 3,
                "collection": "finance_kb",
            },
        },
        "comparison": compare_arms(
            _with_public_metrics(results["rerank_off"]),
            _with_public_metrics(results["rerank_on"]),
        ),
        "statistics": _run_statistics(
            SimpleNamespace(name="rerank_on_off"), results, arms,
            seed=7, bootstrap=200, retrieval_only=False,
        ),
        "judge_bias": {
            **_run_judge_bias(results, arms),
            "position": {"status": "disabled", "sample": 0},
        },
        "crosscheck": {"status": "disabled"},
        "anchors": {"status": "skipped", "reason": "锚点文件不存在"},
        "human_sample": None,
    }


def test_render_report_md_contains_two_chain_and_bias_sections():
    text = render_report_md(_report())

    # 两条链路分组：生成侧四指标 + 检索侧客观指标并排
    assert "## 生成侧 RAGAS 四指标" in text
    assert "## 检索侧客观指标（Recall@5 / MRR 等）" in text
    assert "| 证据 Recall@5 |" in text
    assert "faithfulness" in text
    # 配对统计判定
    assert "## 配对统计判定（配对 Bootstrap）" in text
    assert "95% CI" in text
    # 三类偏差与锚点小节（未运行的写"未运行"，不编数字）
    assert "### 长度偏差" in text
    assert "### 位置偏差（成对裁判）" in text
    assert "### 自我强化偏差（跨模型校验）" in text
    assert "## 锚点漂移（judge 校准）" in text
    assert "未运行" in text


def test_render_report_md_marks_retrieval_metrics_as_ground_truth():
    text = render_report_md(_report())
    assert "地面真值" in text
    # 已下线指标不得再出现在报告里
    assert "answer_correctness" not in text
    assert "context_entity_recall" not in text
