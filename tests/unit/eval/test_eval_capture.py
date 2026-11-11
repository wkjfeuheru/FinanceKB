"""采集回归：规划/停止原因必须进逐题记录，延迟聚合剔除冷启动。"""

from __future__ import annotations

from src.eval.ab_runner import _latency_summary, _paired_records, _pipeline_state


def test_pipeline_state_keeps_top_level_diagnostics():
    pipeline = {
        "plan": {"subproblems": [{"id": "q1", "query": "营收"}]},
        "coverage": None,
        "decision": "generate",
        "stop_reason": "",
        "retry_count": 0,
        "attempted_queries": ["营收"],
        "warnings": [],
    }

    state = _pipeline_state(pipeline)

    assert state["decision"] == "generate"
    assert state["retry_count"] == 0
    assert state["attempted_queries"] == ["营收"]
    # 空值不占位：诊断里出现空覆盖会把归因误判成「已评估、无缺失」。
    assert "coverage" not in state
    assert "stop_reason" not in state


def test_latency_summary_excludes_cold_start():
    rows = [{"retrieval_latency_ms": value} for value in (1000.0, 100.0, 200.0, 300.0, 400.0)]

    stats = _latency_summary(rows)["retrieval"]

    assert stats["n"] == 5
    assert stats["cold_start_ms"] == 1000.0
    assert stats["p50_ms"] == 250.0
    assert stats["p95_ms"] == 385.0


def test_latency_summary_ignores_missing_and_zero():
    rows = [{"retrieval_latency_ms": None}, {"retrieval_latency_ms": 0.0}, {}]
    assert _latency_summary(rows) == {}


def test_paired_records_keep_stratum_for_population_filter():
    rows = _paired_records(
        {
            "per_query": [
                {
                    "case_id": "c1",
                    "tier": "silver",
                    "stratum": "cross_doc",
                    "metrics": {"recall_at_5": 1.0},
                }
            ]
        }
    )

    assert rows[0]["stratum"] == "cross_doc"
    assert rows[0]["metrics"] == {"recall_at_5": 1.0}
