"""Judge 偏差统计（长度 / 位置 / 自我强化）的纯函数行为。

全部用例离线：只喂造好的记录，不调用任何模型。
"""

from __future__ import annotations

from finance_rag.src.eval.judge_bias import (
    length_matched_deltas,
    length_report,
    position_report,
    self_preference_report,
    spearman,
)


def _row(case_id: str, value: float | None, chars: int | None = None) -> dict:
    row: dict = {"case_id": case_id, "metrics": {"faithfulness": value}}
    if chars is not None:
        row["answer_chars"] = chars
    return row


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def test_spearman_known_values_and_degenerate_input():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    # 任一侧零方差 → None（不是 0，避免被当成"无相关"）
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    # 样本不足
    assert spearman([1], [1]) is None
    assert spearman([], []) is None
    # 并列秩仍可计算
    assert spearman([1, 2, 3, 4], [1, 2, 2, 2]) is not None


# ---------------------------------------------------------------------------
# 长度偏差
# ---------------------------------------------------------------------------


def test_length_report_flags_suspicious_correlation_with_sample_threshold():
    rows = [_row(f"c{i}", i / 10, chars=100 + i * 50) for i in range(12)]
    report = length_report({"arm": rows}, ["faithfulness"], corr_threshold=0.3)

    assert report["arms"]["arm"]["mean_chars"] is not None
    assert report["length_bias_suspect"] is True
    assert report["suspects"][0]["metric"] == "faithfulness"
    assert report["suspects"][0]["n"] == 12

    # 样本量不足（默认 min_samples=10）时不得报疑似
    small = rows[:5]
    assert length_report({"arm": small}, ["faithfulness"])["suspects"] == []


def test_length_report_uses_response_when_answer_chars_missing():
    rows = [{"case_id": "c1", "metrics": {"faithfulness": 1.0}, "response": "abcd"}]
    report = length_report({"arm": rows}, ["faithfulness"])
    assert report["arms"]["arm"]["mean_chars"] == 4.0


def test_length_matched_deltas_detects_sign_flip_from_length_confounding():
    # 8 对长度相当：B 比 A 低 0.2
    matched_a = [_row(f"m{i}", 0.8, chars=100) for i in range(8)]
    matched_b = [_row(f"m{i}", 0.6, chars=100) for i in range(8)]
    # 4 对长度悬殊（A 极短、B 极长）：B 比 A 高 1.0 —— 典型长度混杂
    long_a = [_row(f"l{i}", 0.0, chars=50) for i in range(4)]
    long_b = [_row(f"l{i}", 1.0, chars=400) for i in range(4)]

    report = length_matched_deltas(
        matched_a + long_a, matched_b + long_b, ["faithfulness"],
        max_ratio=1.25, min_pairs=8,
    )

    item = report["metrics"]["faithfulness"]
    assert item["n_matched"] == 8
    assert item["delta_raw"] > 0 > item["delta_matched"]
    assert item["sign_flip"] is True
    assert report["length_confounded_metrics"] == ["faithfulness"]


def test_length_matched_deltas_returns_none_when_pairs_insufficient():
    rows_a = [_row("c1", 0.1, chars=100)]
    rows_b = [_row("c1", 0.9, chars=100)]
    report = length_matched_deltas(rows_a, rows_b, ["faithfulness"], min_pairs=8)
    item = report["metrics"]["faithfulness"]
    assert item["n_matched"] == 1
    assert item["delta_matched"] is None  # 配对不足不给近似结论
    assert item["sign_flip"] is False


# ---------------------------------------------------------------------------
# 位置偏差
# ---------------------------------------------------------------------------


def test_position_report_quantifies_flip_rate_and_balance():
    records = [
        {"status": "evaluated", "presented_order": "a_first", "winner": "a",
         "order_flip": False, "swapped": True},
        {"status": "evaluated", "presented_order": "b_first", "winner": "a",
         "order_flip": False, "swapped": True},
        {"status": "evaluated", "presented_order": "a_first", "winner": "tie",
         "order_flip": True, "swapped": True},
        {"status": "judge_error", "presented_order": "a_first", "winner": None,
         "order_flip": False, "swapped": True},
        {"status": "skipped_empty", "presented_order": "a_first", "winner": None,
         "order_flip": False, "swapped": True},
    ]
    report = position_report(records, flip_threshold=0.3)

    assert report["n"] == 3
    assert report["judge_errors"] == 1
    assert report["skipped_empty"] == 1
    assert report["win_rate"]["a"] == round(2 / 3, 4)
    assert report["win_rate"]["tie"] == round(1 / 3, 4)
    # 先出现者只在第 1 条获胜（第 2 条先出现的是 B 但 A 胜，第 3 条是 tie）
    assert report["first_position_win_rate"] == 0.5
    assert report["order_flip_rate"] == round(1 / 3, 4)
    assert report["position_bias_suspect"] is True
    assert report["presentation_balance"] == {
        "a_first": 2, "b_first": 1, "a_first_share": round(2 / 3, 4),
    }


def test_position_report_without_swap_has_no_flip_rate():
    records = [
        {"status": "evaluated", "presented_order": "a_first", "winner": "a",
         "order_flip": False, "swapped": False},
    ]
    report = position_report(records)
    assert report["order_flip_rate"] is None
    assert report["position_bias_suspect"] is False
    assert report["swapped_count"] == 0


# ---------------------------------------------------------------------------
# 自我强化偏差
# ---------------------------------------------------------------------------


def test_self_preference_report_direction_agreement_and_bias():
    primary = {
        "a": [_row("c1", 0.5), _row("c2", 0.6)],
        "b": [_row("c1", 0.7), _row("c2", 0.8)],
    }
    secondary = {
        "a": [_row("c1", 0.4), _row("c2", 0.45)],
        "b": [_row("c1", 0.75), _row("c2", 0.85)],
    }
    report = self_preference_report(
        primary, secondary, ["faithfulness"],
        generator_family="deepseek", primary_judge_family="deepseek",
        bias_threshold=0.1,
    )

    metric = report["metrics"]["faithfulness"]
    assert metric["n_pairs"] == 4
    assert metric["primary_delta"] == 0.2
    assert metric["secondary_delta"] == 0.375
    assert metric["delta_direction_agree"] is True
    assert metric["bias"] == -0.0375
    assert report["self_judged"] is True
    assert report["self_preference_suspect"] is False
    assert report["self_preference_unverified"] is False


def test_self_preference_report_flags_large_judge_bias():
    primary = {"a": [_row("c1", 0.9)], "b": [_row("c1", 0.9)]}
    secondary = {"a": [_row("c1", 0.5)], "b": [_row("c1", 0.5)]}
    report = self_preference_report(
        primary, secondary, ["faithfulness"],
        generator_family="deepseek", primary_judge_family="qwen",
        bias_threshold=0.1,
    )
    assert report["metrics"]["faithfulness"]["bias"] == -0.4
    assert report["self_preference_suspect"] is True
    assert report["self_judged"] is False
    assert report["judge_bias_suspects"][0]["metric"] == "faithfulness"


def test_self_preference_report_marks_unverified_when_no_overlap():
    primary = {"a": [_row("c1", 0.5)], "b": [_row("c1", 0.7)]}
    secondary = {"a": [_row("other", 0.5)], "b": [_row("other", 0.7)]}
    report = self_preference_report(
        primary, secondary, ["faithfulness"],
        generator_family="deepseek", primary_judge_family="deepseek",
    )
    assert report["self_preference_unverified"] is True
    assert report["paired_total"] == 0
    assert report["self_judged"] is True
