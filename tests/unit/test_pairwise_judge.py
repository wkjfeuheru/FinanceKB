"""成对裁判：顺序随机 + 换序复评（位置偏差防护）。

用假模型（只实现 ``invoke``）覆盖：换序冲突判 tie、顺序可复现、
非法输出记错误不伪造结论、空答案跳过、呈现顺序平衡、Prompt 含反长度指令。
"""

from __future__ import annotations

import json

import pytest

from finance_rag.src.eval.pairwise_judge import (
    PAIRWISE_JUDGE_PROMPT,
    PairwiseJudgeError,
    compare_answers,
    extract_json_object,
    format_contexts,
    judge_order,
    run_pairwise,
)


class FakeJudge:
    """按顺序吐出预设响应；记录 Prompt 便于断言。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("假裁判的预设响应已用尽")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _row(answer: str, *, contexts=("上下文一",), question_type="single_hop") -> dict:
    return {
        "case_id": "c1",
        "question_type": question_type,
        "response": answer,
        "retrieved_contexts": list(contexts),
    }


def _verdict(value: str) -> str:
    return json.dumps({"winner": value, "reason": "理由"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------


def test_prompt_contains_length_discipline_and_both_answers():
    judge = FakeJudge([_verdict("一")])
    result = judge_order(
        judge,
        question="2024 年净利润是多少？",
        row_a=_row("净利润 18.76 亿元。"),
        row_b=_row("公司注册资本变更完成。"),
        order="a_first",
    )
    assert result["winner"] == "a"
    prompt = judge.prompts[0]
    assert "简洁准确优于详细冗长" in prompt
    assert "净利润 18.76 亿元。" in prompt
    assert "公司注册资本变更完成。" in prompt
    assert PAIRWISE_JUDGE_PROMPT.startswith("你是严格的 RAG 回答质量裁判")


def test_winner_is_mapped_back_to_arm_for_each_presentation_order():
    assert judge_order(
        FakeJudge([_verdict("一")]), question="q",
        row_a=_row("a"), row_b=_row("b"), order="a_first",
    )["winner"] == "a"
    assert judge_order(
        FakeJudge([_verdict("一")]), question="q",
        row_a=_row("a"), row_b=_row("b"), order="b_first",
    )["winner"] == "b"


def test_format_contexts_handles_empty_and_truncates():
    assert format_contexts([]) == "（未检索到相关文档）"
    assert format_contexts(None) == "（未检索到相关文档）"
    rendered = format_contexts(["x" * 3000], limit=10)
    assert rendered == "[1] " + "x" * 10


# ---------------------------------------------------------------------------
# 位置偏差：顺序随机 + 换序复评
# ---------------------------------------------------------------------------


def test_order_swap_conflict_becomes_tie_and_flags_position_bias():
    # 两次都选"先出现的那个" → 换序后结论翻转，说明是位置偏差
    judge = FakeJudge([_verdict("一"), _verdict("一")])
    record = compare_answers(
        judge, case_id="c1", question="q", row_a=_row("A"), row_b=_row("B"), seed=7,
    )

    assert record["status"] == "evaluated"
    assert record["order_flip"] is True
    assert record["winner"] == "tie"
    assert record["winner_first_eval"] != record["winner_swapped_eval"]
    assert {record["winner_first_eval"], record["winner_swapped_eval"]} == {"a", "b"}


def test_order_invariant_verdict_keeps_the_winner():
    """同一臂在换序后仍被选中（裁判没有位置偏好）→ 结论采纳、不记翻转。"""
    # 先用同一 seed 探出该题的呈现顺序；"一" 永远指向先出现的那一臂
    probe = compare_answers(
        FakeJudge([_verdict("一")]), case_id="c1", question="q",
        row_a=_row("A"), row_b=_row("B"), seed=7, swap=False,
    )
    first_arm = probe["winner"]
    # 第一次评"一"（先出现者），换序后再评"二"（换序后先出现者=同一臂）
    judge = FakeJudge([_verdict("一"), _verdict("二")])
    record = compare_answers(
        judge, case_id="c1", question="q",
        row_a=_row("A"), row_b=_row("B"), seed=7,
    )
    assert record["presented_order"] == probe["presented_order"]
    assert record["winner"] == first_arm
    assert record["order_flip"] is False


def test_order_reproducible_by_seed_and_balanced_over_cases():
    rows = [{"case_id": f"c{i}", "question_type": "single_hop", "question": "q",
             "row_a": _row("A"), "row_b": _row("B")} for i in range(20)]

    first = run_pairwise(rows, FakeJudge([_verdict("一")] * 40), seed=11, swap=False)
    second = run_pairwise(rows, FakeJudge([_verdict("一")] * 40), seed=11, swap=False)
    assert [r["presented_order"] for r in first] == [r["presented_order"] for r in second]

    a_first = sum(r["presented_order"] == "a_first" for r in first)
    assert 4 <= a_first <= 16  # 随机化确实在起作用（不是恒定顺序）


# ---------------------------------------------------------------------------
# 失败模式：绝不伪造结论
# ---------------------------------------------------------------------------


def test_unparsable_output_becomes_judge_error_not_tie():
    judge = FakeJudge(["我觉得都还行，没有 JSON"])
    record = compare_answers(
        judge, case_id="c1", question="q", row_a=_row("A"), row_b=_row("B"),
    )
    assert record["status"] == "judge_error"
    assert record["winner"] is None
    assert "无法解析" in record["error"]


def test_illegal_winner_value_is_rejected():
    with pytest.raises(PairwiseJudgeError):
        judge_order(
            FakeJudge([_verdict("maybe")]), question="q",
            row_a=_row("A"), row_b=_row("B"), order="a_first",
        )


def test_model_exception_becomes_judge_error():
    judge = FakeJudge([RuntimeError("超时")])
    record = compare_answers(
        judge, case_id="c1", question="q", row_a=_row("A"), row_b=_row("B"),
    )
    assert record["status"] == "judge_error"
    assert "超时" in record["error"]


def test_empty_answer_is_skipped_without_calling_the_judge():
    judge = FakeJudge([])  # 一旦被调用就会抛 AssertionError
    record = compare_answers(
        judge, case_id="c1", question="q", row_a=_row(""), row_b=_row("B"),
    )
    assert record["status"] == "skipped_empty"
    assert record["winner"] is None
    assert record["answer_chars"] == {"a": 0, "b": 1}


def test_unknown_presentation_order_is_rejected():
    with pytest.raises(ValueError):
        judge_order(
            FakeJudge([_verdict("一")]), question="q",
            row_a=_row("A"), row_b=_row("B"), order="c_first",
        )


# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------


def test_extract_json_object_accepts_json_in_markdown():
    payload = extract_json_object('```json\n{"winner": "tie"}\n```')
    assert payload["winner"] == "tie"
    with pytest.raises(PairwiseJudgeError):
        extract_json_object("没有 JSON")
