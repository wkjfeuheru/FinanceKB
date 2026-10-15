"""成对裁判（pairwise LLM judge）：位置偏差的顺序随机 + 换序复评。

为什么需要它：RAGAS 的四项指标是**绝对打分**，两次运行之间的差异可能只是裁判
当天尺度漂移；而"哪一臂的回答更好"这种比较，必须让裁判同时看到两个候选。
一旦同屏比较，就会引入位置偏差（裁判偏爱先出现或后出现的候选）。

防护做法（两条一起用）：

1. **顺序随机**：每题用 ``Random(f"{seed}:{case_id}")`` 决定谁先出现，保证
   跨题平衡且可复现（``presentation_balance`` 可审计随机化是否真的生效）；
2. **换序复评**：同一题再评一次（把两候选位置对调）。两次结论一致才采纳，
   冲突则判 ``tie`` 并记 ``order_flip``。翻转率本身就量化了位置偏差强度。

Prompt 里显式声明"简洁准确优于详细冗长"，用于抑制长度偏差（长度遥测与长度匹配
对照见 :mod:`judge_bias`）。

本模块不 import ragas：JSON 提取优先用 ragas 的工具（函数内惰性导入），
失败时回退到本地正则 + ``json``，因此可以在不装 ragas 的环境里单测。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

#: 单个回答在 Prompt 中的截断长度，避免超长回答把 Prompt 撑爆。
ANSWER_LIMIT = 4000

#: 每条检索上下文的截断长度。
CONTEXT_LIMIT = 2000

PAIRWISE_JUDGE_PROMPT = """你是严格的 RAG 回答质量裁判。下面是同一个用户问题的两个候选回答，
两个候选各自带有自己检索到的上下文。

请只依据**各自提供的上下文**判断忠实性（上下文没有支撑的陈述属于幻觉），
然后判断哪个候选对用户更有用：信息正确、覆盖问题要点、且更贴合用户原问题者胜。

评分纪律（必须遵守）：
- **简洁准确优于详细冗长**：不要因为回答更长、更详细、条目更多而加分；
  啰嗦但没有多给出正确信息的回答不加分，甚至应扣分。
- 不要因为候选编号（一/二）或出现顺序而改变结论。
- 两个候选都没有明显优势时判 `tie`，不要强行选一个。

用户问题：
{question}

候选一（来自臂 {first_arm}）检索到的上下文：
{first_contexts}

候选一（来自臂 {first_arm}）的回答：
{first_answer}

候选二（来自臂 {second_arm}）检索到的上下文：
{second_contexts}

候选二（来自臂 {second_arm}）的回答：
{second_answer}

只输出 JSON，不要输出其它文字：
{{"winner": "一" 或 "二" 或 "tie", "reason": "一句话理由（不超过 60 字）"}}"""


class PairwiseJudgeError(RuntimeError):
    """裁判输出无法解析或取值非法。"""


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value[:limit]


def format_contexts(
    contexts: Sequence[Any] | None,
    *,
    limit: int = CONTEXT_LIMIT,
) -> str:
    """把检索结果拼成带序号的上下文文本（与 ``build_context_text`` 同形）。"""
    items = [str(item) for item in (contexts or []) if item]
    if not items:
        return "（未检索到相关文档）"
    return "\n\n---\n\n".join(
        f"[{index}] {_truncate(item, limit)}" for index, item in enumerate(items, 1)
    )


def _invoke(model: Any, prompt: str) -> str:
    """调用 langchain 聊天模型并取纯文本（兼容假模型直接返回字符串）。"""
    result = model.invoke(prompt)
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if content is None:
        return str(result)
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """从裁判输出里取出 JSON 对象；失败抛 :class:`PairwiseJudgeError`。"""
    raw = text or ""
    try:
        from ragas.prompt.utils import extract_json  # 惰性导入：不装 ragas 也能用

        payload = extract_json(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:  # noqa: BLE001 - 回退到本地解析
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    raise PairwiseJudgeError(f"裁判输出无法解析为 JSON：{raw[:200]}")


def _normalize_winner(payload: dict[str, Any]) -> str:
    """把裁判输出归一成 ``"a"`` / ``"b"`` / ``"tie"``（一 → 先出现者）。"""
    raw = str(payload.get("winner", "")).strip().lower()
    if raw in {"tie", "draw", "equal", "平", "平局", "相当"}:
        return "tie"
    if raw in {"一", "1", "first", "a"}:
        return "first"
    if raw in {"二", "2", "second", "b"}:
        return "second"
    raise PairwiseJudgeError(f"裁判 winner 取值非法：{payload.get('winner')!r}")


def judge_order(
    model: Any,
    *,
    question: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    order: str,
) -> dict[str, Any]:
    """按给定呈现顺序评一次。

    Args:
        order: ``"a_first"`` 或 ``"b_first"``。

    Returns:
        ``{"presented_order", "winner", "reason"}``，``winner`` 已映射回臂。
    """
    if order not in {"a_first", "b_first"}:
        raise ValueError(f"未知呈现顺序：{order}")
    first, second = ("a", "b") if order == "a_first" else ("b", "a")
    rows = {"a": row_a, "b": row_b}
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        question=question,
        first_arm=first,
        first_contexts=format_contexts(rows[first].get("retrieved_contexts")),
        first_answer=_truncate(rows[first].get("response"), ANSWER_LIMIT) or "（空回答）",
        second_arm=second,
        second_contexts=format_contexts(rows[second].get("retrieved_contexts")),
        second_answer=_truncate(rows[second].get("response"), ANSWER_LIMIT) or "（空回答）",
    )
    payload = extract_json_object(_invoke(model, prompt))
    verdict = _normalize_winner(payload)
    if verdict == "first":
        winner = first
    elif verdict == "second":
        winner = second
    else:
        winner = "tie"
    return {
        "presented_order": order,
        "winner": winner,
        "reason": str(payload.get("reason", ""))[:200],
    }


def compare_answers(
    model: Any,
    *,
    case_id: str,
    question: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    seed: int = 7,
    swap: bool = True,
) -> dict[str, Any]:
    """对同一题的两臂回答做顺序随机 + 换序复评。

    ``swap=False`` 时只评一次（省钱），但位置偏差**无法**量化，结果里
    ``swapped`` 为 ``False``，统计时会被排除在翻转率之外。
    """
    chars_a = len(str(row_a.get("response") or ""))
    chars_b = len(str(row_b.get("response") or ""))
    base = {
        "case_id": case_id,
        "question_type": str(row_a.get("question_type") or row_b.get("question_type") or ""),
        "answer_chars": {"a": chars_a, "b": chars_b},
    }
    if not chars_a or not chars_b:
        return {**base, "status": "skipped_empty", "winner": None,
                "order_flip": False, "swapped": bool(swap), "reason": "存在空回答"}

    first_order = (
        "a_first" if random.Random(f"{seed}:{case_id}").random() < 0.5 else "b_first"
    )
    other_order = "b_first" if first_order == "a_first" else "a_first"
    try:
        first = judge_order(
            model, question=question, row_a=row_a, row_b=row_b, order=first_order
        )
        if swap:
            second = judge_order(
                model, question=question, row_a=row_a, row_b=row_b, order=other_order
            )
        else:
            second = None
    except Exception as exc:  # 解析失败/调用失败都记错误，绝不伪造 tie
        logger.warning("成对裁判失败（%s）：%s", case_id, exc)
        return {**base, "status": "judge_error", "winner": None, "error": str(exc),
                "order_flip": False, "swapped": bool(swap),
                "presented_order": first_order}

    if second is None:
        winner, order_flip = first["winner"], False
    elif first["winner"] == second["winner"]:
        winner, order_flip = first["winner"], False
    else:
        winner, order_flip = "tie", True

    return {
        **base,
        "status": "evaluated",
        "presented_order": first_order,
        "winner": winner,
        "winner_first_eval": first["winner"],
        "winner_swapped_eval": second["winner"] if second else None,
        "order_flip": order_flip,
        "swapped": bool(swap),
        "reason": first.get("reason", ""),
        "reason_swapped": (second or {}).get("reason", ""),
    }


def run_pairwise(
    pairs: Sequence[dict[str, Any]],
    judge: Any,
    *,
    seed: int = 7,
    swap: bool = True,
) -> list[dict[str, Any]]:
    """逐题执行成对裁判（顺序执行，样本量由调用方控制）。

    Args:
        pairs: ``[{"case_id", "question", "row_a", "row_b"}, ...]``。
        judge: 聊天模型；假模型只需实现 ``invoke(prompt) -> str``。
    """
    records: list[dict[str, Any]] = []
    for item in pairs:
        records.append(
            compare_answers(
                judge,
                case_id=str(item["case_id"]),
                question=str(item["question"]),
                row_a=item["row_a"],
                row_b=item["row_b"],
                seed=seed,
                swap=swap,
            )
        )
    return records
