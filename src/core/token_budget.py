"""模型窗口级总预算：Token 估算、历史滑窗与证据裁剪（RAG-15）。

背景（``docs/rag-review.md``）
------------------------------
问题、历史与证据此前都只有「单块字符截断」，没有模型窗口级总预算：长对话可能
超窗导致调用失败，规划还会把**全部历史**塞进 prompt。

估算策略
--------
调用链上没有本地分词器（不同 provider 的分词器不一致），因此用**保守的字符到
Token 换算**：按实测最坏密度 0.707 token/字符（bge-m3 分词器 + 中文/数字混合
语料）向上取整。宁可低估可用容量，也不让真实请求超窗。

裁剪顺序与不变量
----------------
* 历史：从**最新往前**保留，单条消息超限时按句边界裁剪并附截断标记；
* 证据：按相关度顺序逐条纳入，超出预算时在最后一条的**行/句边界**裁剪并标记；
* 被丢弃的证据不进入上下文，也不进入来源编号与事实核验输入——编号始终连续，
  不会出现悬空或错位的 `[n]`。
"""

from __future__ import annotations

from typing import Any

from src.core import config

#: 字符 → Token 的保守换算系数（实测最坏密度，宁多算不少算）
TOKENS_PER_CHAR = 0.707

#: 截断标记（上下文与历史共用，便于排障时一眼看出被裁剪）
TRUNCATION_NOTE = "……（已按上下文预算截断）"
HISTORY_TRUNCATION_NOTE = "……（历史消息过长已截断）"


def estimate_tokens(text: Any) -> int:
    """保守估算文本的 Token 数（空文本为 0）。"""
    if text is None:
        return 0
    content = text if isinstance(text, str) else str(text)
    if not content:
        return 0
    return int(len(content) * TOKENS_PER_CHAR) + 1


def available_input_tokens(*, context_tokens: int = 0) -> int:
    """模型输入侧的可用预算：窗口 − 预留输出 − 安全余量 − 已占用（如提示词）。"""
    return max(
        0,
        config.MODEL_CONTEXT_WINDOW
        - config.MODEL_RESERVED_OUTPUT_TOKENS
        - config.MODEL_SAFETY_MARGIN_TOKENS
        - max(0, int(context_tokens)),
    )


def clip_to_budget(text: str, max_tokens: int, *, note: str = TRUNCATION_NOTE) -> tuple[str, bool]:
    """把文本裁剪到 Token 预算内，返回 ``(文本, 是否被截断)``。

    优先在行边界（表格行、Markdown 行）裁剪；没有换行时退回句末标点；
    再退回硬切断。因此表格不会被从金额、行或标签中间切断。
    """
    content = text or ""
    if max_tokens <= 0:
        return "", bool(content.strip())
    if estimate_tokens(content) <= max_tokens:
        return content, False

    # 预算按字符数折算，并预留截断标记自身占用的长度
    note_tokens = estimate_tokens(note)
    char_budget = max(0, int((max_tokens - note_tokens) / TOKENS_PER_CHAR))
    if char_budget <= 0:
        return "", True
    head = content[:char_budget]

    separator_used = ""
    for separator in ("\n", "。", "；", ";", "，", ",", " "):
        position = head.rfind(separator)
        if position > 0:
            head = head[: position + (1 if separator != " " else 0)]
            separator_used = separator
            break
    if separator_used == "\n":
        # 行边界（表格行 / Markdown 行）：截断标记另起一行，
        # 这样最后一行仍是完整的表格行，不会把标记拼进单元格
        return f"{head.rstrip()}\n{note}", True
    return f"{head.rstrip()}{note}", True


def fit_history(
    history: list[dict[str, Any]] | None,
    max_tokens: int,
    *,
    max_turns: int | None = None,
    per_message_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 Token 预算保留**最新**的历史轮次，返回 ``(历史, 报告)``。

    报告字段：``kept`` / ``dropped`` / ``truncated_messages`` / ``tokens``，
    供调用方记录诊断，不改变「历史只用于理解指代」的既有语义。
    """
    messages = list(history or [])
    report: dict[str, Any] = {
        "input_messages": len(messages),
        "kept": 0,
        "dropped": 0,
        "truncated_messages": 0,
        "tokens": 0,
    }
    if not messages or max_tokens <= 0:
        report["dropped"] = len(messages)
        return [], report

    if max_turns is None:
        max_turns = config.CHAT_MAX_HISTORY_TURNS
    if per_message_tokens is None:
        per_message_tokens = estimate_tokens("中" * config.CHAT_MAX_HISTORY_MESSAGE_CHARS)

    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        if len(kept) >= max(0, int(max_turns)):
            break
        content = str(message.get("content") or "")
        remaining = max_tokens - used
        if remaining <= 0:
            break
        allowance = min(remaining, per_message_tokens)
        clipped, truncated = clip_to_budget(content, allowance, note=HISTORY_TRUNCATION_NOTE)
        if not clipped.strip():
            break
        if truncated:
            report["truncated_messages"] += 1
        kept.append({**message, "content": clipped})
        used += estimate_tokens(clipped)
    kept.reverse()

    report["kept"] = len(kept)
    report["dropped"] = len(messages) - len(kept)
    report["tokens"] = used
    return kept, report


def fit_evidence(
    docs: list[dict[str, Any]],
    max_tokens: int,
    *,
    block_budgets: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按总预算裁剪证据块（**保持原有相关度顺序**），返回 ``(证据, 报告)``。

    ``block_budgets`` 给出各类块的**单块上限**（表格更宽），总预算才是硬约束。
    超出总预算的块被丢弃（标记在报告里），最后一条在行/句边界裁剪并带标记。
    """
    report: dict[str, Any] = {
        "input_blocks": len(docs),
        "kept": 0,
        "dropped": 0,
        "truncated_blocks": 0,
        "tokens": 0,
    }
    if not docs:
        return [], report
    if max_tokens <= 0:
        report["dropped"] = len(docs)
        return [], report

    limits = block_budgets or {}
    kept: list[dict[str, Any]] = []
    used = 0
    for doc in docs:
        block_type = str(doc.get("block_type") or "text")
        per_block = limits.get(block_type) or limits.get("text")
        content = str(doc.get("content") or "")
        remaining = max_tokens - used
        if remaining <= 0:
            break
        allowance = remaining if per_block is None else min(remaining, per_block)
        clipped, truncated = clip_to_budget(content, allowance)
        if not clipped.strip():
            break
        if truncated:
            report["truncated_blocks"] += 1
        kept.append({**doc, "content": clipped, "truncated": bool(doc.get("truncated")) or truncated})
        used += estimate_tokens(clipped)
    report["kept"] = len(kept)
    report["dropped"] = len(docs) - len(kept)
    report["tokens"] = used
    return kept, report


def require_within_window(input_tokens: int, *, kind: str) -> None:
    """校验「输入 + 预留输出」不超窗口；超限时给出可操作的明确错误。"""
    total = input_tokens + config.MODEL_RESERVED_OUTPUT_TOKENS
    if total > config.MODEL_CONTEXT_WINDOW:
        raise ValueError(
            f"{kind}输入过长：约 {input_tokens} token + 预留输出 "
            f"{config.MODEL_RESERVED_OUTPUT_TOKENS} 超过模型窗口 "
            f"{config.MODEL_CONTEXT_WINDOW}，请缩短问题或历史"
        )


__all__ = [
    "HISTORY_TRUNCATION_NOTE",
    "TOKENS_PER_CHAR",
    "TRUNCATION_NOTE",
    "available_input_tokens",
    "clip_to_budget",
    "estimate_tokens",
    "fit_evidence",
    "fit_history",
    "require_within_window",
]
