"""把检索分层和 Agentic 对照收成一份面试能对着讲的报告。"""

from __future__ import annotations

from typing import Any


def render_interview_report(payload: dict[str, Any]) -> str:
    """数字只来自 ``payload``。没有实测的段落写阻塞原因，不补简历里的 90% / 1.0 / 14.1%。"""
    lines = [
        "# 30 题分层评测",
        "",
        f"- 运行：{payload.get('run_id') or '未完成'}",
        f"- 评测集：`{payload.get('dataset') or ''}`",
        "- 语料：当前 18 篇研报。30 题 = 8 道显式单跳 + 10 道跨文档 + 2 道负样本 + 10 道模糊配对。",
        "- 模糊题与原题共享证据，检索主指标用独立正例（explicit + cross_doc，18 题），不把 30 题加成一个总 Recall。",
        "",
        "## 指标定义",
        "",
        "- Recall@5：top-5 子块 id 里，命中的证据块占该题证据块的比例。负样本不进入这项。",
        "- MRR@5：第一个证据块在 top-5 中的位次的倒数；没进 top-5 记 0。",
        "- parent_restore_rate：子块已进 top-5 的证据里，父块长度允许替换、且引文出现在 PostgreSQL 父块全文中的比例。父块过长而未替换的次数单独计数。",
        "- context_precision：Ragas 对生成回答所用上下文的精确率，只在「标准检索 vs 三开关全开」这组里计算。",
        "",
        "## 检索四级",
        "",
    ]
    layers = payload.get("layers") or []
    if not layers:
        lines.append(payload.get("retrieval_error") or "尚未运行。")
        lines.append("")
    else:
        lines.extend([
            "| 层 | 独立正例 n | Recall@5 | MRR | 父块还原 | 未替换 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for layer in layers:
            independent = layer.get("independent_positive") or {}
            metrics = independent.get("metrics") or {}
            parent = independent.get("parent_restore") or {}
            lines.append(
                f"| {layer.get('strategy')} | {independent.get('n', 0)} | "
                f"{_fmt(metrics.get('recall_at_5'))} | {_fmt(metrics.get('mrr'))} | "
                f"{_fmt(parent.get('rate'))} | {parent.get('not_replaced', 0)} |"
            )
        lines.extend(["", "### 分层", ""])
        for layer in layers:
            lines.append(f"#### {layer.get('strategy')}")
            lines.append("")
            lines.extend(_stratum_table(layer.get("strata") or {}))
            vague = layer.get("vague_pairs") or {}
            delta = (vague.get("mean_delta") or {}).get("recall_at_5")
            lines.append(f"- 模糊题 Recall@5 相对配对题的平均差值：{_fmt(delta)}")
            lines.append("")
    lines.extend(["## Agentic 对照", ""])
    agentic = payload.get("agentic")
    if not agentic:
        lines.append(payload.get("agentic_error") or "尚未运行。")
        lines.append("")
    elif agentic.get("error"):
        lines.append(f"未跑完：{agentic['error']}")
    else:
        lines.append("标准检索关闭改写、元数据路由和反思；对照臂三件同时打开。")
        lines.append("")
        lines.extend(_comparison_table(agentic))
        lines.append("")
        lines.append(agentic.get("stratum_markdown") or "")
    lines.extend(["", "## 三个开关的证据召回", ""])
    ablations = payload.get("ablations") or []
    if not ablations:
        lines.append(payload.get("ablation_error") or "尚未运行。")
        lines.append("")
    for item in ablations:
        lines.append(f"### {item.get('name')}")
        lines.append("")
        if item.get("error"):
            lines.append(f"未跑完：{item['error']}")
        else:
            lines.extend(_comparison_table(item))
        lines.append("")
    lines.extend(["## 复现", ""])
    for command in payload.get("commands") or []:
        lines.append(f"- `{command}`")
    note = payload.get("note")
    if note:
        lines.extend(["", "## 说明", "", note])
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _stratum_table(strata: dict[str, Any]) -> list[str]:
    lines = ["| 层 | n | Recall@5 | MRR |", "|---|---:|---:|---:|"]
    for name in ("explicit", "cross_doc", "vague", "negative"):
        bucket = strata.get(name)
        if not bucket:
            continue
        metrics = bucket.get("metrics") or {}
        lines.append(
            f"| {name} | {bucket.get('n', 0)} | {_fmt(metrics.get('recall_at_5'))} | "
            f"{_fmt(metrics.get('mrr'))} |"
        )
    return lines


def _comparison_table(report: dict[str, Any]) -> list[str]:
    comparison = (report.get("comparison") or {}).get("metrics") or {}
    if not comparison and report.get("stderr_tail"):
        return [f"未跑完：{report['stderr_tail']}"]
    lines = ["| 指标 | 臂 A | 臂 B | Δ |", "|---|---:|---:|---:|"]
    preferred = [
        "context_precision", "recall_at_5", "chunk_mrr", "mrr",
        "faithfulness", "answer_relevancy",
    ]
    names = [name for name in preferred if name in comparison]
    names.extend(sorted(set(comparison) - set(names)))
    for name in names:
        row = comparison[name]
        lines.append(
            f"| {name} | {_fmt(row.get('arm_a'))} | {_fmt(row.get('arm_b'))} | {_fmt(row.get('delta'))} |"
        )
    error = report.get("arm_error")
    if error:
        lines.append("")
        lines.append(f"臂错误：{error}")
    return lines
