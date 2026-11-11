"""200 条评测集的评分规则、生成器守卫与失败阶段归因。

本模块只放**纯函数**（不连知识库、不调模型），供三条链路复用：

* ``scripts/run_eval200_ab.py``：跑 @10 检索 A/B 并出报告；
* ``src/eval/ab_runner.py``：复用失败归因（仓库只有这一份阶段定义）；
* ``src/eval/cli.py``：生成评测集时复用「抄题/引文不可定位」守卫。

阶段定义与优先级见 :data:`FAILURE_STAGES`。归因只对「候选臂失败」的题生效，
两臂都失败的题归 ``shared_failure``，不把原有缺陷算到优化头上。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any

from .metrics import assess_parent_restore, summarize_parent_restore
from .statistics import bootstrap_mean_ci

#: 报告固定深度。Precision 分母恒为 10，便于跨运行比较。
K = 10

#: 引文可定位所需的最短归一化长度；过短的公共片段不算「引文」。
MIN_QUOTE_CHARS = 12

#: 失败阶段（顺序即归因优先级，从基础设施到排序）。
FAILURE_STAGES: tuple[str, ...] = (
    "error",
    "not_searched",
    "filter_dropped",
    "rewrite_drift",
    "vague_unresolved",
    "cross_doc_partial",
    "parent_split",
    "miss_coarse",
    "rerank_demoted",
    "rerank_cut",
    "other",
    "shared_failure",
)

_TRANSIENT = re.compile(
    r"429|rate.?limit|too many requests|限流|timeout|timed out|超时|50[234]|temporar|unavailable",
    re.IGNORECASE,
)

#: 主观提问标记。真正的金融事实题很少出现这些词。
_SUBJECTIVE_MARKERS = (
    "怎么看", "如何看", "怎么样", "如何评价", "值得买", "能不能买",
    "投资建议", "有什么看法", "后市如何", "还会涨", "能不能涨",
)

#: 可检索的具体指标/口径。与上面的主观标记配合判定「是否可出检索题」。
_CONCRETE_ASPECTS = re.compile(
    r"营收|收入|净利润|毛利率|净利率|增速|同比|环比|占比|金额|价格|订单|产能|"
    r"市占率|评级|目标价|估值|PE|PB|分红|股息|现金流|资产负债|代码|期间|"
    r"季度|年报|中报|回购|解禁|限售|募投|研发|费用"
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def is_transient_failure(text: str) -> bool:
    """限流/超时/网关错误：这类失败不代表能力，不能算进优化效果。"""
    return bool(_TRANSIENT.search(str(text or "")))


# ---------------------------------------------------------------------------
# 引文定位与生成守卫
# ---------------------------------------------------------------------------


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """去掉空白并保留「归一化下标 → 原文下标」映射。"""
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text or ""):
        if not char.isspace():
            chars.append(char)
            mapping.append(index)
    return "".join(chars), mapping


def locate_quote(content: str, hint: str) -> str | None:
    """在 ``content`` 中定位 ``hint`` 的最长公共片段，返回**原文**切片。

    ``hint`` 可以带前后缀（LLM 常把引文包在解释里），也可以与原文空格、
    换行不同。返回的是 ``content`` 的真实子串，因此可以直接作为证据引文。
    """
    if not content or not hint:
        return None
    normalized_content, mapping = _normalize_with_map(content)
    normalized_hint, _ = _normalize_with_map(hint)
    if len(normalized_hint) < MIN_QUOTE_CHARS or len(normalized_content) < MIN_QUOTE_CHARS:
        return None
    match = SequenceMatcher(
        None, normalized_content, normalized_hint, autojunk=False
    ).find_longest_match(0, len(normalized_content), 0, len(normalized_hint))
    if match.size < MIN_QUOTE_CHARS:
        return None
    start = mapping[match.a]
    end = mapping[match.a + match.size - 1] + 1
    return content[start:end]


def copies_span(text: str, content: str) -> bool:
    """判断 ``text`` 是否抄录了 ``content`` 中的长片段（抄题/泄题）。"""
    return locate_quote(content, text) is not None


def vague_rejection_reason(query: str) -> str | None:
    """生成器守卫：问句是否「看着具体、实际不可检索」。"""
    text = (query or "").strip()
    if not text:
        return "空问题"
    if len(text) < 6:
        return "问题过短，缺少可检索信息"
    if any(marker in text for marker in _SUBJECTIVE_MARKERS) and not _CONCRETE_ASPECTS.search(text):
        return "问句偏主观且没有可检索的指标或期间；请补充具体指标、期间或标的"
    return None


_SECTION_LABELS = ("标准答案", "问题", "原文引文A", "原文引文B", "原文引文")


def parse_generation(raw: str, multi: bool = False) -> dict[str, Any] | None:
    """解析生成器输出，返回 ``{"query", "answer", "quotes"}``。

    同时兼容 ``## 问题`` 与 ``<问题>`` 两种分隔写法，以及全部写在一行的情况。
    """
    text = (raw or "").strip()
    if not text or text.upper().startswith("SKIP"):
        return None
    for tag, heading in (
        ("<问题>", "## 问题"),
        ("<标准答案>", "### 标准答案"),
        ("<原文引文A>", "### 原文引文A"),
        ("<原文引文B>", "### 原文引文B"),
    ):
        text = text.replace(tag, heading)
    parts = re.split(r"#{2,3}[ \t]*", text)
    query = ""
    answer = ""
    quotes: list[str] = []
    for part in parts:
        body = part.strip()
        if not body:
            continue
        for label in _SECTION_LABELS:
            if not body.startswith(label):
                continue
            value = body[len(label):].strip()
            if label == "问题":
                query = value
            elif label == "标准答案":
                answer = value
            elif value:
                quotes.append(value)
            break
    if len(query) < 4:
        return None
    if (len(quotes) < 2) if multi else (not quotes):
        return None
    return {"query": query, "answer": answer, "quotes": quotes}


# ---------------------------------------------------------------------------
# 证据与过滤器
# ---------------------------------------------------------------------------


def _date_within(value: str, constraint: Any) -> bool:
    if isinstance(constraint, dict):
        lower = str(constraint.get("gte") or constraint.get("gt") or "")
        upper = str(constraint.get("lte") or constraint.get("lt") or "")
        if lower and value < lower:
            return False
        if upper and value > upper:
            return False
        return True
    if isinstance(constraint, (list, tuple, set)):
        return value in {str(item) for item in constraint}
    return value == str(constraint)


def filters_exclude(hard_filters: dict[str, Any], evidence_meta: Sequence[dict[str, Any]]) -> bool:
    """硬过滤是否会挡掉标注证据（任何一条被挡即视为有风险）。"""
    if not hard_filters or not evidence_meta:
        return False
    for key, constraint in hard_filters.items():
        if constraint in (None, "", [], {}):
            continue
        for item in evidence_meta:
            value = str((item or {}).get(key) or "")
            if not value:
                continue
            if key in {"date", "start_date", "published_at"}:
                if not _date_within(value, constraint):
                    return True
                continue
            allowed = {
                str(entry)
                for entry in (constraint if isinstance(constraint, (list, tuple, set)) else [constraint])
            }
            if value not in allowed:
                return True
    return False


def _entities_retained(record: dict[str, Any], arm: dict[str, Any]) -> bool:
    """原问句里出现的代码/券商，改写后必须还在至少一条检索词里。"""
    query = str(record.get("query") or "")
    hints = record.get("entity_hints") or {}
    attempted = [str(item) for item in arm.get("attempted_queries") or [] if str(item).strip()]
    if not attempted:
        return True
    for key in ("codes", "brokers", "security_names"):
        for entity in hints.get(key) or []:
            entity = str(entity)
            if entity and entity in query and not any(entity in item for item in attempted):
                return False
    return True


def _not_rewritten(record: dict[str, Any], arm: dict[str, Any]) -> bool:
    attempted = [str(item) for item in arm.get("attempted_queries") or [] if str(item).strip()]
    if not attempted:
        return True
    normalize = lambda value: re.sub(r"\s+", "", str(value))  # noqa: E731
    return normalize(attempted[0]) == normalize(record.get("query") or "")


# ---------------------------------------------------------------------------
# 失败阶段归因
# ---------------------------------------------------------------------------


def _coarse_ids(arm: dict[str, Any]) -> tuple[list[list[str]], set[str]]:
    groups = [
        [str(item) for item in group or []]
        for group in arm.get("coarse_top50") or []
    ]
    return groups, {item for group in groups for item in group}


def _classify_one(record: dict[str, Any], arm: dict[str, Any]) -> str | None:
    arm = arm or {}
    blob = f"{arm.get('error') or ''} {arm.get('clarification') or ''}"
    if arm.get("error") or is_transient_failure(blob):
        return "error"
    if arm.get("clarification") or arm.get("stop_reason") in {"invalid_plan", "tool_error"}:
        return "not_searched"

    gold = {str(item) for item in record.get("gold_chunk_ids") or [] if str(item)}
    metrics = arm.get("metrics") or {}
    recall = _num(metrics.get("recall_at_10", metrics.get("recall_at_5")))

    if filters_exclude(arm.get("hard_filters") or {}, record.get("evidence_meta") or []):
        return "filter_dropped"
    if not _entities_retained(record, arm):
        return "rewrite_drift"
    if str(record.get("stratum") or "") == "vague" and _not_rewritten(record, arm):
        return "vague_unresolved"

    if recall is not None and recall >= 1.0 and "missing" in (arm.get("parent_statuses") or []):
        return "parent_split"
    if record.get("question_type") == "multi_hop" and recall is not None and 0 < recall < 1.0:
        return "cross_doc_partial"

    groups, coarse = _coarse_ids(arm)
    if groups and gold and not (coarse & gold):
        return "miss_coarse"
    if gold and (coarse & gold):
        rerank_ids = {str(item) for item in arm.get("rerank_candidate_chunk_ids") or []}
        return "rerank_demoted" if (coarse & gold) <= rerank_ids else "rerank_cut"
    if recall is not None and recall == 0.0:
        return "other"
    return None


def _retrieval_failed(record: dict[str, Any], arm: dict[str, Any] | None) -> bool:
    """只判断「这一臂有没有拿到全部证据」，不看臂内诊断。

    基线臂往往只有指标、没有 plan/attempted_queries 等运行诊断；拿归因规则
    去跑基线会把「没有诊断」误判成「没改写」，导致所有题都被吞成共享失败。
    """
    if not arm:
        return False
    if arm.get("error"):
        return True
    metrics = arm.get("metrics") or {}
    recall = _num(metrics.get("recall_at_10", metrics.get("recall_at_5")))
    return recall is not None and recall < 1.0


def classify_failure(
    record: dict[str, Any],
    arm: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> str | None:
    """候选臂相对基线臂的失败阶段；两臂同错归 ``shared_failure``。"""
    stage = _classify_one(record, arm)
    if stage is None or stage in {"error", "not_searched"} or not baseline:
        return stage
    if _retrieval_failed(record, baseline):
        return "shared_failure"
    return stage


def stage_counts(stages: Iterable[str | None]) -> dict[str, int]:
    """按 :data:`FAILURE_STAGES` 顺序统计阶段计数（忽略 None）。"""
    counter = Counter(stage for stage in stages if stage)
    ordered = {name: counter[name] for name in FAILURE_STAGES if counter[name]}
    ordered.update({name: count for name, count in counter.items() if name not in ordered})
    return ordered


# ---------------------------------------------------------------------------
# 评分与报告
# ---------------------------------------------------------------------------


def score_retrieval(retrieved: Sequence[str], gold: Sequence[str], k: int = K) -> dict[str, float]:
    """固定分母 Precision@k、Recall@k 与 MRR。"""
    gold_ids = {str(item) for item in gold if str(item)}
    top = [str(item) for item in list(retrieved)[:k] if str(item)]
    hits = [rank for rank, item in enumerate(top, 1) if item in gold_ids]
    if not gold_ids:
        return {"precision_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    hit_ids = {top[rank - 1] for rank in hits}
    return {
        f"precision_at_{k}": round(len(hit_ids) / k, 4),
        f"recall_at_{k}": round(len(hit_ids) / len(gold_ids), 4),
        "mrr": round(1.0 / min(hits), 4) if hits else 0.0,
    }


def precision_ceiling(cases: Sequence[dict[str, Any]], k: int = K) -> float:
    """全部命中时的 Precision@k 上限（每条证据占满一个槽位）。"""
    if not cases:
        return 0.0
    evidence = [len(case.get("evidence") or []) for case in cases]
    return round(sum(evidence) / len(evidence) / k, 4)


def parent_status(quote: str, content: str | None, *, failed: bool = False) -> str:
    """父块能否还原引文；复用 ``metrics.assess_parent_restore`` 的口径。"""
    if failed:
        return "lookup_failed"
    return assess_parent_restore(quote, content)


def parent_restore_summary(statuses: Sequence[str]) -> dict[str, Any]:
    return summarize_parent_restore(list(statuses))


def summarize_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把 ``[{stratum, metrics}]`` 聚合成总体 + 分层 + MRR 区间。"""
    overall = _aggregate(rows)
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(row.get("stratum") or "") for row in rows}):
        if not stratum:
            continue
        by_stratum[stratum] = _aggregate(
            [row for row in rows if str(row.get("stratum") or "") == stratum]
        )
    overall["by_stratum"] = by_stratum
    return overall


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("metrics") or {}).keys())
    means: dict[str, float | None] = {}
    for name in sorted(names):
        values = [
            value
            for row in rows
            if (value := _num((row.get("metrics") or {}).get(name))) is not None
        ]
        means[name] = round(sum(values) / len(values), 4) if values else None
    interval = bootstrap_mean_ci(
        [value for row in rows if (value := _num((row.get("metrics") or {}).get("mrr"))) is not None]
    )
    means["n"] = len(rows)
    means["mrr_ci"] = [round(interval[0], 4), round(interval[1], 4)] if interval else None
    means["mrr_half_width"] = round((interval[1] - interval[0]) / 2, 4) if interval else None
    return means


def judge_mediocre(summary: dict[str, Any]) -> list[str]:
    """结论是否「平庸」：MRR 不稳、区间过宽、Agentic 增益不足。"""
    reasons: list[str] = []
    hybrid = summary.get("hybrid") or {}
    mrr = _num(hybrid.get("mrr"))
    interval = hybrid.get("mrr_ci") or []
    half = _num(hybrid.get("mrr_half_width"))
    if mrr is None:
        reasons.append("MRR 缺失，无法判定检索链是否稳定")
    elif len(interval) == 2 and _num(interval[0]) is not None and interval[0] < 0.5:
        reasons.append(f"MRR 95% 下界 {interval[0]:.2f} 未稳定超过 0.5")
    if half is not None and half > 0.05:
        reasons.append(f"MRR 区间过宽（半宽 {half:.2f} > 0.05），样本量不足")
    standard = summary.get("standard") or {}
    agentic = summary.get("agentic") or {}
    recall_gain = (_num(agentic.get("recall_at_10")) or 0.0) - (_num(standard.get("recall_at_10")) or 0.0)
    precision_gain = (
        (_num(agentic.get("precision_at_10")) or 0.0)
        - (_num(standard.get("precision_at_10")) or 0.0)
    )
    if recall_gain < 0.05 and precision_gain < 0.01:
        reasons.append(
            f"Agentic 增益不足：Recall@10 {recall_gain:+.1%}（< 5pp），"
            f"Precision@10 {precision_gain:+.1%}（< 1pp）"
        )
    return reasons


def mrr_phrase(hybrid: dict[str, Any]) -> str:
    """MRR 结论措辞：区间下界过 0.5 才写「稳定」。"""
    mrr = _num(hybrid.get("mrr"))
    interval = hybrid.get("mrr_ci") or []
    if mrr is None:
        return "MRR 缺失"
    if len(interval) == 2 and _num(interval[0]) is not None:
        low, high = float(interval[0]), float(interval[1])
        if low >= 0.5:
            return f"MRR 稳定为 {mrr:.2f}（95% 区间 {low:.2f}–{high:.2f}）"
        return f"MRR 为 {mrr:.2f}，95% 区间 {low:.2f}–{high:.2f} 下界未过 0.5，不写稳定"
    return f"MRR 为 {mrr:.2f}"


def _pct(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.1%}"


def _score(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.4f}"


def _render_stratum_table(summary: dict[str, Any]) -> list[str]:
    standard = (summary.get("standard") or {}).get("by_stratum") or {}
    agentic = (summary.get("agentic") or {}).get("by_stratum") or {}
    strata = sorted({*standard, *agentic})
    if not strata:
        return []
    lines = [
        "| 分层 | n | 标准 Recall@10 | Agentic Recall@10 | 标准 Precision@10 | Agentic Precision@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stratum in strata:
        left = standard.get(stratum) or {}
        right = agentic.get(stratum) or {}
        n = right.get("n") or left.get("n") or 0
        lines.append(
            f"| {stratum} | {n} | {_pct(left.get('recall_at_10'))} | "
            f"{_pct(right.get('recall_at_10'))} | {_pct(left.get('precision_at_10'))} | "
            f"{_pct(right.get('precision_at_10'))} |"
        )
    return lines


def render_report(summary: dict[str, Any]) -> str:
    """渲染人类可读报告；只有判定平庸时才给「改进建议」。"""
    hybrid = summary.get("hybrid") or {}
    seconds = summary.get("dense") or {}
    standard = summary.get("standard") or {}
    agentic = summary.get("agentic") or {}
    mediocre = list(summary.get("mediocre") or judge_mediocre(summary))
    lines = [
        "# 200 条评测集 A/B 结果",
        "",
        "评测集为机器生成 silver，引文已逐字落在证据块内，不是人工 Gold。",
        "Precision 的分母固定为 10；单证据题即使命中第一也只记 10%。",
        "",
        "## 简历空格",
        "",
        f"- {mrr_phrase(hybrid)}",
        f"- Precision@10 {_pct(seconds.get('precision_at_10'))} -> {_pct(hybrid.get('precision_at_10'))}",
        f"- Recall@10 {_pct(seconds.get('recall_at_10'))} -> {_pct(hybrid.get('recall_at_10'))}",
        "",
        "## 检索链",
        "",
        f"- 纯稠密、不重排：MRR {_score(seconds.get('mrr'))}，"
        f"Precision@10 {_pct(seconds.get('precision_at_10'))}，"
        f"Recall@10 {_pct(seconds.get('recall_at_10'))}",
        f"- 混合 + Reranker：MRR {_score(hybrid.get('mrr'))}，"
        f"Precision@10 {_pct(hybrid.get('precision_at_10'))}，"
        f"Recall@10 {_pct(hybrid.get('recall_at_10'))}",
    ]
    restore = summary.get("parent_restore") or {}
    if restore.get("rate") is not None:
        lines.append(f"- 父块还原率：{restore}")
    lines += [
        "",
        "## Agentic",
        "",
        f"- 标准检索 -> 改写/路由/反思：Precision@10 "
        f"{_pct(standard.get('precision_at_10'))} -> {_pct(agentic.get('precision_at_10'))}，"
        f"Recall@10 {_pct(standard.get('recall_at_10'))} -> {_pct(agentic.get('recall_at_10'))}",
    ]
    stratum_table = _render_stratum_table(summary)
    if stratum_table:
        lines += ["", *stratum_table]
    if mediocre:
        lines += ["", "## 判定", ""] + [f"- {reason}" for reason in mediocre]
        lines += ["", "## 改进建议", ""] + _suggestions(summary)
    stage_counts = summary.get("stage_counts") or {}
    if stage_counts:
        lines += ["", "## 失误阶段", ""]
        lines += [f"- {name}: {count}" for name, count in stage_counts.items()]
    hybrid_counts = summary.get("hybrid_stage_counts") or {}
    if hybrid_counts:
        lines += ["", "混合检索臂相对纯稠密：", ""]
        lines += [f"- {name}: {count}" for name, count in hybrid_counts.items()]
    examples = summary.get("examples") or {}
    if examples:
        lines += ["", "## 例子", ""]
        for stage, items in examples.items():
            lines.append(f"### {stage}")
            lines += [f"- {item}" for item in items]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _suggestions(summary: dict[str, Any]) -> list[str]:
    stages = {
        *(summary.get("stage_counts") or {}).keys(),
        *(summary.get("hybrid_stage_counts") or {}).keys(),
    }
    advice = {
        "filter_dropped": "路由抽出的券商/日期不要直接做硬过滤；过滤后证据为空时应回退无过滤重检。",
        "vague_unresolved": "模糊问句先做实体抽取，把券商、代码、报告期写回检索词再召回。",
        "cross_doc_partial": "跨文档题拆成按来源的子查询，合并时给每一侧保底名额。",
        "miss_coarse": "证据没进粗排前 50：提高稀疏权重，或对含数字/代码的问句强制双路召回。",
        "rerank_demoted": "粗排已命中却被重排挤掉：放宽每篇配额，或按来源保底保留。",
        "rerank_cut": "断崖/截断发生在重排前：检查 cliff 阈值与候选深度。",
        "parent_split": "命中子块但父块还原失败：检查父块边界与引文对齐。",
        "rewrite_drift": "改写丢了原问实体：加实体保留校验，丢失时回退原问句。",
    }
    if not stages:
        return ["- 先看失误阶段：没有可归因样本时，不要调检索参数。"]
    return [f"- {advice[name]}" for name in advice if name in stages] or [
        "- 失败未落入既定阶段，需逐题看粗排名单与改写文本。"
    ]


# ---------------------------------------------------------------------------
# 生成器守卫：配额填充与审计
# ---------------------------------------------------------------------------


def fill_quotas(
    pool: Sequence[dict[str, Any]],
    complete: Callable[[str, list[list[dict[str, Any]]]], list[str]],
    *,
    targets: dict[str, int],
    cap: int,
    batch_size: int = 4,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按分层配额生成条目，并保证引文可定位、来源不超配额。

    ``complete(kind, items)`` 中 ``items`` 的每一项是**一条候选的材料**
    （单跳为 1 个块，跨文档为 2 个不同来源的块），返回等长文本列表。
    """
    import random

    rejections: Counter[str] = Counter(
        {"问题抄录了证据正文": 0, "引文不可定位": 0, "解析失败": 0, "跨文档引文不足": 0}
    )
    rng = random.Random(seed)
    chunks = [chunk for chunk in pool if str(chunk.get("content") or "").strip()]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_source.setdefault(str(chunk.get("source") or ""), []).append(chunk)
    for items in by_source.values():
        rng.shuffle(items)
    sources = sorted(by_source)
    rng.shuffle(sources)
    usage: Counter[str] = Counter()
    used_ids: set[str] = set()

    def source_free(source: str) -> bool:
        return usage[source] < cap and any(
            str(chunk.get("id") or "") not in used_ids for chunk in by_source[source]
        )

    def take_single() -> list[dict[str, Any]] | None:
        for source in sources:
            if not source_free(source):
                continue
            chunk = next(c for c in by_source[source] if str(c.get("id") or "") not in used_ids)
            used_ids.add(str(chunk.get("id") or ""))
            usage[source] += 1
            return [chunk]
        return None

    def take_pair() -> list[dict[str, Any]] | None:
        candidates = [source for source in sources if source_free(source)]
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                left_chunk = next(c for c in by_source[left] if str(c.get("id") or "") not in used_ids)
                right_chunk = next(c for c in by_source[right] if str(c.get("id") or "") not in used_ids)
                for chunk in (left_chunk, right_chunk):
                    used_ids.add(str(chunk.get("id") or ""))
                usage[left] += 1
                usage[right] += 1
                return [left_chunk, right_chunk]
        return None

    cases: list[dict[str, Any]] = []
    sequence = 0
    for kind, target in targets.items():
        for _ in range(int(target)):
            material = take_pair() if kind == "cross_doc" else take_single()
            if material is None:
                break
            sequence += 1
            generated = complete(kind, [material])
            parsed = parse_generation(
                (generated or [""])[0],
                multi=kind == "cross_doc",
            )
            if parsed is None:
                rejections["解析失败"] += 1
                _release(material, usage, used_ids)
                continue
            contents = [str(chunk.get("content") or "") for chunk in material]
            if any(copies_span(parsed["query"], content) for content in contents):
                rejections["问题抄录了证据正文"] += 1
                _release(material, usage, used_ids)
                continue
            evidence, problem = _anchor_evidence(material, parsed["quotes"], kind, sequence)
            if problem:
                rejections[problem] += 1
                _release(material, usage, used_ids)
                continue
            cases.append(
                {
                    "id": f"gen-{kind}-{sequence:03d}",
                    "stratum": kind,
                    "question_type": "multi_hop" if kind == "cross_doc" else "single_hop",
                    "query": parsed["query"],
                    "reference_answer": parsed["answer"],
                    "evidence": evidence,
                    "provenance": {"stratum": kind, "human_approved": False, "builder": "fill_quotas"},
                }
            )
    return cases, dict(rejections)


def _release(
    material: Sequence[dict[str, Any]],
    usage: Counter[str],
    used_ids: set[str],
) -> None:
    for chunk in material:
        used_ids.discard(str(chunk.get("id") or ""))
        usage[str(chunk.get("source") or "")] -= 1


def _anchor_evidence(
    material: Sequence[dict[str, Any]],
    quotes: Sequence[str],
    kind: str,
    sequence: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """把引文钉到具体证据块；跨文档要求两条引文来自不同来源。"""
    evidence: list[dict[str, Any]] = []
    remaining = list(material)
    for index, quote in enumerate(quotes, 1):
        if not remaining:
            break
        matched = None
        for chunk in remaining:
            span = locate_quote(str(chunk.get("content") or ""), quote)
            if span is not None:
                matched = (chunk, span)
                break
        if matched is None:
            return [], "引文不可定位"
        chunk, span = matched
        remaining.remove(chunk)
        evidence.append(
            {
                "chunk_id": str(chunk.get("id") or ""),
                "source": str(chunk.get("source") or ""),
                "quote": span,
                "evidence_id": f"gen-{kind}-{sequence:03d}-e{index}",
            }
        )
    if kind == "cross_doc":
        if len(evidence) < 2 or len({item["source"] for item in evidence}) < 2:
            return [], "跨文档引文不足"
    elif not evidence:
        return [], "引文不可定位"
    return evidence[:2] if kind == "cross_doc" else evidence[:1], None


def audit_cases(
    cases: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    targets: dict[str, int],
    cap: int,
    k: int = K,
) -> list[str]:
    """审计生成结果：配额、去重、引文可定位、来源上限。"""
    issues: list[str] = []
    counts: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        stratum = str(case.get("stratum") or "")
        counts[stratum] += 1
        if not case_id or case_id in seen_ids:
            issues.append(f"{case_id}: 条目 ID 缺失或重复")
        seen_ids.add(case_id)
        evidence = case.get("evidence") or []
        chunks = [by_id.get(str(item.get("chunk_id") or "")) for item in evidence]
        if any(chunk is None for chunk in chunks):
            issues.append(f"{case_id}: 证据块不在知识库")
            continue
        for item, chunk in zip(evidence, chunks, strict=True):
            source = str(chunk.get("source") or "")
            usage[source] += 1
            quote = str(item.get("quote") or "")
            if not quote or quote not in str(chunk.get("content") or ""):
                issues.append(f"{case_id}: 引文不是原文子串")
            if copies_span(str(case.get("query") or ""), str(chunk.get("content") or "")):
                issues.append(f"{case_id}: 问题抄录了证据正文")
        if stratum == "cross_doc":
            if len(evidence) < 2 or len({item.get("source") for item in evidence}) < 2:
                issues.append(f"{case_id}: 跨文档条目未覆盖两个来源")
        elif len(evidence) != 1:
            issues.append(f"{case_id}: 单跳条目应恰好 1 条证据")
    for stratum, target in targets.items():
        if counts[stratum] != int(target):
            issues.append(f"{stratum}: 配额 {counts[stratum]}/{target}")
    for source, count in usage.items():
        if count > cap:
            issues.append(f"{source}: 证据使用 {count} 次超过上限 {cap}")
    if len({str(item.get("chunk_id")) for case in cases for item in case.get("evidence") or []}) != sum(
        len(case.get("evidence") or []) for case in cases
    ):
        issues.append("证据块被多个条目复用")
    return issues
