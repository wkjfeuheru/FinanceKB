"""无外部依赖的统一评测指标实现。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: 生成侧统一使用的 RAGAS 四指标：检索链路两项（噪音 / 知识盲区）+ 生成链路两项
#: （防幻觉 / 防跑偏）。放在本模块（而非 ``ragas_eval``）是为了让不依赖 ragas
#: 的评测模块——锚点样本、人工标注校准——共用同一份名字，避免多处副本漂移。
GENERATION_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
METRICS_VERSION = "2.0"


def evidence_hit_rank(ref: Any, docs: list[dict[str, Any]]) -> int | None:
    """同源 ID/父块覆盖或实际完整引文均可证明证据命中。"""
    for rank, doc in enumerate(docs, 1):
        if str(doc.get("source") or "") != ref.source:
            continue
        ids = {str(doc.get(key) or "") for key in ("id", "chunk_id", "pk")}
        ids.update(str(item) for item in doc.get("covered_child_ids", []) or [])
        if ref.chunk_id and ref.chunk_id in ids:
            return rank
        content = str(doc.get("content") or doc.get("text") or "")
        if ref.quote and compute_text_coverage([content], [ref.quote]) == 1.0:
            return rank
    return None


def fact_completion_ranks(case: Any, docs: list[dict[str, Any]]) -> dict[str, int | None]:
    refs = {ref.evidence_id: ref for ref in case.evidence}
    ranks = {key: evidence_hit_rank(ref, docs) for key, ref in refs.items()}
    result = {}
    for fact in case.facts:
        completed = []
        for group in fact.acceptable_evidence_sets:
            hits = [ranks.get(key) for key in group]
            if hits and all(hit is not None for hit in hits):
                completed.append(max(hits))
        result[fact.id] = min(completed) if completed else None
    return result


def compute_structured_fact_metrics(case: Any, docs: list[dict[str, Any]], k: int = 5) -> dict[str, float | None]:
    """按标注组合计分，alias 的 AND 必须在单份同源正文中成立。"""
    names = ("evidence_group_recall_at_5", "fact_mrr", "structured_required_fact_coverage_at_5", "all_required_documents_at_5")
    required_sources = {ref.source for ref in case.evidence if ref.source}
    actual_sources = {str(doc.get("source") or "") for doc in docs[:k]}
    all_documents = float(required_sources <= actual_sources) if required_sources else None
    if not case.facts:
        return {**dict.fromkeys(names), names[3]: all_documents}
    top = docs[:k]
    ranks = fact_completion_ranks(case, docs)
    coverage = []
    for fact in case.facts:
        coverage.append(any(
            str(doc.get("source") or "") in fact.allowed_sources
            and any(group and all(compute_text_coverage([str(doc.get("content") or doc.get("text") or "")], [phrase]) == 1.0 for phrase in group) for group in fact.aliases)
            for doc in top
        ))
    return {
        names[0]: round(sum(rank is not None and rank <= k for rank in ranks.values()) / len(case.facts), 4),
        names[1]: round(sum(1.0 / rank if rank else 0.0 for rank in ranks.values()) / len(case.facts), 4),
        names[2]: round(sum(coverage) / len(case.facts), 4),
        names[3]: all_documents,
    }

# 与 HybridRetriever._expand_to_parents 的替换区间一致：过短或过长的父块
# 不会替换子块正文，因此不能算作「父块还原」。
PARENT_REPLACE_MIN_CHARS = 50
PARENT_REPLACE_MAX_CHARS = 3000


def compute_document_recall(
    retrieved_sources: list[str], evidence_sources: list[str], k: int = 5,
) -> float | None:
    """按唯一 source 计算文档召回；同篇多个块只命中一次。"""
    evidence = {source for source in evidence_sources if source}
    if not evidence:
        return None
    return round(len(set(retrieved_sources[:k]) & evidence) / len(evidence), 4)


def _normalize_coverage_text(text: str) -> str:
    # NFKC 对齐全角数字/百分号；忽略空白与 Unicode 标点，保留数值符号。
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?<=[A-Za-z])[^\w%‰+\-−]+(?=[A-Za-z])", " ", normalized)
    chars: list[str] = []
    for index, char in enumerate(normalized):
        # 保留英文单词之间的分隔，避免去空白后将独立词合并成一个 token。
        if char.isspace() and index > 0 and index + 1 < len(normalized) and normalized[index - 1].isascii() and normalized[index - 1].isalpha() and normalized[index + 1].isascii() and normalized[index + 1].isalpha():
            chars.append(" ")
        numeric_decimal = char == "." and index > 0 and index + 1 < len(normalized) and normalized[index - 1].isdigit() and normalized[index + 1].isdigit()
        if char in "%‰+-−" or numeric_decimal or (not char.isspace() and not unicodedata.category(char).startswith("P")):
            chars.append(char)
    return "".join(chars)


def compute_text_coverage(contents: list[str], targets: list[str]) -> float | None:
    """确定性子串覆盖率；空标注不进分母，块间拼接不能制造命中。"""
    normalized_targets = [_normalize_coverage_text(text) for text in targets]
    normalized_targets = [text for text in normalized_targets if text]
    if not normalized_targets:
        return None
    normalized_contents = [_normalize_coverage_text(text) for text in contents]
    hits = sum(any(_contains_coverage_target(content, target) for content in normalized_contents) for target in normalized_targets)
    return round(hits / len(normalized_targets), 4)


def _contains_coverage_target(content: str, target: str) -> bool:
    """中文允许短语子串；数字与 ASCII token 不能从另一个 token 内截取。"""
    def ascii_word(char: str) -> bool:
        return char.isascii() and (char.isalnum() or char == "_")

    offset = 0
    while (start := content.find(target, offset)) >= 0:
        end = start + len(target)
        before = content[start - 1] if start else ""
        after = content[end] if end < len(content) else ""
        partial_left = (
            (ascii_word(target[0]) and ascii_word(before))
            or (target[0].isdigit() and before in ".+-−" and bool(before))
            or (target[0] in "+-−" and before.isdigit())
        )
        partial_right = (
            (ascii_word(target[-1]) and ascii_word(after))
            or (target[-1].isdigit() and after in ".%‰" and bool(after))
        )
        if not partial_left and not partial_right:
            return True
        offset = start + 1
    return False


def compute_chunk_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    evidence_chunk_ids: list[str] | tuple[str, ...] | None,
    k: int = 5,
) -> dict[str, float | None]:
    """计算块级 Recall@k、固定分母 Precision@k 和 MRR。"""
    evidence = {str(chunk_id) for chunk_id in (evidence_chunk_ids or []) if chunk_id}
    if not evidence:
        return {"recall_at_k": None, "precision_at_k": None, "mrr": None}
    top_k = [str(chunk_id) for chunk_id in retrieved_chunk_ids[:k] if chunk_id]
    hit_ranks = [rank for rank, chunk_id in enumerate(top_k, 1) if chunk_id in evidence]
    hit_ids = {top_k[rank - 1] for rank in hit_ranks}
    return {
        "recall_at_k": round(len(hit_ids) / len(evidence), 4),
        "precision_at_k": round(len(hit_ids) / k, 4),
        "mrr": round(1.0 / min(hit_ranks), 4) if hit_ranks else 0.0,
    }


def assess_parent_restore(quote: str, parent_content: str | None) -> str:
    """子块已经命中后，判断 PostgreSQL 父块能否还原这条引文。

    - ``restored``：父块长度落在生产替换区间，且引文是父块全文的子串
    - ``not_replaced``：父块过短或过长，生产逻辑会保留子块原文
    - ``missing``：父块会被替换，但引文不在全文里
    - ``no_parent``：没有父块内容
    """
    if parent_content is None:
        return "no_parent"
    if not (PARENT_REPLACE_MIN_CHARS <= len(parent_content) <= PARENT_REPLACE_MAX_CHARS):
        return "not_replaced"
    if quote and quote in parent_content:
        return "restored"
    return "missing"


def summarize_parent_restore(statuses: list[str]) -> dict[str, float | int | None]:
    """按命中证据块计 ``parent_restore_rate``，未替换次数单独列出。"""
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    hits = len(statuses)
    restored = counts.get("restored", 0)
    return {
        "hits": hits,
        "restored": restored,
        "not_replaced": counts.get("not_replaced", 0),
        "missing": counts.get("missing", 0),
        "no_parent": counts.get("no_parent", 0),
        "lookup_failed": counts.get("lookup_failed", 0),
        "rate": round(restored / hits, 4) if hits else None,
    }
