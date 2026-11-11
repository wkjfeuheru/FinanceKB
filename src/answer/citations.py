"""引用编号账本、声明结构解析和来源定位；旧入口保留兼容转发。"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

import numpy as np

from src.answer.faithfulness import _detail_mismatch as _detail_mismatch
from src.answer.models import (
    ClaimCheck,
    _validation_result,
)
from src.answer.models import (
    ClaimReport as ClaimReport,
)
from src.answer.models import (
    SupportingQuote as SupportingQuote,
)
from src.core.config import (
    CITATION_SIMILARITY_THRESHOLD,
    ENABLE_CITATION_VALIDATION,
    ENABLE_REFUSAL,
    LOW_CONFIDENCE_THRESHOLD,
    REFUSAL_MIN_RERANK_SCORE,
    REFUSAL_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 拒答固定文案（检索为空 / 引用校验失败时使用）
REFUSAL_ANSWER = (
    "抱歉，我无法给出可靠回答：检索到的文档内容不足以支撑该问题，请补充相关文档或换个问法。"
)


def normalize_citations(answer: str) -> str:
    """统一分组引用为独立编号；不删除任何编号，越界引用仍由账本拒绝。"""
    return re.sub(
        r"\[\s*\d+(?:\s*[,，、]\s*\d+)*\s*\]",
        lambda match: "".join(f"[{int(value)}]" for value in re.findall(r"\d+", match[0])),
        answer,
    )


class CitationLedger:
    """本次上下文的固定编号快照；来源、提示词和核验共用同一顺序。"""

    def __init__(self, documents: list[dict]):
        self._documents = deepcopy(documents)

    @property
    def documents(self) -> list[dict]:
        return deepcopy(self._documents)

    @property
    def entries(self) -> list[tuple[int, dict]]:
        return list(enumerate(self.documents, 1))

    def matches_sources(self, sources: list[dict]) -> bool:
        if len(sources) != len(self._documents):
            return False
        for index, (doc, source) in enumerate(zip(self._documents, sources), 1):
            if source.get("index") != index:
                return False
            for key in ("id", "chunk_id", "document_version", "source"):
                if key in doc and key in source and doc[key] != source[key]:
                    return False
        return True

    def validate(self, answer: str) -> dict:
        answer = normalize_citations(answer)
        numbers = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        known = set(range(1, len(self._documents) + 1))
        dangling = sorted(numbers - known)
        claims = _answer_claims(answer)
        missing = [item["claim"] for item in claims if not item["citation_indices"]]
        valid = bool(claims) and not dangling and not missing
        return {
            "status": "verified" if valid else "failed",
            "used": sorted(numbers & known),
            "unused": sorted(known - numbers),
            "dangling": dangling,
            "missing": missing,
        }


class CitationValidator:
    """验证回答中的 [1][2] 引用是否与来源内容匹配。

    使用嵌入语义相似度检查引用上下文是否确实存在于来源文档中。
    """

    def __init__(self, embed_fn):
        """初始化引用验证器。

        Args:
            embed_fn: 嵌入函数，接受文本返回 float 列表。
        """
        self._embed = embed_fn

    def validate(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """验证回答中的引用标记。

        Args:
            answer: LLM 生成的回答文本。
            sources: 检索到的来源文档列表（含 content 字段）。

        Returns:
            {"valid": bool, "issues": list[str], "score": float}
        """
        citations = list(re.finditer(r"\[(\d+)\]", answer))

        if not citations:
            return _validation_result("failed", [], ["事实回答缺少引用"], 0)

        issues: list[str] = []
        checked = 0

        for match in citations:
            ref_num = int(match.group(1))
            context = self._get_citation_context(answer, match.start())

            if ref_num < 1:
                issues.append(f"引用[{ref_num}]超出范围（引用编号必须从1开始）")
                continue
            if ref_num > len(sources):
                issues.append(f"引用[{ref_num}]超出范围（仅有{len(sources)}个来源）")
                continue

            source = sources[ref_num - 1]
            source_content = source.get("content", "")

            if not source_content:
                issues.append(f"引用[{ref_num}]的来源内容为空")
                continue

            if self._verify_in_source(context, source_content):
                checked += 1
            else:
                issues.append(f"引用[{ref_num}]的内容与来源不匹配")

        total = len(citations)
        score = checked / total if total > 0 else 1.0

        return {
            "valid": len(issues) == 0,
            "status": "verified" if not issues else "failed",
            "issues": issues,
            "score": round(score, 4),
            "total_citations": total,
            "verified_citations": checked,
            "checks": [],
        }

    @staticmethod
    def _get_citation_context(answer: str, pos: int, window: int = 100) -> str:
        """获取引用标记周围的上下文文本。"""
        start = max(0, pos - window)
        end = min(len(answer), pos + window)
        return answer[start:end]

    def _verify_in_source(self, claim: str, source_content: str) -> bool:
        """检查声明是否存在于来源中（基于嵌入语义相似度）。

        Args:
            claim: 引用上下文文本。
            source_content: 来源文档内容。

        Returns:
            相似度是否超过阈值。
        """
        try:
            claim_emb = np.array(self._embed(claim))
            source_emb = np.array(self._embed(source_content[:500]))
            similarity = float(
                np.dot(claim_emb, source_emb)
                / (np.linalg.norm(claim_emb) * np.linalg.norm(source_emb) + 1e-8)
            )
            return similarity >= CITATION_SIMILARITY_THRESHOLD
        except Exception as exc:
            logger.warning("引用验证嵌入计算失败：%s", exc)
            return False


def run_citation_validation(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """执行旧嵌入引用验证；禁用或异常返回 unknown，无引用/来源返回 failed。"""
    total = len(re.findall(r"\[(\d+)\]", answer))
    if not ENABLE_CITATION_VALIDATION:
        return _validation_result("unknown", [], ["引用校验未启用"], total)
    if not sources or not total:
        return _validation_result("failed", [], ["事实回答缺少引用或来源"], total)
    try:
        from src.index.milvus_kb import get_knowledge_base

        kb = get_knowledge_base()
        embed_fn = kb.get_embeddings().embed_query
        return CitationValidator(embed_fn).validate(answer, sources)
    except Exception as exc:
        logger.warning("引用验证失败：%s", exc)
        return _validation_result("unknown", [], ["引用校验不可用"], total)


# Markdown 列表标记：只匹配「行首 + 编号/项目符号 + 空白」，因此年份与金额
# （``2025年营业收入``、``1.5亿元``）不会被误删（RAG-12）。
_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*(?:\d{1,3}[.)][ \t]+|[-*+•·][ \t]+|\d{1,3}、[ \t]*)")
_NAMED_ITEM_RE = re.compile(r"^[ \t]*\*\*[^*\n]{1,40}\*\*[ \t]*[:：]")

# 来源归属：报告日期与页码。它们必须对**来源元数据**核验，而不是要求出现在
# 原文引文里（报告日期通常只存在于元数据中，RAG-13）。
_ATTRIBUTION_DATE_RE = re.compile(
    r"(?:报告日期|发布日期|发布时间|报告时间)\s*[:：]?\s*(\d{4}-\d{2}-\d{2})"
    r"|(?:[（(\[【]\s*)(\d{4}-\d{2}-\d{2})(?=\s*[）)\]】])"
)
_ATTRIBUTION_PAGE_RE = re.compile(r"第\s*(\d{1,4})\s*(?:[-–~]\s*(\d{1,4})\s*)?页")


def _strip_list_markers(text: str) -> str:
    """剥离确认为列表标记的行首前缀（不改动正文中的数字）。"""
    return _LIST_MARKER_RE.sub("", text or "")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+")


def _split_attributions(claim: str) -> tuple[str, list[tuple[str, Any]]]:
    """把声明拆成「正文事实」与「来源归属」两部分。

    归属信息（``[1]（2026-08-15）``、``第 4 页``）走来源元数据核验；
    其余正文（金额、单位、主体、期间、增长方向）原样留给数值与实体检查，
    因此增加日期支持不会放宽这些检查。

    只有明确写成「报告日期：…」或括注的日期才算归属：正文里的裸日期
    （``2026年8月15日发布的中报``）仍按事实中的期间处理。
    """
    text = claim or ""
    attributions: list[tuple[str, Any]] = []
    for match in _ATTRIBUTION_DATE_RE.finditer(text):
        value = match.group(1) or match.group(2)
        if value:
            attributions.append(("date", value))
    for match in _ATTRIBUTION_PAGE_RE.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        attributions.append(("page", (start, end)))
    fact = _ATTRIBUTION_DATE_RE.sub(" ", text)
    fact = _ATTRIBUTION_PAGE_RE.sub(" ", fact)
    return fact, attributions


def _attribution_mismatch(
    attributions: list[tuple[str, Any]],
    quotes: list[tuple[int, str]],
    docs: list[dict],
) -> bool:
    """归属信息必须与**被引用来源**的真实元数据一致（RAG-13）。"""
    cited = [docs[index - 1] for index, _ in quotes]
    for kind, value in attributions:
        if kind == "date":
            if not any(str(doc.get("date") or "").strip() == value for doc in cited):
                return True
            continue
        start, end = value
        if not any(_page_within(doc, start, end) for doc in cited):
            return True
    return False


def _page_within(doc: dict, start: int, end: int) -> bool:
    """页码是否落在该来源的页码区间内；元数据缺失视为无法证实。"""
    doc_start = int(doc.get("start_page") or 0)
    doc_end = int(doc.get("end_page") or doc_start or 0)
    if doc_start <= 0:
        return False
    if doc_end <= 0:
        doc_end = doc_start
    if start > end:
        start, end = end, start
    return doc_start <= start and end <= doc_end


def _consume_claim_part(part: str, claims: list[dict], *, inherit: bool, is_list: bool) -> None:
    """把一个句段写入声明列表。

    ``inherit`` 只用于紧挨着上一条、且自身没有引用的列表项。
    """
    part = part.strip()
    if not part:
        return
    indices = [int(number) for number in re.findall(r"\[(\d+)\]", part)]
    text = re.sub(r"\[\d+\]", "", part).strip().strip("。！？!?；;. ")
    if not text:
        if claims and indices:
            claims[-1]["citation_indices"].extend(indices)
        return
    # 句号之后的引用属于前一句，不属于其后开始的新句。
    prefix = re.match(r"^(?:\[\d+\]\s*)+", part)
    if prefix and claims:
        previous = [int(number) for number in re.findall(r"\[(\d+)\]", prefix[0])]
        claims[-1]["citation_indices"].extend(previous)
        indices = indices[len(previous) :]
    if inherit and not indices and claims and claims[-1]["citation_indices"]:
        indices = list(claims[-1]["citation_indices"])
    claims.append({"claim": text, "citation_indices": indices, "is_list": is_list})


def _inherit_lead_in_citations(claims: list[dict]) -> None:
    """以冒号结尾、自身没有引用的引导句，沿用紧随列表项上的引用。

    模型常把编号写在列表项末尾，并在引导句和列表之间空一行。
    引导句仍要进入核验，只是不再因为缺少重复编号被整篇拒答。
    不以冒号结尾的普通句子不继承。
    """
    for index, claim in enumerate(claims):
        if claim["citation_indices"] or claim.get("is_list"):
            continue
        if not str(claim["claim"]).endswith(("：", ":")):
            continue
        following: list[dict] = []
        cursor = index + 1
        while cursor < len(claims) and claims[cursor].get("is_list"):
            following.append(claims[cursor])
            cursor += 1
        if not following or any(not item["citation_indices"] for item in following):
            continue
        inherited: list[int] = []
        for item in following:
            for number in item["citation_indices"]:
                if number not in inherited:
                    inherited.append(number)
        claim["citation_indices"] = inherited


def _markdown_cells(line: str) -> list[str] | None:
    line = line.strip()
    if "|" not in line:
        return None
    cells = re.split(r"(?<!\\)\|", line)
    if not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [cell.strip() for cell in cells] or None


def _answer_claims(answer: str) -> list[dict]:
    """保守地逐句覆盖全文；无法确认无须核验的正文仍进入门禁。

    引用可位于句号前后。句段完整性检查阻止模型只核验易通过的声明，
    但不构成对自然语言蕴含的数学证明。

    先剥离**确认为 Markdown 列表标记**的前缀再切分（RAG-12）：
    ``1. 营业收入100亿元[1]。`` 曾被切成无引用声明 ``1`` 与正常事实，
    导致核验失败、重试仍失败、最终拒答。剥离只匹配「行首 + 编号/项目符号 +
    空白」，因此 ``2025年营业收入…``、``1.5亿元…`` 这类以数字开头的事实不受影响。

    引导句已经带引用时，紧随的无编号列表项继承该引用。空行会断开这条向前继承；
    普通句子即使跟在已引用声明后面，仍然必须自带引用。

    反过来，引用写在列表项上、引导句只以冒号收尾时，引导句沿用这些列表项的编号。
    编号项下面缩进更深的子项，共用该编号项中出现过的引用。
    """
    answer = normalize_citations(answer or "")
    claims: list[dict] = []
    gap = False
    lines = (answer or "").splitlines() or [""]
    headers = None
    separator_line = -1
    for line_number, raw_line in enumerate(lines):
        if line_number == separator_line:
            continue
        if not raw_line.strip():
            gap = True
            headers = None
            continue
        cells = _markdown_cells(raw_line)
        next_cells = _markdown_cells(lines[line_number + 1]) if line_number + 1 < len(lines) else None
        if cells and next_cells and len(cells) == len(next_cells) and all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in next_cells
        ):
            headers = [re.sub(r"\[\d+\]", "", cell).strip() for cell in cells]
            separator_line = line_number + 1
            continue
        if headers is not None and cells and len(cells) == len(headers):
            # 表头是每行的字段标签，不能单独当成无引用事实；每行独立携带引用。
            row = "；".join(f"{header}：{cell}" for header, cell in zip(headers, cells, strict=True))
            _consume_claim_part(row, claims, inherit=False, is_list=True)
            claims[-1].update(indent=0, gap_before=gap)
            gap = False
            continue
        headers = None
        is_list = (
            _LIST_MARKER_RE.match(raw_line) is not None
            or _NAMED_ITEM_RE.match(raw_line) is not None
        )
        line = _LIST_MARKER_RE.sub("", raw_line, count=1) if is_list else raw_line
        blocked = gap
        gap = False
        start = len(claims)
        line_indent = len(raw_line) - len(raw_line.lstrip(" \t"))
        previous_list_indent = next(
            (item.get("indent") for item in reversed(claims) if item.get("is_list")),
            None,
        )
        # 从更深的子项回到上一级时，不能把子项的引用带给下一个编号项。
        leaving_nested = previous_list_indent is not None and line_indent < int(
            previous_list_indent
        )
        for index, part in enumerate(_SENTENCE_SPLIT_RE.split(line)):
            _consume_claim_part(
                part,
                claims,
                inherit=is_list and (index > 0 or (not blocked and not leaving_nested)),
                is_list=is_list,
            )
        if start < len(claims):
            indent = line_indent
            claims[start]["indent"] = indent
            claims[start]["gap_before"] = blocked
            for claim in claims[start + 1 :]:
                claim["indent"] = indent
                claim["gap_before"] = False
        if is_list:
            pooled: list[int] = []
            for claim in claims[start:]:
                for number in claim["citation_indices"]:
                    if number not in pooled:
                        pooled.append(number)
            if pooled:
                for claim in claims[start:]:
                    if not claim["citation_indices"]:
                        claim["citation_indices"] = list(pooled)
    _share_nested_list_citations(claims)
    _inherit_lead_in_citations(claims)
    for claim in claims:
        claim.pop("is_list", None)
        claim.pop("indent", None)
        claim.pop("gap_before", None)
    return claims


def _share_nested_list_citations(claims: list[dict]) -> None:
    """编号项下的缩进子项共用该编号项里出现的引用。

    模型常把 ``[n]`` 只写在最后一条子项上，标题和「核心产品」行本身没有编号。
    同一缩进的平级列表项不互相借用；空行会断开编号项。没有出现任何编号的
    列表项保持无引用。
    """
    index = 0
    while index < len(claims):
        claim = claims[index]
        if not claim.get("is_list"):
            index += 1
            continue
        end = index + 1
        base = int(claim.get("indent") or 0)
        while end < len(claims) and claims[end].get("is_list"):
            if claims[end].get("gap_before") or int(claims[end].get("indent") or 0) <= base:
                break
            end += 1
        pooled: list[int] = []
        for item in claims[index:end]:
            for number in item["citation_indices"]:
                if number not in pooled:
                    pooled.append(number)
        if pooled:
            for item in claims[index:end]:
                if not item["citation_indices"]:
                    item["citation_indices"] = list(pooled)
        index = end


_QUOTE_BREAK_RE = re.compile(r"(?:…{2,}|\.{3,}|[\r\n]+)")
_QUOTE_SEGMENT_GAP = 160


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _segment_needles(segment: str) -> list[str]:
    """短标签加冒号是表格转写，正文里通常只有冒号后面的单元格。"""
    needles = [segment.strip()]
    for separator in ("：", ":"):
        if separator not in segment:
            continue
        label, rest = segment.split(separator, 1)
        if rest.strip() and 0 < len(label.strip()) <= 12:
            needles.append(rest.strip())
        break
    return [needle for needle in (_compact_text(item) for item in needles) if len(needle) >= 8]


def _segments_in_content(parts: list[str], compact: str, start: int, *, first: bool) -> bool:
    if not parts:
        return True
    search_from = 0 if first else start
    for needle in _segment_needles(parts[0]):
        cursor = search_from
        while True:
            pos = compact.find(needle, cursor)
            if pos < 0:
                break
            if not first and pos - start > _QUOTE_SEGMENT_GAP:
                break
            if _segments_in_content(parts[1:], compact, pos + len(needle), first=False):
                return True
            cursor = pos + 1
    return False


def _quote_in_source(
    index: int, quote: str, docs: list[dict], *, visual_indices: set[int] | None = None
) -> bool:
    """文本来源的引文必须逐字出现；图片来源以原图为准，不要求出现在图注里。

    表格转写可以用换行把同一行的单元格连起来，省略号两边也必须按原文顺序出现。
    只忽略空白和不超过 12 个字的「标签：」前缀，不接受改写。
    """
    if not quote or index < 1 or index > len(docs):
        return False
    doc = docs[index - 1]
    content = doc.get("content")
    if not isinstance(content, str):
        return False
    if (
        index in (visual_indices or set())
        and str(doc.get("block_type") or "") == "image"
        and str(doc.get("image_key") or "").strip()
    ):
        return True
    if quote in content:
        return True
    parts = [part.strip() for part in _QUOTE_BREAK_RE.split(quote) if part.strip()]
    return bool(parts) and _segments_in_content(parts, _compact_text(content), 0, first=True)


def apply_refusal_policy(
    validation: dict[str, Any] | None,
    source_count: int,
    *,
    rerank_scores: list[float] | None = None,
) -> tuple[bool, bool]:
    """根据引用验证结果与来源相关性决定拒答/低置信标记。

    Returns:
        ``(answer_rejected, low_confidence)``。

        * 检索为空（source_count == 0）→ 拒答；
        * 相关性不足（全部来源 rerank_score < REFUSAL_MIN_RERANK_SCORE）→ 拒答；
        * 存在引用且校验分数 < REFUSAL_SCORE_THRESHOLD → 拒答；
        * 分数介于 [REFUSAL_SCORE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD) → 低置信；
        * 其余（含未开启拒答 / 未做验证 / 无引用）→ 正常返回。
    """
    if not ENABLE_REFUSAL:
        return False, False
    if source_count == 0:
        return True, False
    # 相关性守卫：重排开启时，全部来源相关性过低视为检索不足
    if rerank_scores is not None and len(rerank_scores) > 0:
        if max(rerank_scores) < REFUSAL_MIN_RERANK_SCORE:
            return True, False
    if validation is None:
        return False, False
    if validation.get("status") in {"unknown", "failed"}:
        return True, False
    total = int(validation.get("total_citations", 0) or 0)
    if total == 0:
        return False, False
    score = float(validation.get("score", 1.0))
    if score < REFUSAL_SCORE_THRESHOLD:
        return True, False
    if score < LOW_CONFIDENCE_THRESHOLD:
        return False, True
    return False, False


async def validate_claims(answer: str, context_docs: list[dict], **options) -> dict:
    """兼容旧调用入口；所有语义核验由 claims 模块执行。"""
    from src.answer.claims import validate_claims as validate

    if options.get("enabled") is None:
        options["enabled"] = ENABLE_CITATION_VALIDATION
    return await validate(answer, context_docs, **options)


def _check_quotes(check: ClaimCheck, docs: list[dict]) -> bool:
    from src.answer.claims import _check_quotes as check_quotes

    return check_quotes(check, docs)


def _restriction_issue(check: ClaimCheck, allowed, blocked):
    from src.answer.claims import _restriction_issue as restriction_issue

    return restriction_issue(check, allowed, blocked)
