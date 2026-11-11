"""检索服务层：把下游请求合同映射到固定集合的混合检索。

调用方只能选择 ``research`` / ``policy`` 两种 scope，集合名由服务端常量决定，
请求体无法指定任意 Milvus 集合。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal

from src.api.schemas.search import SearchHit, SearchRequest, SearchResponse
from src.core.config import POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
from src.index.milvus_kb import get_knowledge_base

logger = logging.getLogger(__name__)

SearchScope = Literal["research", "policy"]

SEARCH_SCOPES: MappingProxyType[str, str] = MappingProxyType(
    {
        "research": REPORT_COLLECTION_NAME,
        "policy": POLICY_COLLECTION_NAME,
    }
)

#: 深分页上限：候选池最多 200 条，与 ``page * page_size <= 200`` 的请求约束对齐。
MAX_CANDIDATE_CHUNKS = 200


def search_research(req: SearchRequest) -> SearchResponse:
    """检索研报集合。"""
    return run_search("research", req)


def search_policy(req: SearchRequest) -> SearchResponse:
    """检索政策集合（本期集合为空，返回正常的空结果）。"""
    return run_search("policy", req)


def run_search(scope: SearchScope, req: SearchRequest) -> SearchResponse:
    """按 scope 检索并切分页窗口。"""
    collection = SEARCH_SCOPES[scope]
    as_of = _utc_now()
    kb = get_knowledge_base(collection)
    # 空集合（如本期 finance_policy）直接返回空结果：不建集合、不加载嵌入/重排模型。
    if not kb.has_rows():
        return _empty_response(collection, req, as_of)

    candidate_k = min(MAX_CANDIDATE_CHUNKS, req.page * req.page_size * 2)
    matches = kb.hybrid_search(
        req.query,
        k=candidate_k,
        use_rerank=True,
        rerank_top_n=candidate_k,
        filters=req.filters.to_milvus(),
    )

    start = (req.page - 1) * req.page_size
    window = matches[start:start + req.page_size]
    has_more = start + len(window) < len(matches)
    return SearchResponse(
        collection=collection,
        as_of=as_of,
        page=req.page,
        page_size=req.page_size,
        returned=len(window),
        has_more=has_more,
        next_page=req.page + 1 if has_more else None,
        items=[_to_hit(match) for match in window],
    )


def _empty_response(
    collection: str, req: SearchRequest, as_of: str
) -> SearchResponse:
    return SearchResponse(
        collection=collection,
        as_of=as_of,
        page=req.page,
        page_size=req.page_size,
        returned=0,
        has_more=False,
        next_page=None,
        items=[],
    )


def _to_hit(match: dict[str, Any]) -> SearchHit:
    rerank_score = match.get("rerank_score")
    return SearchHit(
        id=str(match.get("id") or ""),
        source=str(match.get("source") or ""),
        title=str(match.get("title") or "金融文档"),
        content=str(match.get("content") or ""),
        score=float(match.get("score") or 0.0),
        rerank_score=float(rerank_score) if rerank_score is not None else None,
        chunk=int(match.get("chunk") or 0),
        block_type=str(match.get("block_type") or "text"),
        start_page=int(match.get("start_page") or 0),
        end_page=int(match.get("end_page") or 0),
        heading_path=str(match.get("heading_path") or ""),
        date=str(match.get("date") or ""),
        report_type=str(match.get("report_type") or ""),
        broker=str(match.get("broker") or ""),
        ingested_at=_epoch_to_iso(match.get("ingested_at")),
    )


def _utc_now() -> str:
    return (
        datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _epoch_to_iso(value: Any) -> str | None:
    try:
        timestamp = int(value or 0)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    return (
        datetime.fromtimestamp(timestamp, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "MAX_CANDIDATE_CHUNKS",
    "SEARCH_SCOPES",
    "SearchScope",
    "run_search",
    "search_policy",
    "search_research",
]
