"""下游检索 API：研报 / 政策两条规范合同。

集合由服务端按路径固定（research → ``finance_report``，policy → ``finance_policy``），
请求体不接收集合名。鉴权使用 ``DATA_API_TOKENS`` 静态 Bearer Token。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import require_data_api_token
from src.api.schemas.search import SearchRequest, SearchResponse
from src.core.exceptions import EmbeddingError, VectorStoreError
from src.retrieve import search_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/research",
    response_model=SearchResponse,
    summary="研报检索",
    description="在 finance_report 集合上执行混合检索（BM25 + 稠密向量 + 重排序）。",
)
async def search_research(
    req: SearchRequest,
    _token: Annotated[str, Depends(require_data_api_token)],
) -> SearchResponse:
    return await _run_search(search_service.search_research, req)


@router.post(
    "/policy",
    response_model=SearchResponse,
    summary="政策检索",
    description=(
        "在 finance_policy 集合上执行混合检索。集合本期为空，"
        "正常返回 200 + 空 items。"
    ),
)
async def search_policy(
    req: SearchRequest,
    _token: Annotated[str, Depends(require_data_api_token)],
) -> SearchResponse:
    return await _run_search(search_service.search_policy, req)


async def _run_search(
    handler: Callable[[SearchRequest], SearchResponse],
    req: SearchRequest,
) -> SearchResponse:
    """在线程池中执行同步检索，并把基础设施故障映射为 503。"""
    try:
        return await asyncio.to_thread(handler, req)
    except (VectorStoreError, EmbeddingError) as exc:
        logger.error("检索 API 基础设施故障：%s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="检索服务暂时不可用，请稍后重试",
        ) from exc
