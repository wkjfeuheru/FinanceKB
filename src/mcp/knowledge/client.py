"""knowledge-api HTTP 客户端（mcp-knowledge 专用）。

只做协议转发，不引入 Milvus / 嵌入 / 重排依赖：模型只由 knowledge-api 进程加载。
错误信息对调用方脱敏，不返回内部 Token 或上游堆栈。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.mcp.common.config import KNOWLEDGE_API_BASE_URL, KNOWLEDGE_API_TOKEN
from src.queries.source_requests import SourceError, execute_async

# 研报检索在 CPU 上要跑 BGE 重排，单次实测 25–70s（随宿主机负载波动）；
# 30s 会稳定超时，可用 KNOWLEDGE_API_TIMEOUT_SECONDS 覆盖。
DEFAULT_TIMEOUT_SECONDS = 120.0


def _default_timeout_seconds() -> float:
    raw = os.getenv("KNOWLEDGE_API_TIMEOUT_SECONDS", "").strip()
    return float(raw) if raw else DEFAULT_TIMEOUT_SECONDS


class KnowledgeAPIError(SourceError):
    """上游 knowledge-api 调用失败（已归类，消息可直接返回给调用方）。"""

    def __init__(self, status_code: int, message: str):
        code = {
            401: "permission_denied",
            403: "permission_denied",
            422: "invalid_argument",
            429: "rate_limited",
            504: "upstream_timeout",
        }.get(status_code, "upstream_unavailable")
        safe_message = {
            401: "知识检索服务内部鉴权失败，请联系管理员",
            403: "知识检索服务内部鉴权失败，请联系管理员",
            422: "检索参数无效，请检查 query / page / page_size / filters",
        }.get(status_code, "知识检索服务暂时不可用，请稍后重试")
        super().__init__(
            safe_message,
            code=code,
            source="knowledge-api",
            retryable=status_code in (429, 503, 504),
        )
        self.status_code = status_code
        self.message = message


class KnowledgeAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or KNOWLEDGE_API_BASE_URL).rstrip("/")
        self._token = token if token is not None else KNOWLEDGE_API_TOKEN
        self._timeout = timeout if timeout is not None else _default_timeout_seconds()

    async def search(self, scope: str, payload: dict[str, Any]) -> dict[str, Any]:
        """调用 ``/api/v1/search/{scope}``，返回上游 JSON。"""
        url = f"{self._base_url}/api/v1/search/{scope}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:

            async def request(timeout: float) -> httpx.Response:
                response = await client.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response

            try:
                response = await execute_async(
                    request, source="knowledge-api", timeout=self._timeout
                )
            except SourceError as exc:
                status = {
                    "permission_denied": 401,
                    "invalid_argument": 422,
                    "rate_limited": 429,
                    "upstream_timeout": 504,
                }.get(exc.code, 503)
                error = KnowledgeAPIError(status, "知识检索上游请求失败")
                error.retryable = exc.retryable
                error.attempt_count = exc.attempt_count
                raise error from None

        try:
            payload = response.json()
        except ValueError as exc:
            raise KnowledgeAPIError(502, "knowledge api returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise KnowledgeAPIError(502, "knowledge api returned an unexpected payload")
        return payload


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "KnowledgeAPIClient",
    "KnowledgeAPIError",
]
