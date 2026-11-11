"""MCP 服务静态 Bearer 鉴权中间件。

``/healthz`` 放行（供容器健康检查）；其余路径必须携带与 ``MCP_API_TOKENS``
匹配的 Bearer Token。中间件只做鉴权，不修改请求路径，Caddy 可原样反代。
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        {"detail": detail},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


class BearerAuthMiddleware:
    """纯 ASGI 中间件：保持流式响应与 lifespan 透传。"""

    def __init__(
        self,
        app: ASGIApp,
        tokens: Iterable[str],
        *,
        health_paths: Iterable[str] = ("/healthz",),
    ) -> None:
        self.app = app
        self.tokens = tuple(tokens)
        self.health_paths = frozenset(health_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") in self.health_paths:
            await JSONResponse({"status": "ok"})(scope, receive, send)
            return

        if not self.tokens:
            logger.error("MCP_API_TOKENS 未配置，拒绝所有 MCP 请求")
            await _unauthorized("MCP authentication is not configured")(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")
        scheme, _, credentials = authorization.partition(" ")
        credentials = credentials.strip()
        if scheme.lower() != "bearer" or not credentials:
            await _unauthorized("Not authenticated")(scope, receive, send)
            return
        if not credentials.isascii():
            await _unauthorized("Invalid token")(scope, receive, send)
            return

        candidate = credentials.encode("ascii")
        matched = False
        for expected in self.tokens:
            matched |= secrets.compare_digest(candidate, expected.encode("utf-8"))
        if not matched:
            await _unauthorized("Invalid token")(scope, receive, send)
            return

        await self.app(scope, receive, send)


__all__ = ["BearerAuthMiddleware"]
