"""MCP 服务装配：FastMCP + Streamable HTTP + Bearer 鉴权。

五个 MCP 服务共用本模块，各自通过 ``python -m src.mcp.<domain>`` 独立启动。
每个服务只暴露本域工具；不需要在应用内挂载其他域的路径。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import uvicorn
from jsonschema import Draft202012Validator, ValidationError
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, CallToolRequest, ErrorData, ServerResult, Tool
from starlette.applications import Starlette

from src.queries.source_requests import CallBudget, SourceError, call_budget, setting

from .auth import BearerAuthMiddleware
from .config import MCP_API_TOKENS, MCP_HOST, MCP_PORT

logger = logging.getLogger(__name__)

ASGIApp = Callable[..., object]


class StrictToolsMCP(FastMCP):
    def streamable_http_app(self) -> Starlette:
        app = super().streamable_http_app()
        handlers = self.session_manager.app.request_handlers
        dispatch = handlers[CallToolRequest]

        async def call_known_tool(request: CallToolRequest) -> ServerResult:
            if not any(tool.name == request.params.name for tool in await self.list_tools()):
                raise McpError(ErrorData(code=INVALID_PARAMS, message="Unknown tool"))
            return await dispatch(request)

        # Dispatch unknown names before the SDK's tool handler converts exceptions to isError.
        handlers[CallToolRequest] = call_known_tool
        return app

    async def list_tools(self) -> list[Tool]:
        return [
            tool.model_copy(
                update={
                    "inputSchema": {**tool.inputSchema, "additionalProperties": False},
                }
            )
            for tool in await super().list_tools()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        # FastMCP permits coercion by default. Validate the published schema first.
        tool = next((tool for tool in await self.list_tools() if tool.name == name), None)
        if tool is not None:
            try:
                Draft202012Validator(tool.inputSchema).validate(arguments)
            except ValidationError as exc:
                field = ".".join(map(str, exc.absolute_path)) or "arguments"
                raise ToolError(
                    json.dumps(
                        {
                            "error": {
                                "code": "invalid_argument",
                                "retryable": False,
                                "attempt_count": 0,
                                "message": f"{field} 未通过 {exc.validator} 校验，请参见 tools/list",
                            }
                        },
                        ensure_ascii=False,
                    )
                ) from None
        duration = setting("MCP_CALL_TIMEOUT_SECONDS", 120, 1, 600)
        budget = CallBudget(time.monotonic() + duration)
        token = call_budget.set(budget)
        try:
            async with asyncio.timeout(duration):
                return await super().call_tool(name, arguments)
        except TimeoutError:
            raise ToolError(
                json.dumps(
                    {
                        "error": {
                            "code": "upstream_timeout",
                            "message": "工具调用已达到总时限",
                            "retryable": True,
                        }
                    },
                    ensure_ascii=False,
                )
            ) from None
        except ToolError as exc:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
                if isinstance(cause, SourceError):
                    raise ToolError(
                        json.dumps({"error": cause.payload()}, ensure_ascii=False)
                    ) from None
            if isinstance(cause, ValueError):
                code, message = "invalid_argument", str(cause)
            else:
                code, message = "upstream_unavailable", "工具执行失败"
            raise ToolError(
                json.dumps(
                    {
                        "error": {
                            "code": code,
                            "message": message,
                            "retryable": False,
                        }
                    },
                    ensure_ascii=False,
                )
            ) from None
        finally:
            budget.cancelled.set()
            call_budget.reset(token)


def _transport_security() -> TransportSecuritySettings:
    """按 ``MCP_ALLOWED_HOSTS`` 配置 Host 白名单。

    服务只经 Caddy 内网反代访问，默认关闭 DNS rebinding 保护；部署时可用
    ``MCP_ALLOWED_HOSTS=api.example.com`` 收紧（逗号分隔，支持 ``:端口``）。
    """
    allowed_hosts = [
        host.strip() for host in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    if not allowed_hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
    )


def build_mcp_app(
    name: str,
    streamable_http_path: str,
    register_tools: Callable[[FastMCP], None],
    *,
    instructions: str = "",
    mcp_class: type[FastMCP] = FastMCP,
) -> BearerAuthMiddleware:
    """构建单个 MCP 服务的 ASGI 应用（含鉴权）。"""
    mcp = mcp_class(
        name,
        instructions=instructions or None,
        streamable_http_path=streamable_http_path,
        transport_security=_transport_security(),
    )
    register_tools(mcp)
    health_paths = ("/healthz", f"{streamable_http_path.rstrip('/')}/healthz")
    return BearerAuthMiddleware(
        mcp.streamable_http_app(), MCP_API_TOKENS, health_paths=health_paths
    )


def run_server(app: ASGIApp) -> None:
    """以 uvicorn 启动 MCP ASGI 应用，监听 ``MCP_HOST:MCP_PORT``。"""
    if not MCP_API_TOKENS:
        logger.error("MCP_API_TOKENS 未配置：服务将拒绝所有请求，请检查环境变量")
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")


__all__ = ["build_mcp_app", "run_server"]
