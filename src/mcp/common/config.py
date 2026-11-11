"""MCP 服务侧配置：监听地址、鉴权令牌与上游知识检索 API。"""

from __future__ import annotations

import os

from src.core.config import (
    KNOWLEDGE_API_BASE_URL,
    KNOWLEDGE_API_TOKEN,
    MCP_API_TOKENS,
)

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0").strip() or "0.0.0.0"
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

__all__ = [
    "KNOWLEDGE_API_BASE_URL",
    "KNOWLEDGE_API_TOKEN",
    "MCP_API_TOKENS",
    "MCP_HOST",
    "MCP_PORT",
]
