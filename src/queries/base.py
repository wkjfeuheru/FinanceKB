"""结构化查询的公共响应契约，区分数据不可用和查询成功但无记录。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

#: 占位返回的固定原因；接入真实数据源后由各查询模块替换。
DATA_SOURCE_NOT_CONNECTED = "data source not connected"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


Item = TypeVar("Item", default=dict[str, Any])


class QueryEnvelope(BaseModel, Generic[Item]):
    """结构化查询的统一返回信封。"""

    available: bool = Field(description="本次请求是否取得所需数据。false时查看reason；true且items为空表示查询成功但无匹配。执行错误通过MCP isError返回。")
    domain: str = Field(description="数据域：market / fundamentals / macro / news")
    reason: str = Field(default="", description="不可用原因；可用时为空")
    as_of: str = Field(description="服务端生成响应的 UTC 时间")
    items: list[Item] = Field(default_factory=list)


def unavailable(domain: str, reason: str = DATA_SOURCE_NOT_CONNECTED) -> QueryEnvelope:
    """构造稳定的占位信封：``available=false`` + 空 items。"""
    return QueryEnvelope(
        available=False,
        domain=domain,
        reason=reason,
        as_of=utc_now_iso(),
        items=[],
    )


def require_text(value: Any, field: str) -> str:
    """必填字符串参数校验（工具层只做轻量语义校验）。"""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{field} 不能为空")
    return text


__all__ = [
    "DATA_SOURCE_NOT_CONNECTED",
    "QueryEnvelope",
    "require_text",
    "unavailable",
    "utc_now_iso",
]
