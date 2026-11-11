"""mcp-knowledge：RAG 检索 MCP 包装器（薄代理）。

两个工具的参数与 HTTP ``SearchRequest`` 一致；实际检索由 knowledge-api 完成
（唯一加载 ONNX 嵌入与 BGE 重排的进程）。本模块**不得** import Milvus / ONNX /
sentence-transformers / ``src.retrieve`` 服务层，否则会重复加载模型。
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from src.api.schemas.search import SearchFilters, SearchRequest, SearchResponse
from src.mcp.common.server import StrictToolsMCP, build_mcp_app
from src.mcp.knowledge.client import KnowledgeAPIClient, KnowledgeAPIError
from src.queries.source_requests import SourceError

NAME = "mcp-knowledge"
STREAMABLE_HTTP_PATH = "/mcp/knowledge"
INSTRUCTIONS = (
    "研报与政策检索 MCP。search_research 检索 finance_report 集合，"
    "search_policy 检索 finance_policy 集合（本期为空，返回正常的空 items）。"
)

_client: KnowledgeAPIClient | None = None
Query = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    Field(description="检索问题或关键词，去除首尾空白后 1–500 字符。"),
]
Page = Annotated[int, Field(ge=1, description="页码，从 1 开始；page × page_size 必须不超过 200。")]
PageSize = Annotated[
    int, Field(ge=1, le=50, description="每页最多 1–50 条，默认 8；与页码乘积不超过 200。")
]
Filters = Annotated[
    SearchFilters | None, Field(description="可选券商、报告类型、报告日期过滤；省略表示不限制。")
]


class PolicyFilters(BaseModel):
    """政策检索的日期过滤（政策文档无券商/报告类型语义）。"""

    model_config = ConfigDict(extra="forbid")

    date_from: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    ] | None = Field(default=None, description="政策发布日期起点 YYYY-MM-DD，含端点；null 表示不限，不能晚于 date_to。")
    date_to: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    ] | None = Field(default=None, description="政策发布日期终点 YYYY-MM-DD，含端点；null 表示不限。")

    @model_validator(mode="after")
    def _validate_range(self) -> PolicyFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from 不能晚于 date_to")
        return self

    def to_search_filters(self) -> SearchFilters:
        return SearchFilters(date_from=self.date_from, date_to=self.date_to)


PolicyFilterArg = Annotated[
    PolicyFilters | None, Field(description="可选政策发布日期过滤；省略表示不限制。")
]


def get_client() -> KnowledgeAPIClient:
    global _client
    if _client is None:
        _client = KnowledgeAPIClient()
    return _client


def _tool_error(exc: KnowledgeAPIError) -> ToolError:
    if exc.status_code == 401:
        return ToolError("知识检索服务内部鉴权失败，请联系管理员")
    if exc.status_code in (400, 422):
        return ToolError("检索参数无效，请检查 query / page / page_size / filters")
    return ToolError("知识检索服务暂时不可用，请稍后重试")


def _build_request(
    query: str,
    page: int,
    page_size: int,
    filters: SearchFilters | None,
) -> SearchRequest:
    try:
        return SearchRequest(
            query=query,
            page=page,
            page_size=page_size,
            filters=filters or SearchFilters(),
        )
    except ValidationError as exc:
        raise ToolError("检索参数无效，请检查 query / page / page_size / filters") from exc


async def _search(scope: str, request: SearchRequest) -> SearchResponse:
    try:
        payload = await get_client().search(scope, request.model_dump(mode="json"))
    except KnowledgeAPIError as exc:
        raise _tool_error(exc) from exc
    try:
        response = SearchResponse.model_validate(payload)
    except ValidationError:
        raise SourceError("知识检索服务返回了无法识别的结果", source="knowledge-api") from None
    return response


def register_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        name="search_research",
        description="检索研报知识库 finance_report，返回相关片段、标题、来源页码、相关性分数和分页信息。无匹配时 items 为空。",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def search_research(
        query: Query,
        page: Page = 1,
        page_size: PageSize = 8,
        filters: Filters = None,
    ) -> SearchResponse:
        return await _search("research", _build_request(query, page, page_size, filters))

    @mcp.tool(
        name="search_policy",
        description=(
            "检索政策知识库 finance_policy，返回相关片段、标题、来源页码、分数和分页信息。"
            "无匹配或集合为空时 items 为空。"
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def search_policy(
        query: Query,
        page: Page = 1,
        page_size: PageSize = 8,
        filters: PolicyFilterArg = None,
    ) -> SearchResponse:
        mapped = filters.to_search_filters() if filters is not None else None
        return await _search("policy", _build_request(query, page, page_size, mapped))


def create_app():
    return build_mcp_app(
        NAME,
        STREAMABLE_HTTP_PATH,
        register_tools,
        instructions=INSTRUCTIONS,
        mcp_class=StrictToolsMCP,
    )


__all__ = [
    "NAME",
    "STREAMABLE_HTTP_PATH",
    "create_app",
    "get_client",
    "register_tools",
]
