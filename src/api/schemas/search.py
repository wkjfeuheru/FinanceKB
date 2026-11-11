"""下游检索 API 的请求/响应合同。

合同面向外部系统（Research-Copilot 等），因此：

* 全部模型 ``extra="forbid"``，未知字段直接 422——避免调用方误传
  ``collection`` / ``category`` / ``industry`` 等已废弃或未支持的字段后静默生效；
* 请求体不包含集合名，scope 由 HTTP 路径决定（research / policy）；
* 响应不返回 ``total``：向量检索没有精确总数，伪造总数会让分页语义失真。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.chunking.metadata_extractor import REPORT_TYPES

_DATE_FORMAT = "%Y-%m-%d"


def _is_valid_date(value: str) -> bool:
    if len(value) != 10:
        return False
    try:
        datetime.strptime(value, _DATE_FORMAT)
    except ValueError:
        return False
    return True


class SearchFilters(BaseModel):
    """检索过滤条件：只支持 broker / report_type / date。"""

    model_config = ConfigDict(extra="forbid")

    broker: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=20, description="券商名称列表，最多 20 个；空列表表示不限。"
    )
    report_type: list[Annotated[str, Field(json_schema_extra={"enum": list(REPORT_TYPES)})]] = Field(
        default_factory=list, max_length=len(REPORT_TYPES), description="报告类型列表，取值见 enum；空列表表示不限。"
    )
    date_from: Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")] | None = Field(
        default=None, description="报告日期起点 YYYY-MM-DD，含端点；null 表示不限，不得晚于 date_to。"
    )
    date_to: Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")] | None = Field(
        default=None, description="报告日期终点 YYYY-MM-DD，含端点；null 表示不限。"
    )

    @field_validator("report_type")
    @classmethod
    def _validate_report_type(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if value not in REPORT_TYPES]
        if invalid:
            raise ValueError(f"report_type 只能取 {', '.join(REPORT_TYPES)}；收到 {invalid}")
        return values

    @field_validator("date_from", "date_to")
    @classmethod
    def _validate_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _is_valid_date(value):
            raise ValueError("日期格式必须为 YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def _validate_range(self) -> SearchFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from 不能晚于 date_to")
        return self

    def to_milvus(self) -> dict[str, Any]:
        """转换为 ``HybridRetriever._build_filter`` 可消费的结构。"""
        filters: dict[str, Any] = {}
        if self.broker:
            filters["broker"] = list(self.broker)
        if self.report_type:
            filters["report_type"] = list(self.report_type)
        date_range: dict[str, str] = {}
        if self.date_from:
            date_range["gte"] = self.date_from
        if self.date_to:
            date_range["lte"] = self.date_to
        if date_range:
            filters["date"] = date_range
        return filters


class SearchRequest(BaseModel):
    """两条检索 API 共用的请求体。"""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=8, ge=1, le=50)
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @model_validator(mode="after")
    def _validate_window(self) -> SearchRequest:
        if self.page * self.page_size > 200:
            raise ValueError("page * page_size 不得超过 200")
        return self


class SearchHit(BaseModel):
    """单条检索结果。"""

    id: str
    source: str
    title: str
    content: str
    score: float
    rerank_score: float | None = None
    chunk: int
    block_type: str
    start_page: int
    end_page: int
    heading_path: str
    date: str
    report_type: str
    broker: str
    ingested_at: str | None = None


class SearchResponse(BaseModel):
    """检索响应；``as_of`` 为服务端响应时刻（UTC），仅用于标识数据新鲜度。"""

    collection: str
    as_of: str
    page: int
    page_size: int
    returned: int
    has_more: bool
    next_page: int | None
    items: list[SearchHit]


__all__ = [
    "SearchFilters",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]
