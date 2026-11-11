"""宏观和资讯工具共用的实时请求和日期规则。"""

import asyncio
from calendar import monthrange
from collections.abc import Callable
from datetime import date, timedelta
from typing import Annotated, Any, TypeVar

import httpx
from pydantic import BaseModel, Field

from .base import QueryEnvelope, utc_now_iso
from .market_types import date_range

OptionalDate = Annotated[
    str,
    Field(
        pattern=r"^(?:[0-9]{4}-[0-9]{2}-[0-9]{2})?$",
        description="北京时间业务日期YYYY-MM-DD，含端点；空字符串表示缺省。只给起点时向后展开默认窗口，只给终点时向前展开，窗口长度见工具说明。",
    ),
]
Item = TypeVar("Item", bound=BaseModel)
T = TypeVar("T")


def _shift_year(day: date, years: int) -> date:
    year = day.year + years
    return date(year, day.month, min(day.day, monthrange(year, day.month)[1]))


def query_window(
    start_date: str, end_date: str, anchor: date, *, days: int | None = None, future: bool = False
) -> tuple[date, date]:
    if start_date and end_date:
        return date_range(start_date, end_date)
    value = start_date or end_date
    if value:
        anchor, _ = date_range(value, value)
    try:
        if start_date or (future and not end_date):
            end = _shift_year(anchor, 1) if days is None else anchor + timedelta(days=days - 1)
            return anchor, end
        start = _shift_year(anchor, -1) if days is None else anchor - timedelta(days=days - 1)
        return start, anchor
    except (OverflowError, ValueError):
        raise ValueError("日期超出默认查询窗口的可计算范围") from None


async def fetch_content(operation: Callable[..., T], *args: Any) -> T:
    def fetch() -> T:
        with httpx.Client() as client:
            return operation(client, *args)

    return await asyncio.to_thread(fetch)


def content_result(
    model: type[Item], domain: str, items: list[Item], reason: str = ""
) -> QueryEnvelope[Item]:
    return QueryEnvelope[model](
        available=not reason, domain=domain, as_of=utc_now_iso(), items=items, reason=reason
    )
