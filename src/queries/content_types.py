"""宏观指标、统计日程、新闻、公告和业绩日历的结构化契约。"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, HttpUrl

from .fundamentals_types import StockSymbol

Indicator = Annotated[
    Literal["CPI", "GDP"],
    Field(description="CPI 为中国居民消费价格指数，GDP 为中国国内生产总值。"),
]
NewsKeyword = Annotated[
    str,
    Field(min_length=1, max_length=128, description="任意公司或主题搜索关键词，去除首尾空白后为1–128个可打印字符。"),
]


def news_keyword(value: str) -> str:
    value = value.strip()
    if not 1 <= len(value) <= 128 or not value.isprintable():
        raise ValueError("keyword 必须包含 1–128 个可打印字符")
    return value


class MacroMetric(BaseModel):
    series: Literal["national", "urban", "rural", "primary", "secondary", "tertiary"]
    metric: Literal["index", "yoy_pct", "mom_pct", "ytd_index", "gdp"]
    basis: Literal["monthly", "year_to_date"]
    value: Decimal
    unit: Literal["index_base_100", "percent", "CNY_100_million"]


class MacroObservation(BaseModel):
    indicator: Indicator
    frequency: Literal["monthly", "quarterly"]
    period_end: date = Field(description="月末或季末观察期日期")
    publication_date: date | None = Field(default=None, description="主源给出实际发布日期时填写")
    metrics: list[MacroMetric]
    source: Literal["eastmoney"] = "eastmoney"


class EconomicEvent(BaseModel):
    event: str
    planned_date: date
    url: HttpUrl
    source: Literal["nbs"] = "nbs"


class NewsArticle(BaseModel):
    article_id: str
    title: str
    published_at: AwareDatetime
    publisher: str | None = None
    url: HttpUrl | None = None
    source: Literal["eastmoney"] = "eastmoney"


class Filing(BaseModel):
    filing_id: str
    symbol: StockSymbol
    title: str
    categories: list[str]
    disclosure_date: date
    report_date: date | None = Field(default=None, description="公告标题可明确识别的财报期末")
    url: HttpUrl
    source: Literal["eastmoney"] = "eastmoney"


class EarningsAppointment(BaseModel):
    symbol: StockSymbol
    name: str
    report_date: date
    first_appointment_date: date | None = None
    change_dates: list[date] = Field(description="按预约变更次数的顺序排列")
    current_appointment_date: date | None = None
    actual_disclosure_date: date | None = None
    url: HttpUrl | None = None
    source: Literal["eastmoney"] = "eastmoney"


class EarningsMatch(EarningsAppointment):
    matched_date_types: list[Literal["current_appointment", "actual_disclosure"]]
