"""基本面工具与来源适配器共用的结构化数据。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, FiniteFloat

Period = Literal["annual", "quarterly"]
StatementKind = Literal["income", "balance", "cash_flow"]
StockSymbol = Annotated[
    str,
    Field(
        pattern=r"^[0-9]{6}\.(SH|SZ)$",
        description="resolve_symbol 返回的沪深股票标准代码，例如 600519.SH。",
    ),
]


def shanghai_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))


class FinancialLine(BaseModel):
    item_id: str
    name: str
    value: Decimal | None = Field(default=None, description="主源原始科目数值")
    ttm: Decimal | None = None
    group: str | None = None
    parent_item_id: str | None = None
    currency: str | None = None
    unit: str | None = None


class FinancialStatement(BaseModel):
    symbol: StockSymbol
    statement: StatementKind
    period: Period
    period_date: date
    period_basis: Literal["annual", "year_to_date", "as_of"]
    announcement_date: date | None = None
    source_update_date: date | None = None
    reporting_scope: str | None = None
    source: Literal["eastmoney"] = "eastmoney"
    lines: list[FinancialLine]


class ValuationMetric(BaseModel):
    symbol: StockSymbol
    metric: Literal["pe", "pb", "ps"]
    value: FiniteFloat
    trade_date: date
    source_time: datetime | None = Field(default=None, description="主源提供精确时间时填写")
    definition: str | None = None
    source: Literal["eastmoney"] = "eastmoney"


class CompanyOverview(BaseModel):
    name: str
    industry: str | None = None
    description: str | None = None
    main_business: str | None = None
    listed_date: date | None = None
    address: str | None = None
    website: str | None = None
    source: Literal["infoway", "eastmoney"] = "infoway"


class Executive(BaseModel):
    name: str
    role: str | None = None
    appointed_date: date | None = None
    source: Literal["eastmoney"] = "eastmoney"


class Shareholder(BaseModel):
    rank: Annotated[int, Field(ge=1, le=10)]
    name: str
    shares: int | None = None
    holding_pct: Decimal | None = Field(default=None, description="持股比例，百分数")
    share_type: str | None = None


class ShareholderList(BaseModel):
    kind: Literal["top_total", "top_float"]
    report_date: date
    denominator: Literal["total_shares", "float_shares"]
    source: Literal["eastmoney"] = "eastmoney"
    holders: list[Shareholder]


class CompanyProfile(BaseModel):
    symbol: StockSymbol
    company: CompanyOverview | None = None
    executives: list[Executive] = Field(default_factory=list)
    executives_reported: bool = Field(default=False, description="主源是否提供高管清单字段")
    shareholders: list[ShareholderList] = Field(default_factory=list)
    actual_controller: str | None = None
    controller_source: Literal["eastmoney"] | None = None


class DividendEvent(BaseModel):
    symbol: StockSymbol
    report_date: date
    announcement_date: date | None = None
    plan_announcement_date: date | None = None
    record_date: date | None = None
    ex_date: date | None = None
    payment_date: date | None = None
    status: str | None = None
    cash_per_share: Decimal | None = Field(default=None, description="每股税前现金红利，人民币元")
    bonus_shares_per_share: Decimal | None = None
    transfer_shares_per_share: Decimal | None = None
    source: Literal["eastmoney"] = "eastmoney"


class TradingCalendar(BaseModel):
    start_date: date
    end_date: date
    trade_days: list[date]
