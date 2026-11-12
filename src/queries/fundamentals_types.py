"""基本面工具与来源适配器共用的结构化数据。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, FiniteFloat

Period = Literal["annual", "quarterly"]
StatementKind = Literal["income", "balance", "cash_flow"]
SegmentDimension = Literal["product", "industry", "region"]
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


class SegmentRevenue(BaseModel):
    symbol: StockSymbol
    report_date: date = Field(description="报告期末日期，不是公告日；源未标明单季或累计口径。")
    dimension: SegmentDimension = Field(description="主营构成维度：产品、行业或地区；跨维度不可相加。")
    segment_name: str = Field(min_length=1, description="东财原始构成名称，不等同于法定报告分部。")
    revenue: Decimal | None = Field(default=None, allow_inf_nan=False, description="主营收入，元；Decimal字符串，缺失为null。")
    cost: Decimal | None = Field(default=None, allow_inf_nan=False, description="同一构成的主营成本，元；缺失不补零。")
    gross_profit: Decimal | None = Field(default=None, allow_inf_nan=False, description="源提供的主营利润或同一行收入减成本，元；不是净利润。")
    gross_profit_basis: Literal["source", "revenue_minus_cost"] | None = Field(
        default=None, description="source为源主营利润；revenue_minus_cost为同一行收入减成本；null表示无值。"
    )
    gross_margin_pct: Decimal | None = Field(default=None, allow_inf_nan=False, description="源提供的毛利率，百分数（23.9577表示23.9577%）；缺失不推导。")
    revenue_share_pct: Decimal | None = Field(default=None, allow_inf_nan=False, description="源提供的收入占比，百分数；不重算分母，缺失为null。")
    currency: None = Field(default=None, description="源未提供币种，固定为null，不推定为CNY。")
    unit: Literal["元"] = Field(default="元", description="收入、成本与利润的金额单位。")
    source: Literal["eastmoney"] = Field(default="eastmoney", description="东财F10主营构成，覆盖范围以源返回为准。")


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
