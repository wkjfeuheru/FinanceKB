"""行情工具与数据适配器共用的数据结构。"""

from datetime import date, datetime
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, FiniteFloat

InstrumentType = Literal["stock", "index"]
Interval = Literal["1m", "5m", "15m", "30m", "60m", "1d", "1w", "1M"]
INTERVALS = get_args(Interval)
INDEX_INTERVALS = {"1d": 101, "1w": 102, "1M": 103}
MAX_BARS = 5000
Symbol = Annotated[
    str,
    Field(
        pattern=r"^[0-9]{6}\.(SH|SZ|CSI)$",
        description="resolve_symbol 返回的标准代码，例如 600519.SH 或 000300.CSI。",
    ),
]
DateArgument = Annotated[
    str,
    Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        description="北京时间业务日期，格式 YYYY-MM-DD，包含此端点。",
    ),
]


class Instrument(BaseModel):
    symbol: Symbol
    name: str
    exchange: Literal["SSE", "SZSE", "CSI"]
    instrument_type: InstrumentType
    source: Literal["infoway", "eastmoney"]


class Bar(BaseModel):
    symbol: Symbol
    instrument_type: InstrumentType
    interval: Interval
    time: datetime = Field(description="来源 K 线时间，含时区")
    trade_date: date = Field(description="北京时间交易日期")
    open: FiniteFloat
    high: FiniteFloat
    low: FiniteFloat
    close: FiniteFloat
    price_unit: Literal["CNY", "point"]
    volume: FiniteFloat | None = None
    volume_unit: Literal["share", "lot"]
    turnover: FiniteFloat | None = Field(default=None, description="成交额，单位元")
    source: Literal["infoway", "eastmoney"]


class Quote(BaseModel):
    symbol: Symbol
    instrument_type: InstrumentType
    value: FiniteFloat
    value_unit: Literal["CNY", "point"]
    source_time: datetime = Field(description="新浪行情快照时间，含时区")
    change_pct: FiniteFloat | None = Field(default=None, description="涨跌幅，百分数")
    source: Literal["infoway", "sina"]


def date_range(start_date: str, end_date: str) -> tuple[date, date]:
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        raise ValueError("start_date 和 end_date 必须是有效日期（YYYY-MM-DD）") from None
    if start > end:
        raise ValueError("start_date 不得晚于 end_date")
    return start, end
