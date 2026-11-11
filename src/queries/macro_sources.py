"""东财宏观数据与国家统计局统计日程采集。"""

import re
from calendar import monthrange
from datetime import date
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser

import httpx

from .content_http import ContentSourceError, eastmoney_page, fetch_text
from .content_types import EconomicEvent, Indicator, MacroMetric, MacroObservation

NBS_SCHEDULES = {
    2026: "https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html",
}
_NBS_HEADER = ["序号", "内容", *(f"{month}月" for month in range(1, 13))]
_NBS_DAYS = re.compile(r"(?<![0-9])([0-9]{1,2})\s*/\s*[一二三四五六日天]")
_CPI_SERIES = {"national": "NATIONAL", "urban": "CITY", "rural": "RURAL"}
_CPI_FIELDS = (
    ("BASE", "index", "monthly", "index_base_100"),
    ("SAME", "yoy_pct", "monthly", "percent"),
    ("SEQUENTIAL", "mom_pct", "monthly", "percent"),
    ("ACCUMULATE", "ytd_index", "year_to_date", "index_base_100"),
)
_GDP_FIELDS = {
    "national": ("DOMESTICL_PRODUCT_BASE", "SUM_SAME"),
    "primary": ("FIRST_PRODUCT_BASE", "FIRST_SAME"),
    "secondary": ("SECOND_PRODUCT_BASE", "SECOND_SAME"),
    "tertiary": ("THIRD_PRODUCT_BASE", "THIRD_SAME"),
}


def fetch_macro_indicator(client: httpx.Client, indicator: Indicator) -> list[MacroObservation]:
    report = "RPT_ECONOMY_CPI" if indicator == "CPI" else "RPT_ECONOMY_GDP"
    pages, count, rows = eastmoney_page(
        client,
        report,
        {"pageNumber": 1, "pageSize": 2000, "sortColumns": "REPORT_DATE", "sortTypes": "-1"},
    )
    if pages != 1 or len(rows) != count or not rows:
        raise ContentSourceError(f"东财 {indicator} 历史序列不完整")
    result = []
    seen = set()
    try:
        for row in rows:
            raw_date = date.fromisoformat(row["REPORT_DATE"][:10])
            year, month = raw_date.year, raw_date.month
            period_end = date(year, month, monthrange(year, month)[1])
            if period_end in seen or (indicator == "GDP" and month not in (3, 6, 9, 12)):
                raise ValueError
            seen.add(period_end)
            metrics = _metrics(row, indicator)
            if not metrics:
                raise ValueError
            result.append(
                MacroObservation(
                    indicator=indicator,
                    frequency="monthly" if indicator == "CPI" else "quarterly",
                    period_end=period_end,
                    metrics=metrics,
                )
            )
        return result
    except (KeyError, TypeError, ValueError, InvalidOperation):
        raise ContentSourceError(f"东财 {indicator} 指标或观察期格式异常") from None


def _metrics(row: dict, indicator: Indicator) -> list[MacroMetric]:
    if indicator == "CPI":
        fields = (
            (series, f"{prefix}_{suffix}", metric, basis, unit)
            for series, prefix in _CPI_SERIES.items()
            for suffix, metric, basis, unit in _CPI_FIELDS
        )
    else:
        fields = (
            (series, key, metric, "year_to_date", unit)
            for series, (value_key, growth_key) in _GDP_FIELDS.items()
            for key, metric, unit in (
                (value_key, "gdp", "CNY_100_million"),
                (growth_key, "yoy_pct", "percent"),
            )
        )
    return [
        MacroMetric(series=series, metric=metric, basis=basis, unit=unit, value=Decimal(str(value)))
        for series, key, metric, basis, unit in fields
        if (value := row.get(key)) not in (None, "", "-", "--")
    ]


class _ScheduleTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def fetch_economic_calendar(client: httpx.Client, year: int) -> list[EconomicEvent]:
    url = NBS_SCHEDULES.get(year)
    if url is None:
        raise ContentSourceError(f"尚未登记 {year} 年国家统计局年度发布日程")
    parser = _ScheduleTables()
    parser.feed(fetch_text(client, url, "国家统计局年度发布日程"))
    parser.close()
    schedule = next((table[1:] for table in parser.tables if table and table[0] == _NBS_HEADER), [])
    events = []
    try:
        for row in schedule:
            if not row or not row[0].isdigit():
                continue
            if len(row) != len(_NBS_HEADER) or not row[1]:
                raise ValueError
            for month, cell in enumerate(row[2:], 1):
                days = _NBS_DAYS.findall(cell)
                if not days and cell not in ("", "……", "…", "-", "--"):
                    raise ValueError
                events.extend(
                    EconomicEvent(event=row[1], planned_date=date(year, month, int(day)), url=url)
                    for day in days
                )
        if not events:
            raise ValueError
        return events
    except ValueError:
        raise ContentSourceError("国家统计局年度发布日程表格格式异常") from None
