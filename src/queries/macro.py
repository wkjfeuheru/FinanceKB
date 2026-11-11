"""按需查询中国宏观指标和国家统计局发布日程。"""

from .base import QueryEnvelope
from .content import OptionalDate, content_result, fetch_content, query_window
from .content_types import EconomicEvent, Indicator, MacroObservation
from .fundamentals_types import shanghai_now
from .macro_sources import NBS_SCHEDULES, fetch_economic_calendar, fetch_macro_indicator

DOMAIN = "macro"


async def get_macro_indicator(
    indicator: Indicator,
    start_date: OptionalDate = "",
    end_date: OptionalDate = "",
) -> QueryEnvelope[MacroObservation]:
    """按月末或季末观察期查询东财中国CPI/GDP，返回指标值、口径、单位、来源及可用发布日期。默认近12个月，GDP为年初至季末累计；观察期与实际发布日期分别返回。只给start_date=2025-11-10时，终点补为2026-11-10。"""
    start, end = query_window(start_date, end_date, shanghai_now().date())
    rows = await fetch_content(fetch_macro_indicator, indicator)
    if rows and end < min(row.period_end for row in rows):
        return content_result(MacroObservation, DOMAIN, [], "macro_indicator_before_source_history")
    items = sorted(
        (row for row in rows if start <= row.period_end <= end),
        key=lambda row: row.period_end,
    )
    return content_result(MacroObservation, DOMAIN, items)


async def get_economic_calendar(
    start_date: OptionalDate = "",
    end_date: OptionalDate = "",
) -> QueryEnvelope[EconomicEvent]:
    """查询国家统计局2026年计划发布日程，返回事项、计划日期、原始日程链接与来源。默认今天起30天，涉及其他年份时返回available=false。只给start_date=2026-11-01时，终点补为2026-11-30。"""
    start, end = query_window(start_date, end_date, shanghai_now().date(), days=30, future=True)
    years = range(start.year, end.year + 1)
    missing_year = next((year for year in years if year not in NBS_SCHEDULES), None)
    if missing_year:
        return content_result(
            EconomicEvent, DOMAIN, [], f"economic_calendar_unavailable_for_year:{missing_year}"
        )
    rows = []
    for year in years:
        rows.extend(await fetch_content(fetch_economic_calendar, year))
    items = sorted(
        (row for row in rows if start <= row.planned_date <= end),
        key=lambda row: (row.planned_date, row.event),
    )
    return content_result(EconomicEvent, DOMAIN, items)


__all__ = ["DOMAIN", "get_economic_calendar", "get_macro_indicator"]
