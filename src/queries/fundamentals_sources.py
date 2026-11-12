"""基本面主源读取与格式归一化。"""

from __future__ import annotations

import gzip
import json
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
from html.parser import HTMLParser
from typing import Any

import httpx
from infoway import Market

from src.queries.fundamentals_types import (
    CompanyOverview,
    CompanyProfile,
    DividendEvent,
    Executive,
    FinancialLine,
    FinancialStatement,
    Period,
    SegmentDimension,
    SegmentRevenue,
    Shareholder,
    ShareholderList,
    TradingCalendar,
    ValuationMetric,
)
from src.queries.infoway_source import DataSourceError, call_infoway

from .source_requests import get, log_fallback

_F10_BASE = "https://emweb.securities.eastmoney.com/PC_HSF10/"
_FINANCE_BASE = _F10_BASE + "NewFinanceAnalysis/"
_SHAREHOLDER_BASE = _F10_BASE + "ShareholderResearch/"
_DATA_CENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_STATEMENT_ENDPOINTS = {"income": "lrb", "balance": "zcfzb", "cash_flow": "xjllb"}
_REPORT_METADATA = {
    "SECUCODE",
    "CURRENCY",
    "NOTICE_DATE",
    "UPDATE_DATE",
    "OPINION_TYPE",
    "OSOPINION_TYPE",
    "ISNEW",
    "DATA_STATE",
    "DATE_TYPE",
}
_FINANCIAL_FIELDS = {
    "TOTAL_OPERATE_INCOME": ("营业总收入", "元"),
    "OPERATE_INCOME": ("营业收入", "元"),
    "TOTAL_OPERATE_COST": ("营业总成本", "元"),
    "OPERATE_COST": ("营业成本", "元"),
    "OPERATE_PROFIT": ("营业利润", "元"),
    "TOTAL_PROFIT": ("利润总额", "元"),
    "NETPROFIT": ("净利润", "元"),
    "PARENT_NETPROFIT": ("归母净利润", "元"),
    "BASIC_EPS": ("基本每股收益", "元/股"),
    "DILUTED_EPS": ("稀释每股收益", "元/股"),
    "TOTAL_ASSETS": ("资产总计", "元"),
    "TOTAL_LIABILITIES": ("负债合计", "元"),
    "TOTAL_EQUITY": ("所有者权益合计", "元"),
    "NETCASH_OPERATE": ("经营活动现金流量净额", "元"),
    "NETCASH_INVEST": ("投资活动现金流量净额", "元"),
    "NETCASH_FINANCE": ("筹资活动现金流量净额", "元"),
}


def _date(value: str | None) -> date | None:
    if value in (None, "", "-", "--"):
        return None
    return date.fromisoformat(value[:10])


def _per_share(value: Any) -> Decimal | None:
    # Eastmoney's A-share distribution terms are stated per ten shares.
    return Decimal(str(value)) / 10 if value is not None else None


class _CompanyTypeParser(HTMLParser):
    value: str | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "input" and attributes.get("id") == "hidctype":
            self.value = attributes.get("value")


@lru_cache(maxsize=128)
def _company_type(code: str) -> str:
    parser = _CompanyTypeParser()
    parser.feed(
        _eastmoney_response(
            _FINANCE_BASE + "Index",
            {"type": "web", "code": code.lower()},
        ).text
    )
    if not parser.value or not parser.value.isascii() or not parser.value.isdigit():
        raise DataSourceError("东财 F10 未返回有效的公司类型")
    return parser.value


def _statement_lines(row: dict[str, Any]) -> list[FinancialLine]:
    lines = []
    for field, value in row.items():
        if (
            field in _REPORT_METADATA
            or field.startswith(("SECURITY_", "ORG_", "REPORT_"))
            or field.endswith(("_YOY", "_QOQ"))
            or value in (None, "", "-", "--")
            or isinstance(value, bool)
        ):
            continue
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            if field in _FINANCIAL_FIELDS:
                raise
            continue
        if not amount.is_finite():
            raise ValueError
        name, unit = _FINANCIAL_FIELDS.get(field, (field, None))
        currency = row.get("CURRENCY") if unit else None
        lines.append(
            FinancialLine(
                item_id=field,
                name=name,
                value=amount,
                currency=currency,
                unit=unit if currency == "CNY" else None,
            )
        )
    return lines


def fetch_financial_statements(
    symbol: str, period: Period, start: date, end: date
) -> list[FinancialStatement]:
    code = f"{symbol[7:]}{symbol[:6]}"
    params = {
        "companyType": _company_type(code),
        "reportDateType": "1" if period == "annual" else "0",
        "code": code,
    }
    result = []
    for statement, endpoint in _STATEMENT_ENDPOINTS.items():
        try:
            dates = _eastmoney(_FINANCE_BASE + endpoint + "DateAjaxNew", params)["data"]
            if not isinstance(dates, list) or any(
                row["SECURITY_CODE"] != symbol[:6] for row in dates
            ):
                raise ValueError
            available_dates = {date.fromisoformat(row["REPORT_DATE"][:10]) for row in dates}
            selected = sorted((day for day in available_dates if start <= day < end), reverse=True)
            basis = (
                "as_of"
                if statement == "balance"
                else ("annual" if period == "annual" else "year_to_date")
            )
            for i in range(0, len(selected), 5):
                batch = selected[i : i + 5]
                detail = _eastmoney(
                    _FINANCE_BASE + endpoint + "AjaxNew",
                    {
                        **params,
                        "reportType": "1",
                        "dates": ",".join(day.isoformat() for day in batch),
                    },
                )["data"]
                if not isinstance(detail, list):
                    raise ValueError
                seen = set()
                for row in detail:
                    period_date = date.fromisoformat(row["REPORT_DATE"][:10])
                    if row["SECUCODE"] != symbol or period_date not in batch or period_date in seen:
                        raise ValueError
                    lines = _statement_lines(row)
                    if not lines:
                        raise ValueError
                    seen.add(period_date)
                    result.append(
                        FinancialStatement(
                            symbol=symbol,
                            statement=statement,
                            period=period,
                            period_date=period_date,
                            period_basis=basis,
                            announcement_date=_date(row.get("NOTICE_DATE")),
                            source_update_date=_date(row.get("UPDATE_DATE")),
                            lines=lines,
                        )
                    )
                if seen != set(batch):
                    raise ValueError
        except (KeyError, TypeError, ValueError, InvalidOperation):
            raise DataSourceError(f"东财财报 {statement} 响应格式、标的或报告期异常") from None
    return result


def _segment_number(value: Any) -> Decimal | None:
    if value in (None, "", "-", "--"):
        return None
    if isinstance(value, bool):
        raise ValueError
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError
    return number


def fetch_segment_revenue(symbol: str) -> list[SegmentRevenue]:
    try:
        raw = _eastmoney(
            _F10_BASE + "BusinessAnalysis/PageAjax", {"code": f"{symbol[7:]}{symbol[:6]}"}
        )
    except DataSourceError as exc:
        exc.source = "eastmoney"
        raise
    try:
        rows = raw["zygcfx"]
        if not isinstance(rows, list):
            raise ValueError
        dimensions: dict[str, SegmentDimension] = {"1": "industry", "2": "product", "3": "region"}
        records: dict[tuple[str, date, SegmentDimension, str], SegmentRevenue] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError
            if row["SECUCODE"] != symbol or row["SECURITY_CODE"] != symbol[:6]:
                raise ValueError
            raw_date = row["REPORT_DATE"]
            if not isinstance(raw_date, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?: [0-9]{2}:[0-9]{2}:[0-9]{2})?", raw_date
            ):
                raise ValueError
            report_date = datetime.fromisoformat(raw_date).date()
            name = row["ITEM_NAME"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError
            revenue = _segment_number(row.get("MAIN_BUSINESS_INCOME"))
            cost = _segment_number(row.get("MAIN_BUSINESS_COST"))
            profit = _segment_number(row.get("MAIN_BUSINESS_RPOFIT"))
            margin = _segment_number(row.get("GROSS_RPOFIT_RATIO"))
            share = _segment_number(row.get("MBI_RATIO"))
            basis = "source" if profit is not None else None
            numbers = [n for n in (revenue, cost, margin, share) if n is not None]
            with localcontext() as context:
                # Decimal's default precision can round source amounts during arithmetic.
                if numbers:
                    context.prec = max(
                        context.prec,
                        max(n.adjusted() for n in numbers)
                        - min(int(n.as_tuple().exponent) for n in numbers) + 3,
                    )
                if profit is None and revenue is not None and cost is not None:
                    profit = revenue - cost
                    basis = "revenue_minus_cost"
                margin = margin * 100 if margin is not None else None
                share = share * 100 if share is not None else None
            item = SegmentRevenue(
                symbol=symbol,
                report_date=report_date,
                dimension=dimensions[row["MAINOP_TYPE"]],
                segment_name=name,
                revenue=revenue,
                cost=cost,
                gross_profit=profit,
                gross_profit_basis=basis,
                gross_margin_pct=margin,
                revenue_share_pct=share,
            )
            key = (item.symbol, item.report_date, item.dimension, item.segment_name)
            if key in records and records[key] != item:
                raise ValueError
            records[key] = item
        return list(records.values())
    except (KeyError, TypeError, ValueError, InvalidOperation):
        raise DataSourceError("东财主营构成响应格式、标的或报告期异常", source="eastmoney") from None


def fetch_valuation_metrics(symbol: str) -> list[ValuationMetric]:
    raw = _eastmoney(
        _DATA_CENTER_URL,
        {
            "reportName": "RPT_VALUEANALYSIS_DET",
            "columns": "ALL",
            "pageNumber": 1,
            "pageSize": 10,
            "sortColumns": "TRADE_DATE",
            "sortTypes": "-1",
            "source": "WEB",
            "client": "WEB",
            "filter": f'(SECURITY_CODE="{symbol[:6]}")',
        },
    )
    if raw.get("success") is not True:
        raise DataSourceError("东财估值查询失败")
    try:
        rows = raw["result"]["data"]
        if not isinstance(rows, list):
            raise ValueError
        items = []
        for row in rows:
            if row["SECUCODE"] != symbol:
                raise ValueError
            trade_date = date.fromisoformat(row["TRADE_DATE"][:10])
            for metric, field in (("pe", "PE_TTM"), ("pb", "PB_MRQ"), ("ps", "PS_TTM")):
                value = row.get(field)
                if value not in (None, "", "-", "--"):
                    items.append(
                        ValuationMetric(
                            symbol=symbol,
                            metric=metric,
                            value=value,
                            trade_date=trade_date,
                            definition=field,
                        )
                    )
        return items
    except (KeyError, TypeError, ValueError):
        raise DataSourceError("东财估值响应格式或标的异常") from None


def fetch_trading_calendar(start: date, end: date) -> TradingCalendar:
    try:
        raw = call_infoway(
            lambda client: client.basic.get_trading_days(
                Market.CN,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
            )
        )
        days = sorted({datetime.strptime(day, "%Y%m%d").date() for day in raw["trade_days"]})
        if not days or any(not start <= day <= end for day in days):
            raise ValueError
    except (DataSourceError, KeyError, TypeError, ValueError) as exc:
        log_fallback("infoway", "eastmoney", exc)
        days = _eastmoney_trading_days(start, end)
    return TradingCalendar(start_date=start, end_date=end, trade_days=days)


def _eastmoney_trading_days(start: date, end: date) -> list[date]:
    raw = _eastmoney(
        _DATA_CENTER_URL,
        {
            "reportName": "RPTA_WEB_TRADE_DATE",
            "columns": "ALL",
            "pageNumber": 1,
            "pageSize": 100,
            "sortColumns": "DATE",
            "sortTypes": "1",
            "source": "WEB",
            "client": "WEB",
            # Shanghai A-share sessions use the same trading dates as Shenzhen.
            "filter": f"(MARKETCODE=\"069001001\")(DATE>='{start}')(DATE<='{end}')",
        },
    )
    try:
        result = raw["result"]
        rows = result["data"]
        if raw["success"] is not True or result["pages"] != 1 or len(rows) != result["count"]:
            raise ValueError
        days = [date.fromisoformat(row["DATE"][:10]) for row in rows]
        if (
            not days
            or len(set(days)) != len(days)
            or any(
                row["MARKETCODE"] != "069001001" or not start <= day <= end
                for row, day in zip(rows, days, strict=True)
            )
        ):
            raise ValueError
        return sorted(days)
    except (KeyError, TypeError, ValueError):
        raise DataSourceError("东财交易日历查询失败或响应格式异常") from None


def _eastmoney_response(url: str, params: dict[str, str | int]) -> httpx.Response:
    try:
        response = get(
            url, source="eastmoney",
            params=params,
            timeout=8.0,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
        )
        response.raise_for_status()
        return response
    except httpx.HTTPError:
        raise DataSourceError("东财请求失败") from None


def _eastmoney(url: str, params: dict[str, str | int]) -> dict[str, Any]:
    response = _eastmoney_response(url, params)
    try:
        content = response.content
        if content.startswith(b"\x1f\x8b"):
            content = gzip.decompress(content)
        raw = json.loads(content, parse_float=Decimal)
        if not isinstance(raw, dict):
            raise ValueError
        return raw
    except (ValueError, OSError, EOFError):
        raise DataSourceError("东财响应格式异常") from None


def _infoway_company_overview(symbol: str) -> CompanyOverview:
    raw = call_infoway(lambda client: client.stock_info.get_company(symbol, lang="zh-CN"))
    try:
        basic, profile = raw["basic_info"], raw["company_profile"]
        expected_counter = f"ST/{symbol[7:]}/{symbol[:6]}"
        if profile.get("counter_id") not in (None, "", expected_counter):
            raise ValueError
        industry = basic.get("industry") or {}
        return CompanyOverview(
            name=profile.get("company_name_zhcn") or profile.get("company_name") or basic["name"],
            industry=industry.get("name"),
            description=profile.get("company_desc") or basic.get("profile"),
            main_business=profile.get("main_business"),
            listed_date=_date(basic.get("listed_date")),
            address=profile.get("address"),
            website=profile.get("webpage"),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise DataSourceError("Infoway 公司概况响应格式或标的异常") from None


def _eastmoney_company_overview(symbol: str) -> CompanyOverview:
    raw = _eastmoney(_F10_BASE + "CompanySurvey/PageAjax", {"code": f"{symbol[7:]}{symbol[:6]}"})
    try:
        (basic,) = raw["jbzl"]
        issue = raw.get("fxxg") or []
        if basic["SECUCODE"] != symbol or any(row["SECUCODE"] != symbol for row in issue):
            raise ValueError
        return CompanyOverview(
            name=basic["ORG_NAME"],
            industry=basic.get("INDUSTRYCSRC1") or basic.get("EM2016"),
            description=basic.get("ORG_PROFILE"),
            listed_date=_date(issue[0].get("LISTING_DATE")) if issue else None,
            address=basic.get("ADDRESS"),
            website=basic.get("ORG_WEB"),
            source="eastmoney",
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise DataSourceError("东财公司概况响应格式或标的异常") from None


def fetch_company_profile(symbol: str) -> CompanyProfile:
    try:
        company = _infoway_company_overview(symbol)
    except DataSourceError as exc:
        log_fallback("infoway", "eastmoney", exc)
        company = _eastmoney_company_overview(symbol)
    management = _executives(symbol)
    shareholders, controller = _shareholders(symbol)
    return CompanyProfile(
        symbol=symbol,
        company=company,
        executives=management,
        executives_reported=True,
        shareholders=shareholders,
        actual_controller=controller,
        controller_source="eastmoney" if controller else None,
    )


def _executives(symbol: str) -> list[Executive]:
    raw = _eastmoney(
        _F10_BASE + "CompanyManagement/PageAjax", {"code": f"{symbol[7:]}{symbol[:6]}"}
    )
    try:
        rows = raw["gglb"]
        if not isinstance(rows, list) or any(row["SECUCODE"] != symbol for row in rows):
            raise ValueError
        items = []
        for row in rows:
            start = re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", row.get("INCUMBENT_TIME") or "")
            items.append(
                Executive(
                    name=row["PERSON_NAME"],
                    role=row.get("POSITION") or None,
                    appointed_date=date.fromisoformat(start[0]) if start else None,
                )
            )
        return items
    except (KeyError, TypeError, ValueError):
        raise DataSourceError("东财董监高响应格式或标的异常") from None


def _shareholders(symbol: str) -> tuple[list[ShareholderList], str | None]:
    code = f"{symbol[7:]}{symbol[:6]}"
    meta = _eastmoney(_SHAREHOLDER_BASE + "PageAjax", {"code": code})
    groups = []
    try:
        for kind, key, endpoint, ratio, denominator in (
            ("top_total", "sdgd", "PageSDGD", "HOLD_NUM_RATIO", "total_shares"),
            ("top_float", "sdltgd", "PageSDLTGD", "FREE_HOLDNUM_RATIO", "float_shares"),
        ):
            periods = [
                date.fromisoformat(row["END_DATE"][:10])
                for row in meta[f"{key}_date"]
                if row["SECUCODE"] == symbol
                and (kind == "top_float" or str(row.get("IS_REPORTDATE")) == "1")
            ]
            if not periods:
                continue
            report_date = max(periods)
            raw = _eastmoney(
                _SHAREHOLDER_BASE + endpoint, {"code": code, "date": report_date.isoformat()}
            )
            holders = []
            for row in raw[key]:
                if row["SECUCODE"] != symbol or _date(row["END_DATE"]) != report_date:
                    raise ValueError
                holders.append(
                    Shareholder(
                        rank=row["HOLDER_RANK"],
                        name=row["HOLDER_NAME"],
                        shares=row.get("HOLD_NUM"),
                        holding_pct=row.get(ratio),
                        share_type=row.get("SHARES_TYPE"),
                    )
                )
            if holders:
                groups.append(
                    ShareholderList(
                        kind=kind,
                        report_date=report_date,
                        denominator=denominator,
                        holders=sorted(holders, key=lambda row: row.rank),
                    )
                )
        controllers = meta["sjkzr"]
        if any(row["SECUCODE"] != symbol for row in controllers):
            raise ValueError
        names = [row["HOLDER_NAME"] for row in controllers]
        return groups, "、".join(names) if names else None
    except (KeyError, TypeError, ValueError, AttributeError):
        raise DataSourceError("东财股东数据响应格式或报告期异常") from None


def fetch_dividend_history(symbol: str) -> list[DividendEvent]:
    params = {
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": "ALL",
        "pageNumber": 1,
        "pageSize": 500,
        "sortColumns": "NOTICE_DATE",
        "sortTypes": "-1",
        "source": "WEB",
        "client": "WEB",
        "filter": f'(SECURITY_CODE="{symbol[:6]}")',
    }
    raw = _eastmoney(_DATA_CENTER_URL, params)
    if raw.get("success") is not True:
        raise DataSourceError("东财分红查询失败")
    try:
        page = raw["result"]
        if not isinstance(page["data"], list):
            raise TypeError
        # The tool returns at most 100 events, so the first date-ordered page suffices.
        items = []
        for row in page["data"]:
            if row["SECUCODE"] != symbol:
                raise ValueError
            items.append(
                DividendEvent(
                    symbol=symbol,
                    report_date=date.fromisoformat(row["REPORT_DATE"][:10]),
                    announcement_date=_date(row.get("NOTICE_DATE")),
                    plan_announcement_date=_date(row.get("PLAN_NOTICE_DATE")),
                    record_date=_date(row.get("EQUITY_RECORD_DATE")),
                    ex_date=_date(row.get("EX_DIVIDEND_DATE")),
                    payment_date=_date(row.get("PAY_DATE")),
                    status=row.get("ASSIGN_PROGRESS"),
                    cash_per_share=_per_share(row.get("PRETAX_BONUS_RMB")),
                    bonus_shares_per_share=_per_share(row.get("BONUS_RATIO")),
                    transfer_shares_per_share=_per_share(row.get("IT_RATIO")),
                )
            )
        return items
    except (KeyError, TypeError, ValueError, InvalidOperation):
        raise DataSourceError("东财分红数据响应格式或标的异常") from None
