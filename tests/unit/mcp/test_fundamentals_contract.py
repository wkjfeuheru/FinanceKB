from __future__ import annotations

import decimal
import json
from datetime import date, datetime
from decimal import Decimal

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.mcp.common.server import StrictToolsMCP
from src.mcp.fundamentals.server import register_tools
from src.queries import fundamentals, fundamentals_sources, source_requests
from src.queries.fundamentals_types import (
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    TradingCalendar,
    ValuationMetric,
)


@pytest.fixture
def tools(monkeypatch):
    data = {}

    def require_stock(symbol):
        if symbol == "000001.SH":
            raise ValueError("需要股票代码")

    def fetch_rows(symbol, dataset, model):
        return [model.model_validate(row) for row in data[symbol, dataset]["items"]]

    monkeypatch.setattr(fundamentals, "require_stock", require_stock)
    monkeypatch.setattr(
        fundamentals,
        "fetch_financial_statements",
        lambda symbol, *_: fetch_rows(
            symbol,
            "eastmoney_annual",
            FinancialStatement,
        ),
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_valuation_metrics",
        lambda symbol: fetch_rows(
            symbol,
            "eastmoney_valuation",
            ValuationMetric,
        ),
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_trading_calendar",
        lambda *_: TradingCalendar.model_validate(
            data["CN", "trading_days"],
        ),
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_company_profile",
        lambda symbol: fetch_rows(
            symbol,
            "profile_em_management",
            CompanyProfile,
        )[0],
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_dividend_history",
        lambda symbol: fetch_rows(
            symbol,
            "dividend",
            DividendEvent,
        ),
    )
    monkeypatch.setattr(
        fundamentals,
        "shanghai_now",
        lambda: datetime.fromisoformat("2026-11-09T16:00:00+08:00"),
        raising=False,
    )
    server = StrictToolsMCP("fundamentals-test")
    register_tools(server)
    return server, data


@pytest.mark.asyncio
async def test_financial_statements_keeps_period_and_source_precision(tools):
    server, data = tools
    record = {
        "symbol": "600519.SH",
        "statement": "income",
        "period": "annual",
        "period_date": "2024-12-31",
        "period_basis": "annual",
        "source": "eastmoney",
        "announcement_date": "2025-04-03",
        "lines": [{"item_id": "OPERATE_INCOME", "name": "营业收入", "value": "170899152276.34"}],
    }
    data["600519.SH", "eastmoney_annual"] = {
        "items": [record, {**record, "period_date": "2021-12-31"}]
    }
    _, result = await server.call_tool(
        "get_financial_statements", {"symbol": "600519.SH", "years": 3}
    )
    assert result["available"] is True
    assert len(result["items"]) == 1
    assert result["items"][0]["period_date"] == "2024-12-31"
    assert result["items"][0]["announcement_date"] == "2025-04-03"
    assert result["items"][0]["lines"][0]["value"] == "170899152276.34"


@pytest.mark.asyncio
async def test_valuation_does_not_relabel_stale_or_future_metrics(tools):
    server, data = tools
    data["CN", "trading_days"] = {
        "start_date": "2026-10-26",
        "end_date": "2026-11-23",
        "trade_days": ["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"],
    }
    metric = {
        "symbol": "600519.SH",
        "metric": "pe",
        "value": 21.5,
        "source": "eastmoney",
        "trade_date": "2026-11-05",
        "definition": "PE_TTM",
    }
    data["600519.SH", "eastmoney_valuation"] = {
        "items": [
            metric,
            {
                **metric,
                "trade_date": "2026-11-10",
            },
        ]
    }
    _, result = await server.call_tool("get_valuation_metrics", {"symbol": "600519.SH"})
    assert result["available"] is False
    assert "2026-11-09" in result["reason"]
    assert result["items"] == []
    data["600519.SH", "eastmoney_valuation"]["items"].append(
        {
            **metric,
            "trade_date": "2026-11-09",
        }
    )
    _, result = await server.call_tool("get_valuation_metrics", {"symbol": "600519.SH"})
    assert result["available"] is True
    assert [row["trade_date"] for row in result["items"]] == ["2026-11-09"]
    assert result["items"][0]["source_time"] is None


@pytest.mark.asyncio
async def test_company_profile_keeps_primary_sources_and_shareholder_basis(tools):
    server, data = tools
    data["600519.SH", "profile_em_management"] = {
        "items": [
            {
                "symbol": "600519.SH",
                "company": {"name": "贵州茅台酒股份有限公司", "source": "infoway"},
                "executives": [
                    {
                        "name": "示例高管",
                        "role": "董事长",
                        "source": "eastmoney",
                        "appointed_date": "2025-11-28",
                    }
                ],
                "executives_reported": True,
                "shareholders": [
                    {
                        "kind": "top_float",
                        "report_date": "2026-09-30",
                        "denominator": "float_shares",
                        "source": "eastmoney",
                        "holders": [
                            {
                                "rank": 1,
                                "name": "示例股东",
                                "shares": 10000,
                                "holding_pct": "10.5",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    _, result = await server.call_tool("get_company_profile", {"symbol": "600519.SH"})
    profile = result["items"][0]
    assert profile["company"]["source"] == "infoway"
    assert profile["executives"][0]["role"] == "董事长"
    assert profile["executives"][0]["source"] == "eastmoney"
    assert profile["executives"][0]["appointed_date"] == "2025-11-28"
    assert profile["shareholders"][0]["denominator"] == "float_shares"
    assert profile["shareholders"][0]["source"] == "eastmoney"


@pytest.mark.asyncio
async def test_dividend_limit_and_empty_source_result(tools):
    server, data = tools
    row = {
        "symbol": "600519.SH",
        "report_date": "2025-12-31",
        "announcement_date": "2026-06-22",
        "cash_per_share": "28.02423",
        "status": "实施方案",
        "source": "eastmoney",
    }
    data["600519.SH", "dividend"] = {
        "items": [
            {**row, "announcement_date": "2025-12-11"},
            row,
        ]
    }
    _, result = await server.call_tool("get_dividend_history", {"symbol": "600519.SH", "limit": 1})
    assert result["items"][0]["cash_per_share"] == "28.02423"
    assert [r["announcement_date"] for r in result["items"]] == ["2026-06-22"]
    data["600519.SH", "dividend"] = {"items": []}
    _, result = await server.call_tool("get_dividend_history", {"symbol": "600519.SH"})
    assert result["available"] is True and result["items"] == []


@pytest.mark.asyncio
async def test_fundamentals_rejects_wrong_types_and_index_symbol(tools):
    server, _ = tools
    with pytest.raises(ToolError):
        await server.call_tool("get_financial_statements", {"symbol": "600519.SH", "years": "3"})
    with pytest.raises(ToolError, match="股票"):
        await server.call_tool("get_financial_statements", {"symbol": "000001.SH"})


def test_eastmoney_financial_adapter_keeps_revenue_and_cumulative_basis(monkeypatch):
    def request(url, params):
        if "DateAjax" in url:
            return {"data": [{"SECURITY_CODE": "600519", "REPORT_DATE": "2026-09-30 00:00:00"}]}
        assert params["dates"] == "2026-09-30"
        assert params["reportType"] == "1"
        return {
            "data": [
                {
                    "SECUCODE": "600519.SH",
                    "REPORT_DATE": "2026-09-30 00:00:00",
                    "NOTICE_DATE": "2026-10-31 00:00:00",
                    "CURRENCY": "CNY",
                    "TOTAL_OPERATE_INCOME": "133283652508.95",
                    "OPERATE_INCOME": "130970743649.04",
                    "OPERATE_INCOME_YOY": "1.96",
                }
            ]
        }

    monkeypatch.setattr(fundamentals_sources, "_eastmoney", request)
    monkeypatch.setattr(fundamentals_sources, "_company_type", lambda _: "4", raising=False)
    monkeypatch.setattr(
        fundamentals_sources,
        "call_infoway",
        lambda _: pytest.fail("财报仍走 Infoway"),
    )
    rows = fundamentals_sources.fetch_financial_statements(
        "600519.SH",
        "quarterly",
        date(2026, 7, 1),
        date(2026, 10, 1),
    )
    assert len(rows) == 3
    assert {r.statement: r.period_basis for r in rows} == {
        "income": "year_to_date",
        "balance": "as_of",
        "cash_flow": "year_to_date",
    }
    income = next(row for row in rows if row.statement == "income")
    assert income.announcement_date.isoformat() == "2026-10-31"
    assert {r.item_id: (r.name, str(r.value)) for r in income.lines} == {
        "TOTAL_OPERATE_INCOME": ("营业总收入", "133283652508.95"),
        "OPERATE_INCOME": ("营业收入", "130970743649.04"),
    }


_SEGMENT_NUMBERS = {
    "MAIN_BUSINESS_INCOME": "revenue",
    "MAIN_BUSINESS_COST": "cost",
    "MAIN_BUSINESS_RPOFIT": "gross_profit",
    "GROSS_RPOFIT_RATIO": "gross_margin_pct",
    "MBI_RATIO": "revenue_share_pct",
}


def _segment_row(**changes):
    symbol = changes.get("SECUCODE", "300750.SZ")
    return {
        "SECUCODE": symbol,
        "SECURITY_CODE": symbol[:6] if isinstance(symbol, str) else None,
        "REPORT_DATE": "2026-06-30 00:00:00",
        "MAINOP_TYPE": "2",
        "ITEM_NAME": "储能电池系统",
        "MAIN_BUSINESS_INCOME": "53260967000",
        "MAIN_BUSINESS_COST": "40500853000",
        "MAIN_BUSINESS_RPOFIT": "12760114000",
        "GROSS_RPOFIT_RATIO": "0.239577",
        "MBI_RATIO": "0.25",
        **changes,
    }


@pytest.fixture
def segment_server(monkeypatch):
    stock_calls = []

    def require_stock(symbol):
        stock_calls.append(symbol)
        if symbol == "000001.SH":
            raise ValueError("需要股票代码")

    monkeypatch.setattr(fundamentals, "require_stock", require_stock)
    monkeypatch.setattr(
        fundamentals_sources, "call_infoway", lambda *_: pytest.fail("unexpected Infoway call")
    )
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: pytest.fail("unexpected external HTTP request")
    )
    server = StrictToolsMCP("segment-revenue-test")
    register_tools(server)
    return server, stock_calls


@pytest.fixture
def segment_source(monkeypatch):
    payload = {"zygcfx": []}
    calls = []

    def request(url, params):
        calls.append((url, params))
        return payload

    monkeypatch.setattr(fundamentals_sources, "_eastmoney", request)
    return payload, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol,code", [("300750.SZ", "SZ300750"), ("600519.SH", "SH600519")])
@pytest.mark.parametrize("dimension,kind", [("industry", "1"), ("product", "2"), ("region", "3")])
async def test_segment_query_selects_latest_period_and_one_dimension(
    segment_server, segment_source, symbol, code, dimension, kind
):
    server, stock_calls = segment_server
    payload, calls = segment_source
    # Deliberately interleave periods and same-named segments from all dimensions.
    payload["zygcfx"] = [
        _segment_row(SECUCODE=symbol, MAINOP_TYPE=k, ITEM_NAME=name, MAIN_BUSINESS_INCOME=amount)
        for name in ("乙项", "甲项", "丙项")
        for k, amount in (("3", "300"), ("1", "100"), ("2", "200"))
    ]
    payload["zygcfx"].insert(2, _segment_row(SECUCODE=symbol, REPORT_DATE="2025-12-31"))
    _, result = await server.call_tool(
        "get_segment_revenue", {"symbol": symbol, "dimension": dimension}
    )
    assert result["available"] is True and result["reason"] == ""
    assert result["domain"] == "fundamentals" and result["as_of"]
    assert [row["segment_name"] for row in result["items"]] == ["丙项", "乙项", "甲项"]
    assert {(row["symbol"], row["report_date"], row["dimension"], row["revenue"])
            for row in result["items"]} == {(symbol, "2026-06-30", dimension, {"1": "100", "2": "200", "3": "300"}[kind])}
    assert stock_calls == [symbol]
    assert calls == [(
        "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax",
        {"code": code},
    )]


@pytest.mark.asyncio
@pytest.mark.parametrize("date_args", [{}, {"report_date": None}, {"report_date": "2026-06-30"}])
async def test_segment_query_defaults_to_product_without_truncating(
    segment_server, segment_source, date_args
):
    server, _ = segment_server
    payload, _ = segment_source
    payload["zygcfx"] = [
        _segment_row(ITEM_NAME=f"产品{i:02d}") for i in reversed(range(25))
    ] + [_segment_row(REPORT_DATE="2025-12-31", ITEM_NAME="旧产品")]
    _, result = await server.call_tool("get_segment_revenue", {"symbol": "300750.SZ", **date_args})
    assert [row["segment_name"] for row in result["items"]] == [f"产品{i:02d}" for i in range(25)]
    assert {row["dimension"] for row in result["items"]} == {"product"}


@pytest.mark.asyncio
async def test_segment_query_exact_historical_period_does_not_invent_storage(
    segment_server, segment_source
):
    server, _ = segment_server
    payload, _ = segment_source
    payload["zygcfx"] = [
        _segment_row(),
        {
            "SECUCODE": "300750.SZ", "SECURITY_CODE": "300750",
            "REPORT_DATE": "2024-02-29", "MAINOP_TYPE": "2",
            "ITEM_NAME": "动力电池", "MAIN_BUSINESS_INCOME": "9007199254740993.123456789",
        },
    ]
    _, result = await server.call_tool(
        "get_segment_revenue", {"symbol": "300750.SZ", "report_date": "2024-02-29"}
    )
    assert result["available"] is True and result["reason"] == ""
    assert result["items"] == [{
        "symbol": "300750.SZ", "report_date": "2024-02-29", "dimension": "product",
        "segment_name": "动力电池", "revenue": "9007199254740993.123456789",
        "cost": None, "gross_profit": None, "gross_profit_basis": None,
        "gross_margin_pct": None, "revenue_share_pct": None,
        "currency": None, "unit": "元", "source": "eastmoney",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,args,reason",
    [
        ([], {}, "segment_revenue_unavailable"),
        ([], {"report_date": "2025-12-31"}, "segment_revenue_unavailable"),
        ([_segment_row()], {"report_date": "2026-03-31"},
         "segment_revenue_unavailable_for_report_date:2026-03-31"),
        ([_segment_row(MAINOP_TYPE="3"), _segment_row(REPORT_DATE="2025-12-31")], {},
         "segment_revenue_unavailable_for_dimension:2026-06-30:product"),
        ([_segment_row(), _segment_row(REPORT_DATE="2025-12-31")],
         {"report_date": "2025-12-31", "dimension": "industry"},
         "segment_revenue_unavailable_for_dimension:2025-12-31:industry"),
    ],
    ids=["empty", "empty-with-date", "missing-period", "latest-before-dimension", "missing-dimension"],
)
async def test_segment_query_stable_unavailable_reasons(segment_server, segment_source, rows, args, reason):
    server, _ = segment_server
    payload, calls = segment_source
    payload["zygcfx"] = rows
    _, result = await server.call_tool("get_segment_revenue", {"symbol": "300750.SZ", **args})
    assert result["available"] is False
    assert result["reason"] == reason and result["items"] == []
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {}, {"symbol": None}, {"symbol": 300750}, {"symbol": True},
        {"symbol": "300750"}, {"symbol": "300750.sz"}, {"symbol": "920000.BJ"},
        *({"symbol": "300750.SZ", "report_date": value} for value in (
            20260630, True, "20260630", "2026-6-30", "2026-02-29", "2026-06-31",
            "2026-06-30 00:00:00", "2026-06-30junk",
        )),
        *({"symbol": "300750.SZ", "dimension": value} for value in ("all", "Product", 2, None)),
        {"symbol": "300750.SZ", "limit": 1},
        {"symbol": "300750.SZ", "keyword": "储能"},
    ],
)
async def test_segment_invalid_arguments_fail_before_stock_lookup_and_source(
    segment_server, segment_source, arguments
):
    server, stock_calls = segment_server
    _, calls = segment_source
    with pytest.raises(ToolError) as error:
        await server.call_tool("get_segment_revenue", arguments)
    payload = json.loads(str(error.value))["error"]
    assert payload["code"] == "invalid_argument" and payload["retryable"] is False
    assert stock_calls == [] and calls == []


@pytest.mark.asyncio
async def test_segment_index_is_rejected_before_fetch(segment_server, segment_source):
    server, stock_calls = segment_server
    _, calls = segment_source
    with pytest.raises(ToolError) as error:
        await server.call_tool("get_segment_revenue", {"symbol": "000001.SH"})
    assert json.loads(str(error.value))["error"]["code"] == "invalid_argument"
    assert stock_calls == ["000001.SH"] and calls == []


@pytest.mark.parametrize("symbol", ["300750.SZ", "600519.SH"])
def test_segment_adapter_maps_disclosed_storage_amounts(segment_source, symbol):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(SECUCODE=symbol)]
    (row,) = fundamentals_sources.fetch_segment_revenue(symbol)
    assert row.model_dump(mode="json", exclude={"gross_margin_pct", "revenue_share_pct"}) == {
        "symbol": symbol, "report_date": "2026-06-30", "dimension": "product",
        "segment_name": "储能电池系统", "revenue": "53260967000", "cost": "40500853000",
        "gross_profit": "12760114000", "gross_profit_basis": "source",
        "currency": None, "unit": "元", "source": "eastmoney",
    }
    assert row.gross_margin_pct == Decimal("23.9577")
    assert row.revenue_share_pct == Decimal("25")
    assert all(isinstance(getattr(row, field), Decimal) for field in _SEGMENT_NUMBERS.values())


@pytest.mark.parametrize(
    "income,cost,profit,expected,basis",
    [
        ("10.25", "2.10", "7.123456789", "7.123456789", "source"),
        ("10.25", "2.10", None, "8.15", "revenue_minus_cost"),
        ("10", "2", "0", "0", "source"),
        ("0", "0", "--", "0", "revenue_minus_cost"),
        ("-1.25", "2.5", "-", "-3.75", "revenue_minus_cost"),
        ("9007199254740993.123456789", "0.000000001", "", "9007199254740993.123456788", "revenue_minus_cost"),
        (None, "0", None, None, None),
    ],
)
def test_segment_profit_basis_preserves_precision_and_does_not_derive_margin(
    segment_source, income, cost, profit, expected, basis
):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(
        MAIN_BUSINESS_INCOME=income, MAIN_BUSINESS_COST=cost, MAIN_BUSINESS_RPOFIT=profit,
        GROSS_RPOFIT_RATIO=None, MBI_RATIO="-0.0123456789",
    )]
    (row,) = fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert row.revenue == (Decimal(income) if income is not None else None)
    assert row.cost == Decimal(cost)
    assert row.gross_profit == (Decimal(expected) if expected is not None else None)
    assert row.gross_profit_basis == basis
    assert row.gross_margin_pct is None
    assert row.revenue_share_pct == Decimal("-1.23456789")


def test_segment_arithmetic_extends_beyond_default_decimal_precision(segment_source):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(
        MAIN_BUSINESS_INCOME="123456789012345678901234567890",
        MAIN_BUSINESS_COST="1",
        MAIN_BUSINESS_RPOFIT=None,
        GROSS_RPOFIT_RATIO="0.1234567890123456789012345678901",
        MBI_RATIO="0.1234567890123456789012345678901",
    )]
    (row,) = fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert row.gross_profit == Decimal("123456789012345678901234567889")
    assert row.gross_profit_basis == "revenue_minus_cost"
    assert row.gross_margin_pct == Decimal("12.34567890123456789012345678901")
    assert row.revenue_share_pct == Decimal("12.34567890123456789012345678901")


def test_segment_source_arithmetic_leaves_global_decimal_context_unchanged(segment_source):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(
        MAIN_BUSINESS_INCOME="123456789012345678901234567890",
        MAIN_BUSINESS_COST="1",
    )]
    before = decimal.getcontext().copy()
    fundamentals_sources.fetch_segment_revenue("300750.SZ")
    after = decimal.getcontext()
    assert (after.prec, after.Emax, after.Emin, after.rounding, after.capitals) == (
        before.prec, before.Emax, before.Emin, before.rounding, before.capitals,
    )
    assert after.traps == before.traps and after.flags == before.flags


@pytest.mark.parametrize("missing", [None, "", "-", "--", "omitted"])
def test_segment_missing_numbers_are_null_not_zero(segment_source, missing):
    payload, _ = segment_source
    raw = _segment_row(**dict.fromkeys(_SEGMENT_NUMBERS, missing))
    if missing == "omitted":
        for field in _SEGMENT_NUMBERS:
            del raw[field]
    payload["zygcfx"] = [raw]
    (row,) = fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert all(getattr(row, field) is None for field in _SEGMENT_NUMBERS.values())
    assert row.gross_profit_basis is None and row.currency is None


def test_segment_omitted_profit_uses_only_its_own_row(segment_source):
    payload, _ = segment_source
    calculable = _segment_row(ITEM_NAME="完整成本", REPORT_DATE="2024-02-29 23:59:59")
    del calculable["MAIN_BUSINESS_RPOFIT"]
    revenue_only = _segment_row(ITEM_NAME="仅收入", MAIN_BUSINESS_INCOME="100")
    del revenue_only["MAIN_BUSINESS_RPOFIT"]
    del revenue_only["MAIN_BUSINESS_COST"]
    payload["zygcfx"] = [calculable, revenue_only]
    rows = {row.segment_name: row for row in fundamentals_sources.fetch_segment_revenue("300750.SZ")}
    assert rows["完整成本"].report_date == date(2024, 2, 29)
    assert rows["完整成本"].gross_profit == Decimal("12760114000")
    assert rows["完整成本"].gross_profit_basis == "revenue_minus_cost"
    assert rows["仅收入"].revenue == Decimal("100")
    assert rows["仅收入"].cost is None and rows["仅收入"].gross_profit is None
    assert rows["仅收入"].gross_profit_basis is None


@pytest.mark.parametrize("field", _SEGMENT_NUMBERS)
@pytest.mark.parametrize("value", [True, False, "NaN", float("inf"), Decimal("-Infinity"), "乱码"])
def test_segment_adapter_rejects_invalid_numbers(segment_source, field, value):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(**{field: value})]
    with pytest.raises(source_requests.SourceError) as error:
        fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"


@pytest.mark.parametrize(
    "raw",
    [
        {}, {"zygcfx": None}, {"zygcfx": {}}, {"zygcfx": ""},
        *({"zygcfx": [row]} for row in (None, [], "bad-row", {})),
        *({"zygcfx": [_segment_row(**changes)]} for changes in (
            {"SECUCODE": "600519.SH"}, {"SECUCODE": None},
            {"MAINOP_TYPE": "4"}, {"MAINOP_TYPE": 2}, {"MAINOP_TYPE": None},
            {"ITEM_NAME": ""}, {"ITEM_NAME": "  "}, {"ITEM_NAME": 123},
        )),
    ],
)
def test_segment_adapter_rejects_malformed_containers_and_rows(segment_source, raw):
    payload, calls = segment_source
    payload.clear()
    payload.update(raw)
    with pytest.raises(source_requests.SourceError) as error:
        fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument" and len(calls) == 1


@pytest.mark.parametrize(
    "raw_date",
    ["2026-02-29", "2026-06-31", "20260630", "2026-6-30", "2026-06-30junk",
     "2026-06-30T00:00:00", "2026-06-30 24:00:00", "2026-06-30 00:00:00+08:00", None],
)
def test_segment_adapter_requires_exact_valid_source_date(segment_source, raw_date):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(REPORT_DATE=raw_date)]
    with pytest.raises(source_requests.SourceError) as error:
        fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert error.value.source == "eastmoney" and error.value.retryable is False


@pytest.mark.asyncio
async def test_segment_bad_row_outside_requested_slice_is_not_silently_skipped(
    segment_server, segment_source
):
    server, _ = segment_server
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(), _segment_row(
        REPORT_DATE="2025-12-31", MAINOP_TYPE="3", MAIN_BUSINESS_COST="response-secret",
    )]
    with pytest.raises(ToolError) as error:
        await server.call_tool("get_segment_revenue", {"symbol": "300750.SZ"})
    details = json.loads(str(error.value))["error"]
    assert details["source"] == "eastmoney" and details["retryable"] is False
    assert details["code"] != "invalid_argument"
    assert "response-secret" not in str(error.value)


def test_segment_deduplicates_only_identical_normalized_rows(segment_source):
    payload, _ = segment_source
    row = _segment_row()
    payload["zygcfx"] = [
        row, dict(row),
        _segment_row(REPORT_DATE="2026-06-30", MAIN_BUSINESS_INCOME=53260967000, RANK=99),
        _segment_row(MAINOP_TYPE="1"),
        _segment_row(REPORT_DATE="2025-12-31"),
    ]
    rows = fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert len(rows) == 3
    assert {(r.report_date.isoformat(), r.dimension, r.segment_name) for r in rows} == {
        ("2026-06-30", "product", "储能电池系统"),
        ("2026-06-30", "industry", "储能电池系统"),
        ("2025-12-31", "product", "储能电池系统"),
    }


@pytest.mark.parametrize(
    "changes", [{field: "0"} for field in _SEGMENT_NUMBERS] + [{"MAIN_BUSINESS_RPOFIT": None}],
    ids=[*_SEGMENT_NUMBERS.values(), "same-profit-different-basis"],
)
def test_segment_conflicting_duplicate_is_an_error(segment_source, changes):
    payload, _ = segment_source
    payload["zygcfx"] = [_segment_row(), _segment_row(**changes)]
    with pytest.raises(source_requests.SourceError) as error:
        fundamentals_sources.fetch_segment_revenue("300750.SZ")
    assert error.value.source == "eastmoney" and error.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses,body,code,retryable,waits",
    [
        ([503, 200], None, None, None, [1]),
        ([429, 200], None, None, None, [4]),
        ([503, 503, 503], None, "upstream_unavailable", True, [1, 2]),
        ([403], None, "permission_denied", False, []),
        ([200], "response-secret: not JSON", "format", False, []),
        ([200], '["response-secret"]', "format", False, []),
        ([200], '{"zygcfx": "response-secret"}', "format", False, []),
        ([200], '{"zygcfx": []}', "empty", None, []),
    ],
    ids=["503-recovery", "429-retry-after", "503-exhaustion", "403-no-retry",
         "invalid-json", "invalid-json-root", "invalid-container", "empty-no-retry"],
)
async def test_segment_http_uses_shared_retry_and_safe_mcp_errors(
    segment_server, monkeypatch, caplog, statuses, body, code, retryable, waits
):
    server, _ = segment_server
    attempts, actual_waits = [], []
    monkeypatch.setenv("MCP_SOURCE_MAX_RETRIES", "2")
    monkeypatch.setenv("MCP_CALL_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(source_requests.CallBudget, "wait", lambda self, delay: actual_waits.append(delay))
    monkeypatch.setattr(source_requests.random, "uniform", lambda *_: 0)
    # Unquoted JSON decimals catch accidental binary-float parsing at the HTTP boundary.
    success_body = (
        '{"zygcfx":[{"SECUCODE":"300750.SZ","SECURITY_CODE":"300750",'
        '"REPORT_DATE":"2026-06-30","MAINOP_TYPE":"2","ITEM_NAME":"储能电池系统",'
        '"MAIN_BUSINESS_INCOME":9007199254740993.123456789,'
        '"GROSS_RPOFIT_RATIO":0.239577}]}'
    )

    def handle(request):
        status = statuses[len(attempts)]
        attempts.append(request)
        assert request.url.path == "/PC_HSF10/BusinessAnalysis/PageAjax"
        assert dict(request.url.params) == {"code": "SZ300750"}
        return httpx.Response(
            status,
            text=(body if body is not None else success_body) if status == 200 else "response-secret",
            headers={"Retry-After": "4"} if status == 429 else {},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(httpx, "get", client.get)
        if code in (None, "empty"):
            _, result = await server.call_tool("get_segment_revenue", {"symbol": "300750.SZ"})
            if code == "empty":
                assert result["available"] is False and result["items"] == []
                assert result["reason"] == "segment_revenue_unavailable"
            else:
                assert result["available"] is True
                (row,) = result["items"]
                assert row["revenue"] == "9007199254740993.123456789"
                assert isinstance(row["gross_margin_pct"], str)
                assert Decimal(row["gross_margin_pct"]) == Decimal("23.9577")
                assert row["gross_profit"] is None and row["gross_profit_basis"] is None
        else:
            with pytest.raises(ToolError) as error:
                await server.call_tool("get_segment_revenue", {"symbol": "300750.SZ"})
            details = json.loads(str(error.value))["error"]
            assert details["source"] == "eastmoney"
            assert details["retryable"] is retryable
            assert details["attempt_count"] == len(statuses)
            if code == "format":
                assert details["code"] != "invalid_argument"
            else:
                assert details["code"] == code
            assert "response-secret" not in str(error.value)
    assert len(attempts) == len(statuses) and actual_waits == waits
    assert "response-secret" not in caplog.text
