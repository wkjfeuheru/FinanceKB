from __future__ import annotations

from datetime import date, datetime

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.mcp.common.server import StrictToolsMCP
from src.mcp.fundamentals.server import register_tools
from src.queries import fundamentals, fundamentals_sources
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
