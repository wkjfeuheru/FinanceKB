from datetime import datetime

import pytest

from src.mcp.common.server import StrictToolsMCP
from src.mcp.macro import server as macro
from src.mcp.news import server as news
from src.queries.content_types import (
    EarningsAppointment,
    EconomicEvent,
    Filing,
    MacroObservation,
    NewsArticle,
)


@pytest.mark.asyncio
async def test_macro_uses_observation_period_and_calendar_uses_planned_date(monkeypatch):
    fixtures = {
        "macro_indicator": {
            "GDP": {
                "2026-09-30": {
                    "indicator": "GDP",
                    "frequency": "quarterly",
                    "period_end": "2026-09-30",
                    "publication_date": None,
                    "metrics": [
                        {
                            "series": "national",
                            "metric": "gdp",
                            "basis": "year_to_date",
                            "value": "1071071.2",
                            "unit": "CNY_100_million",
                        }
                    ],
                    "source": "eastmoney",
                }
            }
        },
        "economic_calendar": {
            "2026": {
                "row": {
                    "event": "居民消费价格指数月度报告",
                    "planned_date": "2026-11-09",
                    "source": "nbs",
                    "url": "https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html",
                }
            }
        },
    }

    monkeypatch.setattr(
        macro.queries,
        "fetch_macro_indicator",
        lambda _, indicator: [
            MacroObservation.model_validate(row)
            for row in fixtures["macro_indicator"][indicator].values()
        ],
    )
    monkeypatch.setattr(
        macro.queries,
        "fetch_economic_calendar",
        lambda _, year: [
            EconomicEvent.model_validate(row)
            for row in fixtures["economic_calendar"][str(year)].values()
        ],
    )
    mcp = StrictToolsMCP("test")
    macro.register_tools(mcp)
    _, data = await mcp.call_tool(
        "get_macro_indicator",
        {
            "indicator": "GDP",
            "start_date": "2026-09-30",
            "end_date": "2026-09-30",
        },
    )
    assert data["available"] and data["items"][0]["metrics"][0]["basis"] == "year_to_date"
    _, data = await mcp.call_tool(
        "get_economic_calendar",
        {
            "start_date": "2026-11-09",
            "end_date": "2026-11-09",
        },
    )
    assert data["available"] and data["items"][0]["event"] == "居民消费价格指数月度报告"


@pytest.mark.asyncio
async def test_news_filters_source_results_and_marks_earnings_date_matches(monkeypatch):
    fixtures = {
        "news": {
            "半导体": {
                "1": {
                    "article_id": "1",
                    "title": "旧闻",
                    "published_at": "2026-11-02T10:00:00+08:00",
                },
                "2": {
                    "article_id": "2",
                    "title": "新闻",
                    "published_at": "2026-11-09T10:00:00+08:00",
                },
            }
        },
        "filings": {
            "600519.SH": {
                "AN1": {
                    "filing_id": "AN1",
                    "symbol": "600519.SH",
                    "title": "季报",
                    "categories": ["财务报告"],
                    "disclosure_date": "2026-10-31",
                    "report_date": "2026-09-30",
                    "url": "https://data.eastmoney.com/notices/detail/600519/AN1.html",
                }
            }
        },
        "earnings_calendar": {
            "2026-09-30": {
                "600519.SH": {
                    "symbol": "600519.SH",
                    "name": "贵州茅台",
                    "report_date": "2026-09-30",
                    "first_appointment_date": "2026-10-27",
                    "change_dates": ["2026-10-30"],
                    "current_appointment_date": "2026-10-30",
                    "actual_disclosure_date": "2026-10-31",
                },
            }
        },
    }

    def known_stock(symbol):
        assert symbol == "600519.SH"

    monkeypatch.setattr(
        news.queries,
        "fetch_news",
        lambda _, keyword: [
            NewsArticle.model_validate(row) for row in fixtures["news"][keyword].values()
        ],
    )
    monkeypatch.setattr(
        news.queries,
        "fetch_filings",
        lambda _, symbol: [
            Filing.model_validate(row) for row in fixtures["filings"][symbol].values()
        ],
    )
    monkeypatch.setattr(
        news.queries,
        "fetch_earnings_calendar",
        lambda *_: [
            EarningsAppointment.model_validate(row)
            for row in fixtures["earnings_calendar"]["2026-09-30"].values()
        ],
    )
    monkeypatch.setattr(news.queries, "require_eastmoney_stock", known_stock)
    monkeypatch.setattr(news.queries, "shanghai_now", lambda: datetime(2026, 11, 9), raising=False)
    mcp = StrictToolsMCP("test")
    news.register_tools(mcp)
    _, data = await mcp.call_tool("search_news", {"keyword": " 半导体 ", "limit": 1})
    assert data["available"] and [row["title"] for row in data["items"]] == ["新闻"]
    _, data = await mcp.call_tool(
        "search_news",
        {
            "keyword": "半导体",
            "start_date": "2026-11-08",
            "end_date": "2026-11-08",
        },
    )
    assert data["available"] and data["items"] == []
    _, data = await mcp.call_tool("get_filings", {"symbol": "600519.SH"})
    assert data["items"][0]["report_date"] == "2026-09-30"
    _, data = await mcp.call_tool(
        "get_earnings_calendar",
        {
            "start_date": "2026-10-31",
            "end_date": "2026-10-31",
        },
    )
    assert data["items"][0]["matched_date_types"] == ["actual_disclosure"]
    _, data = await mcp.call_tool(
        "get_earnings_calendar",
        {
            "start_date": "2026-10-30",
            "end_date": "2026-10-31",
        },
    )
    assert data["items"][0]["matched_date_types"] == ["current_appointment", "actual_disclosure"]
