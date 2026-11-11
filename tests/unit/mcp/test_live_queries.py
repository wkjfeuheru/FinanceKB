from datetime import date, datetime

import httpx
import pytest

from src.queries import fundamentals, market_sources, news
from src.queries.content_types import NewsArticle
from src.queries.fundamentals_types import TradingCalendar, ValuationMetric
from src.queries.market_types import Instrument


@pytest.mark.asyncio
async def test_valuation_reads_latest_closed_session_without_a_snapshot(monkeypatch):
    monkeypatch.setattr(fundamentals, "require_stock", lambda _: None)
    monkeypatch.setattr(
        fundamentals, "shanghai_now", lambda: datetime.fromisoformat("2026-11-10T19:30:00+08:00")
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_trading_calendar",
        lambda start, end: TradingCalendar(
            start_date=start,
            end_date=end,
            trade_days=[date(2026, 11, 10)],
        ),
    )
    monkeypatch.setattr(
        fundamentals,
        "fetch_valuation_metrics",
        lambda _: [
            ValuationMetric(
                symbol="000858.SZ", metric="pe", value=21, trade_date=date(2026, 11, 9)
            ),
            ValuationMetric(
                symbol="000858.SZ", metric="pe", value=22, trade_date=date(2026, 11, 10)
            ),
        ],
    )
    result = await fundamentals.get_valuation_metrics("000858.SZ")
    assert result.available and [item.value for item in result.items] == [22]


def _denied(*args, **kwargs):
    raise market_sources.MarketSourceError("Infoway 请求失败（错误码 514）")


def test_infoway_quote_permission_falls_back_to_sina(monkeypatch):
    fields = ["五粮液", "70", "71", "72"] + ["0"] * 26 + ["2026-11-10", "15:35:45"]
    body = ('var hq_str_sz000858="' + ",".join(fields) + '";').encode("gb18030")
    monkeypatch.setattr(market_sources, "_infoway", _denied)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200,
            content=body,
            request=httpx.Request("GET", url),
        ),
    )
    (row,) = market_sources.fetch_quotes([{"symbol": "000858.SZ", "instrument_type": "stock"}])
    assert (row["symbol"], row["source"], row["value"]) == ("000858.SZ", "sina", 72)
    assert row["source_time"].isoformat() == "2026-11-10T15:35:45+08:00"


def test_infoway_kline_permission_falls_back_to_unadjusted_eastmoney(monkeypatch):
    monkeypatch.setattr(market_sources, "_stock_bars", _denied)

    def eastmoney(url, params):
        assert params["secid"] == "0.000858" and params["fqt"] == 0
        return {"market": 0, "code": "000858", "klines": ["2026-11-10,72,74,75,71,301,2200000"]}

    monkeypatch.setattr(market_sources, "_eastmoney", eastmoney)
    instrument = Instrument(
        symbol="000858.SZ",
        name="五粮液",
        exchange="SZSE",
        instrument_type="stock",
        source="infoway",
    )
    day = date(2026, 11, 10)
    (bar,) = market_sources.fetch_bars(instrument, day, day, "1d")
    assert (bar.source, bar.volume, bar.volume_unit) == ("eastmoney", 30_100, "share")


@pytest.mark.asyncio
async def test_news_can_search_a_keyword_without_a_saved_scope(monkeypatch):
    monkeypatch.setattr(
        news,
        "fetch_news",
        lambda client, keyword: [
            NewsArticle(article_id="1", title=keyword, published_at="2026-11-10T15:00:00+08:00"),
        ],
    )
    result = await news.search_news("新关键词", "2026-11-10", "2026-11-10")
    assert result.available and [item.title for item in result.items] == ["新关键词"]
