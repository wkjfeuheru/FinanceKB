from datetime import date
from decimal import Decimal

import httpx
import pytest

from src.queries import macro_sources, news_sources


def test_eastmoney_macro_uses_period_end_and_cumulative_gdp():
    raw = {
        "REPORT_DATE": "2026-09-01 00:00:00",
        "TIME": "2026年第1-3季度",
        "DOMESTICL_PRODUCT_BASE": 1071071.2,
        "SUM_SAME": 4.6,
        "FIRST_PRODUCT_BASE": 58993.9,
        "FIRST_SAME": 3.7,
    }

    def response(request):
        assert request.url.params["reportName"] == "RPT_ECONOMY_GDP"
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {"pages": 1, "count": 1, "data": [raw]},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        (row,) = macro_sources.fetch_macro_indicator(client, "GDP")
    assert row.period_end == date(2026, 9, 30)
    assert row.publication_date is None
    assert {(m.series, m.metric, m.basis, m.value, m.unit) for m in row.metrics} == {
        ("national", "gdp", "year_to_date", Decimal("1071071.2"), "CNY_100_million"),
        ("national", "yoy_pct", "year_to_date", Decimal("4.6"), "percent"),
        ("primary", "gdp", "year_to_date", Decimal("58993.9"), "CNY_100_million"),
        ("primary", "yoy_pct", "year_to_date", Decimal("3.7"), "percent"),
    }


def test_nbs_calendar_reads_two_events_in_one_month_cell():
    header = ["序号", "内容", *(f"{month}月" for month in range(1, 13))]
    rows = [
        header,
        ["4", "采购经理指数月度报告", "31/六", "……", "4/三 <br> 注5 <br> 31/二", *(["……"] * 9)],
    ]
    html = (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</table>"
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=html + html))
    with httpx.Client(transport=transport) as client:
        events = macro_sources.fetch_economic_calendar(client, 2026)
    assert [event.planned_date for event in events] == [
        date(2026, 1, 31),
        date(2026, 3, 4),
        date(2026, 3, 31),
    ]


def test_eastmoney_earnings_deduplicates_appointment_and_actual_disclosure():
    row = {
        "SECUCODE": "600519.SH",
        "SECURITY_CODE": "600519",
        "SECURITY_NAME_ABBR": "贵州茅台",
        "REPORT_DATE": "2026-09-30 00:00:00",
        "FIRST_APPOINT_DATE": "2026-10-27 00:00:00",
        "FIRST_CHANGE_DATE": "2026-10-31 00:00:00",
        "SECOND_CHANGE_DATE": "2026-10-30 00:00:00",
        "THIRD_CHANGE_DATE": None,
        "APPOINT_PUBLISH_DATE": "2026-10-30 00:00:00",
        "ACTUAL_PUBLISH_DATE": "2026-10-31 00:00:00",
        "INFO_CODE": "AN202610301831245500",
    }
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "success": True,
                "result": {"pages": 1, "count": 1, "data": [row]},
            },
        )
    )
    with httpx.Client(transport=transport) as client:
        (event,) = news_sources.fetch_earnings_calendar(
            client,
            date(2026, 10, 30),
            date(2026, 10, 31),
        )
    assert event.current_appointment_date == date(2026, 10, 30)
    assert event.actual_disclosure_date == date(2026, 10, 31)


@pytest.mark.parametrize(
    "title",
    ["贵州茅台:贵州茅台2026年第三季度报告", "五粮液:2026年三季度报告", "2026 年三季度报告"],
)
def test_eastmoney_filings_uses_notice_date_and_title_report_period(title):
    row = {
        "art_code": "AN202610301831245500",
        "title": title,
        "codes": [{"stock_code": "600519", "market_code": "1"}],
        "columns": [{"column_name": "三季度报告全文"}],
        "notice_date": "2026-10-31 00:00:00",
        "display_time": "2026-10-30 18:31:37:306",
    }
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "success": 1,
                "data": {"total_hits": 1, "list": [row]},
            },
        )
    )
    with httpx.Client(transport=transport) as client:
        (filing,) = news_sources.fetch_filings(client, "600519.SH")
    assert filing.disclosure_date == date(2026, 10, 31)
    assert filing.report_date == date(2026, 9, 30)


def test_eastmoney_news_preserves_source_link_and_beijing_time():
    body = '{"code":0,"result":{"cmsArticleWebOld":[{"date":"2026-11-09 18:05:49",'
    body += '"code":"202611093926457689","title":"<em>半导体</em>新闻","mediaName":"来源媒体",'
    body += '"url":"http://stock.eastmoney.com/a/202611093926457689.html"}]}}'
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            text=f"{req.url.params['cb']}({body});",
        )
    )
    with httpx.Client(transport=transport) as client:
        (article,) = news_sources.fetch_news(client, "半导体")
    assert article.title == "半导体新闻"
    assert article.published_at.isoformat() == "2026-11-09T18:05:49+08:00"
    assert str(article.url) == "http://stock.eastmoney.com/a/202611093926457689.html"
