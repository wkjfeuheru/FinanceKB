"""东财新闻、公司公告和财报预约的主源适配。"""

import json
import re
from calendar import monthrange
from datetime import date, datetime
from html import unescape
from zoneinfo import ZoneInfo

import httpx

from .content_http import ContentSourceError, eastmoney_page, fetch_text, json_object, source_date
from .content_types import EarningsAppointment, Filing, NewsArticle

_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
_NOTICES_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
_FILING_URL = "https://data.eastmoney.com/notices/detail/"
_REPORT_MONTHS = {
    "第一季度报告": 3,
    "一季度报告": 3,
    "一季报": 3,
    "半年度报告": 6,
    "半年报": 6,
    "第三季度报告": 9,
    "三季度报告": 9,
    "三季报": 9,
    "年度报告": 12,
    "度报告": 12,
    "报": 12,
}
_REPORT_TITLE = re.compile(
    rf"(20[0-9]{{2}})\s*年\s*({'|'.join(_REPORT_MONTHS)})(?=$|[\s(（:：的摘要全文])"
)
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def fetch_news(client: httpx.Client, keyword: str) -> list[NewsArticle]:
    callback = "financekbNews"
    search = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 10,
                "preTag": "<em>",
                "postTag": "</em>",
            },
        },
    }
    source = "东财新闻"
    text = (
        fetch_text(
            client,
            _NEWS_URL,
            source,
            {"cb": callback, "param": json.dumps(search, ensure_ascii=False)},
            referer="https://so.eastmoney.com/",
        )
        .strip()
        .removesuffix(";")
    )
    prefix = callback + "("
    if not text.startswith(prefix) or not text.endswith(")"):
        raise ContentSourceError("东财新闻 JSONP 响应格式异常")
    data = json_object(text[len(prefix) : -1], source)
    try:
        rows = data["result"]["cmsArticleWebOld"]
        if data["code"] != 0 or not isinstance(rows, list):
            raise ValueError
        result = []
        for row in rows:
            timestamp = datetime.fromisoformat(row["date"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=_SHANGHAI)
            title = unescape(re.sub(r"\(</?em>|</?em>\)|</?em>", "", row["title"])).strip()
            if not title or not row["code"]:
                raise ValueError
            result.append(
                NewsArticle(
                    article_id=row["code"],
                    title=title,
                    published_at=timestamp.astimezone(_SHANGHAI),
                    publisher=row.get("mediaName") or None,
                    url=row.get("url") or None,
                )
            )
        return result
    except (KeyError, TypeError, ValueError):
        raise ContentSourceError("东财新闻响应格式异常") from None


def _report_date(title: str) -> date | None:
    match = _REPORT_TITLE.search(title)
    if not match:
        return None
    year, month = int(match[1]), _REPORT_MONTHS[match[2]]
    return date(year, month, monthrange(year, month)[1])


def fetch_filings(client: httpx.Client, symbol: str) -> list[Filing]:
    code = symbol[:6]
    market = "1" if symbol.endswith(".SH") else "0"
    text = fetch_text(
        client,
        _NOTICES_URL,
        "东财公告",
        {
            "sr": "-1",
            "page_size": 100,
            "page_index": 1,
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "stock_list": code,
        },
        referer="https://data.eastmoney.com/notices/",
    )
    raw = json_object(text, "东财公告")
    try:
        if raw.get("success") != 1:
            raise ValueError
        data = raw["data"]
        total, rows = data["total_hits"], data["list"]
        if type(total) is not int or not isinstance(rows, list) or len(rows) != min(100, total):
            raise ValueError
        result = []
        for row in rows:
            if not any(
                security["stock_code"] == code and str(security["market_code"]) == market
                for security in row["codes"]
            ):
                raise ValueError
            filing_id = row["art_code"]
            if not re.fullmatch(r"AN[0-9]+", filing_id):
                raise ValueError
            title = row["title"].strip()
            if not title:
                raise ValueError
            result.append(
                Filing(
                    filing_id=filing_id,
                    symbol=symbol,
                    title=title,
                    categories=list(dict.fromkeys(col["column_name"] for col in row["columns"])),
                    disclosure_date=source_date(row["notice_date"]),
                    report_date=_report_date(title),
                    url=f"{_FILING_URL}{code}/{filing_id}.html",
                )
            )
        return result
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ContentSourceError("东财公告响应格式或标的异常") from None


def _earnings_rows(client: httpx.Client, field: str, start: date, end: date) -> list[dict]:
    report = "RPT_PUBLIC_BS_APPOIN"
    params = {
        "pageNumber": 1,
        "pageSize": 500,
        "sortColumns": f"{field},SECURITY_CODE,REPORT_DATE",
        "sortTypes": "1,1,1",
        "filter": (
            '(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")'
            f"({field}>='{start}')({field}<='{end} 23:59:59')"
        ),
    }
    pages, count, rows = eastmoney_page(client, report, params, allow_empty=True)
    if not 0 <= pages <= 20:
        raise ContentSourceError("东财财报预约超出查询页数上限，请缩小日期范围")
    for page in range(2, pages + 1):
        page_count, row_count, batch = eastmoney_page(
            client, report, {**params, "pageNumber": page}
        )
        if (page_count, row_count) != (pages, count):
            raise ContentSourceError("东财财报预约分页期间总数发生变化")
        rows.extend(batch)
    if len(rows) != count:
        raise ContentSourceError("东财财报预约分页记录不完整")
    return rows


def fetch_earnings_calendar(
    client: httpx.Client, start: date, end: date
) -> list[EarningsAppointment]:
    result: dict[tuple[str, date], EarningsAppointment] = {}
    try:
        for field in ("APPOINT_PUBLISH_DATE", "ACTUAL_PUBLISH_DATE"):
            seen: set[tuple[str, date]] = set()
            for row in _earnings_rows(client, field, start, end):
                symbol = row["SECUCODE"]
                report_date = source_date(row["REPORT_DATE"])
                matched_date = source_date(row[field])
                key = (symbol, report_date)
                if (
                    row["SECURITY_CODE"] != symbol[:6]
                    or report_date is None
                    or matched_date is None
                    or not start <= matched_date <= end
                    or key in seen
                ):
                    raise ValueError
                seen.add(key)
                first = source_date(row["FIRST_APPOINT_DATE"])
                changes = [
                    day
                    for name in ("FIRST_CHANGE_DATE", "SECOND_CHANGE_DATE", "THIRD_CHANGE_DATE")
                    if (day := source_date(row[name])) is not None
                ]
                actual = source_date(row["ACTUAL_PUBLISH_DATE"])
                info_code = row.get("INFO_CODE") or ""
                url = None
                if actual and re.fullmatch(r"AN[0-9]+", info_code):
                    url = f"{_FILING_URL}{symbol[:6]}/{info_code}.html"
                result[key] = EarningsAppointment(
                    symbol=symbol,
                    name=row["SECURITY_NAME_ABBR"],
                    report_date=report_date,
                    first_appointment_date=first,
                    change_dates=changes,
                    current_appointment_date=source_date(row["APPOINT_PUBLISH_DATE"]),
                    actual_disclosure_date=actual,
                    url=url,
                )
        return list(result.values())
    except (KeyError, TypeError, ValueError):
        raise ContentSourceError("东财财报预约响应格式、报告期或标的异常") from None
