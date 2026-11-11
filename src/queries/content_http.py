"""宏观与资讯主源的 HTTP 请求和响应校验。"""

import json
from datetime import date
from decimal import Decimal
from typing import Any

import httpx

from .source_requests import SourceError, get

_DATA_CENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_EASTMONEY = "https://data.eastmoney.com/"


ContentSourceError = SourceError


def fetch_text(
    client: httpx.Client,
    url: str,
    source: str,
    params: dict[str, str | int] | None = None,
    *,
    referer: str | None = None,
) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    return get(
        url, client=client, source=source, params=params, headers=headers, timeout=10.0
    ).text


def json_object(text: str, source: str) -> dict[str, Any]:
    try:
        raw = json.loads(text, parse_float=Decimal)
        if not isinstance(raw, dict):
            raise ValueError
        return raw
    except ValueError:
        raise ContentSourceError(f"{source} 响应格式异常") from None


def eastmoney_page(
    client: httpx.Client,
    report: str,
    params: dict[str, str | int],
    *,
    allow_empty: bool = False,
) -> tuple[int, int, list[dict[str, Any]]]:
    source = "东财 " + report
    text = fetch_text(
        client,
        _DATA_CENTER,
        source,
        {"reportName": report, "columns": "ALL", "source": "WEB", "client": "WEB", **params},
        referer=_EASTMONEY,
    )
    raw = json_object(text, source)
    if allow_empty and raw.get("code") == 9201 and raw.get("result") is None:
        return 0, 0, []
    if raw.get("success") is not True:
        raise ContentSourceError(f"{source} 查询失败")
    try:
        data = raw["result"]
        pages, count, rows = data["pages"], data["count"], data["data"]
        if type(pages) is not int or type(count) is not int or not isinstance(rows, list):
            raise ValueError
        if not (0 <= len(rows) <= count and (count == 0 or pages > 0)):
            raise ValueError
        return pages, count, rows
    except (KeyError, TypeError, ValueError):
        raise ContentSourceError(f"{source} 分页响应格式异常") from None


def source_date(value: str | None) -> date | None:
    if value in (None, "", "-", "--"):
        return None
    return date.fromisoformat(value[:10])
