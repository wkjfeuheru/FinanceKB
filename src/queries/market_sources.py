"""东财 K 线与新浪快照的响应校验与归一化（行情唯一链路，无降级源）。"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from src.queries.market_types import INDEX_INTERVALS, MAX_BARS, Bar, Instrument, Quote

from .source_requests import SourceError as MarketSourceError
from .source_requests import get

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_INDICES = {
    "000001.SH": ("1.000001", "sh000001", "SSE"),
    "399001.SZ": ("0.399001", "sz399001", "SZSE"),
    "000300.CSI": ("1.000300", "sh000300", "CSI"),
}
_STOCK_EASTMONEY_INTERVALS = {
    **INDEX_INTERVALS,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
}
_INDEX_NAMES = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "000300.CSI": "沪深300"}
_FORMAT_ERRORS = (KeyError, TypeError, ValueError, AttributeError, OverflowError, OSError)


class KlineLimitError(ValueError):
    """请求的 K 线数量超出工具上限。"""


def _eastmoney(url: str, params: dict[str, str | int]) -> Any:
    try:
        response = get(
            url, source="eastmoney",
            params=params,
            timeout=8.0,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("rc") != 0:
            raise MarketSourceError("东财请求失败")
        return payload["data"]
    except (httpx.HTTPError, ValueError, KeyError, AttributeError):
        raise MarketSourceError("东财请求失败或响应格式异常") from None


def _eastmoney_stock_lookup(keyword: str) -> list[Instrument]:
    try:
        response = get(
            "https://searchapi.eastmoney.com/api/suggest/get", source="eastmoney",
            params={
                "input": keyword.split(".")[0],
                "type": 14,
                "count": 30,
                "token": "D43BF722C8E33BDC906D9D85E326E8E5",
            },
            timeout=8.0,
        )
        response.raise_for_status()
        table = response.json()["QuotationCodeTable"]
        rows = table["Data"]
        if rows is None and table["Status"] == 0:
            return []
        return [
            Instrument(
                symbol=f"{row['Code']}.{'SH' if row['MktNum'] == '1' else 'SZ'}",
                name=row["Name"],
                exchange="SSE" if row["MktNum"] == "1" else "SZSE",
                instrument_type="stock",
                source="eastmoney",
            )
            for row in rows
            if row["Classify"] == "AStock" and row["MktNum"] in ("0", "1")
        ]
    except (httpx.HTTPError, *_FORMAT_ERRORS):
        raise MarketSourceError("东财股票搜索请求失败或响应格式异常") from None


def lookup_instruments(keyword: str) -> list[Instrument]:
    query = "".join(keyword.split()).casefold()
    indices = [
        Instrument(
            symbol=symbol,
            name=_INDEX_NAMES[symbol],
            exchange=values[2],
            instrument_type="index",
            source="eastmoney",
        )
        for symbol, values in _INDICES.items()
        if query in (symbol.casefold(), symbol[:6], _INDEX_NAMES[symbol].casefold())
    ]
    if indices and not query.isdecimal():
        return indices
    try:
        stocks = _eastmoney_stock_lookup(query)
    except MarketSourceError:
        stocks = []
    candidates = {row.symbol: row for row in indices}
    for row in stocks:
        if query in (row.symbol.casefold(), row.symbol[:6], "".join(row.name.split()).casefold()):
            candidates[row.symbol] = row
    return [candidates[symbol] for symbol in sorted(candidates)]


def require_instrument(symbol: str) -> Instrument:
    items = lookup_instruments(symbol)
    if not items:
        raise ValueError("symbol 未收录，请用 resolve_symbol 查询标准代码")
    return items[0]


def require_stock(symbol: str) -> None:
    if require_instrument(symbol).instrument_type != "stock":
        raise ValueError("symbol 必须是 resolve_symbol 已收录的沪深股票标准代码")


def require_eastmoney_stock(symbol: str) -> None:
    if not any(row.symbol == symbol for row in _eastmoney_stock_lookup(symbol)):
        raise ValueError("symbol 必须是东财收录的沪深股票标准代码")


def fetch_bars(instrument: Instrument, start: date, end: date, interval: str) -> list[Bar]:
    stock = instrument.instrument_type == "stock"
    if not stock and interval not in INDEX_INTERVALS:
        raise ValueError("指数 interval 只能取 1d、1w、1M")
    try:
        return _eastmoney_bars(instrument.symbol, start, end, interval, stock=stock)
    except _FORMAT_ERRORS:
        raise MarketSourceError("东财 K 线响应格式异常") from None


def _eastmoney_bars(
    symbol: str, start: date, end: date, interval: str, *, stock: bool
) -> list[Bar]:
    secid = f"{int(symbol.endswith('.SH'))}.{symbol[:6]}" if stock else _INDICES[symbol][0]
    base = "https://push2his.eastmoney.com/api/qt/stock/"
    if interval == "1m":
        data = _eastmoney(
            base + "trends2/get",
            {
                "secid": secid,
                "ndays": 5,
                "iscr": 0,
                "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            },
        )
    else:
        data = _eastmoney(
            base + "kline/get",
            {
                "secid": secid,
                "klt": _STOCK_EASTMONEY_INTERVALS[interval],
                "fqt": 0,
                "beg": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
                "lmt": MAX_BARS + 1,
            },
        )
    if data is None:
        return []
    market, code = secid.split(".")
    if data["code"] != code or str(data["market"]) != market:
        raise MarketSourceError("东财返回了与请求不符的标的")
    rows = data["trends" if interval == "1m" else "klines"]
    bars: dict[datetime, Bar] = {}
    if interval == "1m":
        if rows and start < date.fromisoformat(rows[0][:10]):
            raise MarketSourceError("东财 1 分钟线的可用历史不覆盖起始日期")
    elif len(rows) > MAX_BARS:
        raise KlineLimitError(f"K 线超过 {MAX_BARS} 条，请缩小日期范围")
    for row in rows:
        timestamp, open_, close, high, low, volume, turnover = row.split(",")[:7]
        when = datetime.fromisoformat(timestamp).replace(tzinfo=_SHANGHAI)
        trade_date = when.date()
        if start <= trade_date <= end:
            bars[when] = Bar(
                symbol=symbol,
                instrument_type="stock" if stock else "index",
                interval=interval,
                trade_date=trade_date,
                time=when,
                open=open_,
                close=close,
                high=high,
                low=low,
                volume=float(volume) * 100 if stock else float(volume),
                volume_unit="share" if stock else "lot",
                price_unit="CNY" if stock else "point",
                turnover=turnover,
                source="eastmoney",
            )
    return [bars[when] for when in sorted(bars)]


def fetch_quotes(instruments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stocks = [row["symbol"] for row in instruments if row["instrument_type"] == "stock"]
    indices = [row["symbol"] for row in instruments if row["instrument_type"] == "index"]
    mapping = {f"{symbol[-2:].lower()}{symbol[:6]}": (symbol, "stock") for symbol in stocks}
    mapping.update({_INDICES[symbol][1]: (symbol, "index") for symbol in indices})
    if not mapping:
        return []
    try:
        quotes = _sina_quotes(mapping)
    except _FORMAT_ERRORS:
        raise MarketSourceError("新浪快照响应格式异常") from None
    return [quote.model_dump() for quote in quotes]


def _sina_quotes(mapping: dict[str, tuple[str, str]]) -> list[Quote]:
    try:
        response = get(
            "https://hq.sinajs.cn/list=" + ",".join(mapping), source="sina",
            timeout=8.0,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        )
        response.raise_for_status()
        body = response.content.decode("gb18030")
    except (httpx.HTTPError, UnicodeError):
        raise MarketSourceError("新浪快照请求失败") from None
    quotes = []
    if "hq_str_" not in body:
        raise MarketSourceError("新浪快照响应格式异常")
    for code, raw in re.findall(r'hq_str_(\w+)="([^"]*)"', body):
        if code not in mapping or not raw:
            continue
        fields = raw.split(",")
        if len(fields) < 32:
            raise MarketSourceError("新浪快照缺少行情时间")
        symbol, kind = mapping[code]
        point, prior = float(fields[3]), float(fields[2])
        if point <= 0:
            continue
        quotes.append(
            Quote(
                symbol=symbol,
                instrument_type=kind,
                value=point,
                value_unit="CNY" if kind == "stock" else "point",
                change_pct=(point / prior - 1) * 100 if prior else None,
                source_time=datetime.fromisoformat(f"{fields[30]}T{fields[31]}").replace(
                    tzinfo=_SHANGHAI
                ),
                source="sina",
            )
        )
    return quotes
