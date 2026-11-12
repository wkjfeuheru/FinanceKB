"""东财商品期货目录定位与最新报价读取（期货唯一链路，无降级源）。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from zoneinfo import ZoneInfo

from src.queries.futures_types import (
    SYMBOL_PATTERN,
    CatalogCommodity,
    CatalogContract,
    CommodityPrice,
)

from .source_requests import SourceError, get

_CATALOG_URL = "https://futsse-static.eastmoney.com/redis"
_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_QUOTE_FIELDS = "f43,f57,f58,f59,f86"
_MARKET_EXCHANGES: dict[str, Literal["SHFE", "DCE", "CZCE", "INE", "GFEX"]] = {
    "113": "SHFE",
    "114": "DCE",
    "115": "CZCE",
    "142": "INE",
    "225": "GFEX",
}
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
_EMPTY_VALUES = (None, "", "-", "--")
_FORMAT_ERRORS = (KeyError, TypeError, ValueError, AttributeError, InvalidOperation)
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)


def fetch_commodity_price(symbol: str) -> CommodityPrice | None:
    """目录核实合约身份后读取东财最新报价；合约存在但缺价时返回 None。"""
    market_id, code = _split_symbol(symbol)
    exchange = _MARKET_EXCHANGES[market_id]
    commodity = _match_commodity(market_id, exchange, code)
    contract = _match_contract(market_id, exchange, commodity, code)
    price, updated = _quote(market_id, contract)
    if price is None:
        return None
    return CommodityPrice(
        symbol=f"{market_id}.{contract.code}",
        contract_code=contract.code,
        contract_name=contract.name,
        commodity_code=commodity.vcode,
        commodity_name=commodity.vname,
        exchange=exchange,
        contract_type="main_continuous" if code[-1:].casefold() == "m" else "delivery",
        price=price,
        source_time=updated,
    )


def _split_symbol(symbol: str) -> tuple[str, str]:
    if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(
            "symbol 必须是 113/114/115/142/225 之一的东财商品期货标识，如 225.lcm、225.lc2705、113.cum"
        )
    market_id, code = symbol.split(".", 1)
    return market_id, code


def _catalog_json(url: str, params: dict[str, str], message: str) -> Any:
    response = get(url, source="eastmoney", params=params, timeout=8.0, headers=_HEADERS)
    try:
        return json.loads(response.content, parse_float=Decimal)
    except ValueError:
        raise SourceError(message, source="eastmoney") from None


def _market_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.isdigit():
        return value
    raise ValueError


def _commodity(market_id: str, exchange: str, row: Any) -> CatalogCommodity:
    if not isinstance(row, dict):
        raise ValueError
    if _market_id(row["mktid"]) != market_id or row["mktshort"] != exchange:
        raise ValueError
    vcode, vname, vtype = row["vcode"], row["vname"], row["vtype"]
    if (
        not isinstance(vcode, str)
        or not vcode
        or not isinstance(vname, str)
        or not vname
        or not isinstance(vtype, str)
        or not vtype.isdigit()
    ):
        raise ValueError
    return CatalogCommodity(vcode=vcode, vname=vname, vtype=vtype)


def _contract(
    market_id: str, exchange: str, commodity: CatalogCommodity, row: Any
) -> CatalogContract:
    if not isinstance(row, dict):
        raise ValueError
    if _market_id(row["mktid"]) != market_id or row["mktshort"] != exchange:
        raise ValueError
    code, name = row["code"], row["name"]
    if (
        not isinstance(code, str)
        or not code
        or not isinstance(name, str)
        or not name
        or row["vcode"].casefold() != commodity.vcode.casefold()
        or row["vname"] != commodity.vname
    ):
        raise ValueError
    return CatalogContract(code=code, name=name)


def _decompose(code: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if code[-1:].casefold() == "m":
        pairs.append((code[:-1], code[-1:]))
    for length in (4, 3):
        suffix = code[-length:]
        prefix = code[:-length]
        if prefix and suffix.isdigit():
            pairs.append((prefix, suffix))
    return pairs


def _match_commodity(market_id: str, exchange: str, code: str) -> CatalogCommodity:
    rows = _catalog_json(
        _CATALOG_URL, {"msgid": market_id}, "东财期货商品目录响应不是有效JSON"
    )
    try:
        if not isinstance(rows, list) or not rows:
            raise ValueError
        commodities: dict[str, CatalogCommodity] = {}
        for row in rows:
            commodity = _commodity(market_id, exchange, row)
            key = commodity.vcode.casefold()
            if key in commodities and commodities[key] != commodity:
                raise ValueError
            commodities[key] = commodity
        prefixes = {prefix.casefold() for prefix, _suffix in _decompose(code)}
        matches = [row for row in commodities.values() if row.vcode.casefold() in prefixes]
    except _FORMAT_ERRORS:
        raise SourceError("东财期货商品目录响应格式异常或为空", source="eastmoney") from None
    if len(matches) > 1:
        raise SourceError("东财期货目录出现歧义品种匹配", source="eastmoney")
    if not matches:
        raise ValueError("当前目录未提供该品种或合约，可能已下市或到期")
    return matches[0]


def _match_contract(
    market_id: str, exchange: str, commodity: CatalogCommodity, code: str
) -> CatalogContract:
    rows = _catalog_json(
        _CATALOG_URL,
        {"msgid": f"{market_id}_{commodity.vtype}"},
        "东财期货合约目录响应不是有效JSON",
    )
    try:
        if not isinstance(rows, list) or not rows:
            raise ValueError
        contracts: dict[str, CatalogContract] = {}
        for row in rows:
            contract = _contract(market_id, exchange, commodity, row)
            key = contract.code.casefold()
            if key in contracts and contracts[key] != contract:
                raise ValueError
            contracts[key] = contract
        matched = contracts.get(code.casefold())
    except _FORMAT_ERRORS:
        raise SourceError("东财期货合约目录响应格式异常或为空", source="eastmoney") from None
    if matched is None:
        raise ValueError("当前目录未提供该合约，可能已到期或未上市")
    return matched


def _price(value: Any) -> Decimal | None:
    if value in _EMPTY_VALUES:
        return None
    if isinstance(value, bool):
        raise ValueError
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError
    return number


def _source_time(value: Any) -> datetime | None:
    if value in _EMPTY_VALUES:
        return None
    if isinstance(value, bool):
        raise ValueError
    seconds = Decimal(value) if isinstance(value, int) else Decimal(str(value))
    if not seconds.is_finite() or seconds < 0 or seconds != seconds.to_integral_value():
        raise ValueError
    try:
        return datetime.fromtimestamp(int(seconds), _SHANGHAI)
    except (ValueError, OSError, OverflowError):
        raise ValueError from None


def _quote(market_id: str, contract: CatalogContract) -> tuple[Decimal | None, datetime | None]:
    raw = _catalog_json(
        _QUOTE_URL,
        {"secid": f"{market_id}.{contract.code}", "fltt": "2", "fields": _QUOTE_FIELDS},
        "东财期货报价响应不是有效JSON",
    )
    try:
        if not isinstance(raw, dict) or raw.get("rc") != 0:
            raise ValueError
        data = raw["data"]
        if not isinstance(data, dict):
            raise ValueError
        if not isinstance(data.get("f57"), str) or data["f57"].casefold() != contract.code.casefold():
            raise ValueError
        if "f43" not in data:
            raise ValueError
        return _price(data["f43"]), _source_time(data.get("f86"))
    except _FORMAT_ERRORS:
        raise SourceError("东财期货报价响应格式或合约身份异常", source="eastmoney") from None


__all__ = ["fetch_commodity_price"]
