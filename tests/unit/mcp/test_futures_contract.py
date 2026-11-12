from __future__ import annotations

import copy
import json
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.mcp.common.server import StrictToolsMCP
from src.mcp.futures.server import register_tools
from src.queries import futures, futures_sources, source_requests

_CATALOG_URL = "https://futsse-static.eastmoney.com/redis"
_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_QUOTE_PARAMS = {"secid": "225.lcm", "fltt": "2", "fields": "f43,f57,f58,f59,f86"}
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _commodity_row(market_id, mktname, mktshort, vcode, vname, vtype, changes=None):
    row = {
        "mktid": market_id,
        "mktname": mktname,
        "mktshort": mktshort,
        "rtime": "2026-11-11 17:43:28",
        "vcode": vcode,
        "vname": vname,
        "vtype": vtype,
    }
    row.update(changes or {})
    return row


def _contract_row(market_id, mktname, mktshort, vcode, vname, vtype, code, name, changes=None):
    row = {
        "code": code,
        "linetype": 0,
        "mktid": int(market_id),
        "mktname": mktname,
        "mktshort": mktshort,
        "name": name,
        "operno": 213279,
        "rtime": "2026-11-12 08:39:30",
        "scale": 0,
        "typeid": int(vtype),
        "uptime": "2026-11-12 08:39:26.560780000",
        "vcode": vcode,
        "vname": vname,
        "vtype": vtype,
    }
    row.update(changes or {})
    return row


def _quote_payload(**changes):
    data = {"f43": 88880.0, "f57": "lcm", "f58": "碳酸锂主连", "f59": 0, "f86": 1794464936}
    data.update(changes)
    return {"rc": 0, "rt": 4, "svr": 1, "lt": 1, "full": 1, "dlmkts": "", "dsc": "0", "data": data}


_GFX_COMMODITIES = [
    _commodity_row("225", "广期所", "GFEX", "si", "工业硅", "1"),
    _commodity_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2"),
    _commodity_row("225", "广期所", "GFEX", "ps", "多晶硅", "3"),
]
_LC_CONTRACTS = [
    _contract_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lc2705", "碳酸锂2705"),
    _contract_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lcm", "碳酸锂主连"),
    _contract_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lcs", "碳酸锂次主连"),
]
_ROUTES = {
    ("msgid", "225"): _GFX_COMMODITIES,
    ("msgid", "225_2"): _LC_CONTRACTS,
    ("secid", "225.lcm"): _quote_payload(),
    ("secid", "225.lc2705"): _quote_payload(f43=88260.0, f57="lc2705", f58="碳酸锂2705"),
}


@pytest.fixture
def stub(monkeypatch):
    calls: list[tuple[str, dict[str, str]]] = []
    routes = copy.deepcopy(_ROUTES)

    def fake_get(url, *, source, timeout=8.0, headers=None, params=None, **kwargs):
        calls.append((url, dict(params)))
        key = ("msgid", params["msgid"]) if "msgid" in params else ("secid", params["secid"])
        return httpx.Response(
            200,
            request=httpx.Request("GET", url, params=params),
            content=json.dumps(routes[key]).encode(),
        )

    monkeypatch.setattr(futures_sources, "get", fake_get)
    return SimpleNamespace(calls=calls, routes=routes)


@pytest.fixture
def server():
    mcp = StrictToolsMCP("futures-test")
    register_tools(mcp)
    return mcp


def test_adapter_maps_catalog_identity_quote_and_three_requests(stub):
    item = futures_sources.fetch_commodity_price("225.lcm")
    assert item.symbol == "225.lcm" and item.contract_code == "lcm"
    assert item.contract_name == "碳酸锂主连"
    assert item.commodity_code == "lc" and item.commodity_name == "碳酸锂"
    assert item.exchange == "GFEX" and item.contract_type == "main_continuous"
    assert item.price == Decimal("88880.0")
    assert item.currency is None and item.price_unit is None
    assert item.source_time == datetime.fromtimestamp(1794464936, _SHANGHAI)
    assert item.source == "eastmoney"
    assert stub.calls == [
        (_CATALOG_URL, {"msgid": "225"}),
        (_CATALOG_URL, {"msgid": "225_2"}),
        (_QUOTE_URL, _QUOTE_PARAMS),
    ]


def test_month_contract_is_delivery_type(stub):
    item = futures_sources.fetch_commodity_price("225.lc2705")
    assert item.contract_type == "delivery"
    assert item.price == Decimal("88260.0")
    assert stub.calls[2] == (
        _QUOTE_URL,
        {"secid": "225.lc2705", "fltt": "2", "fields": "f43,f57,f58,f59,f86"},
    )


@pytest.mark.parametrize(
    "requested,canonical",
    [
        ("225.lcm", "225.lcm"),
        ("225.LCM", "225.lcm"),
        ("225.LcM", "225.lcm"),
        ("225.LC2705", "225.lc2705"),
    ],
)
def test_case_insensitive_request_restores_catalog_case(stub, requested, canonical):
    item = futures_sources.fetch_commodity_price(requested)
    assert item.symbol == canonical
    assert stub.calls[2] == (
        _QUOTE_URL,
        {"secid": canonical, "fltt": "2", "fields": "f43,f57,f58,f59,f86"},
    )


def test_czce_uppercase_codes_round_trip(stub):
    stub.routes["msgid", "115"] = [_commodity_row("115", "郑州商品交易所", "CZCE", "CF", "棉花", "3")]
    stub.routes["msgid", "115_3"] = [
        # 合约行 mktname 与商品行不同（实测形态），目录身份不比对 mktname。
        _contract_row("115", "郑商所", "CZCE", "CF", "棉花", "3", "CF705", "棉花705"),
        _contract_row("115", "郑商所", "CZCE", "CF", "棉花", "3", "CFM", "棉花主连"),
        _contract_row("115", "郑商所", "CZCE", "CF", "棉花", "3", "CFS", "棉花次主连"),
    ]
    stub.routes["secid", "115.CFM"] = _quote_payload(f43=15525.0, f57="CFM", f58="棉花主连")
    item = futures_sources.fetch_commodity_price("115.cfm")
    assert item.symbol == "115.CFM" and item.contract_code == "CFM"
    assert item.commodity_code == "CF" and item.commodity_name == "棉花"
    assert item.exchange == "CZCE" and item.contract_type == "main_continuous"
    assert stub.calls == [
        (_CATALOG_URL, {"msgid": "115"}),
        (_CATALOG_URL, {"msgid": "115_3"}),
        (_QUOTE_URL, {"secid": "115.CFM", "fltt": "2", "fields": "f43,f57,f58,f59,f86"}),
    ]


@pytest.mark.parametrize(
    "market_id,mktname,exchange,vcode,vname,vtype,code,name,contract_type",
    [
        ("113", "上海期货交易所", "SHFE", "cu", "沪铜", "1", "cum", "沪铜主连", "main_continuous"),
        ("113", "上海期货交易所", "SHFE", "au", "沪金", "5", "au2506", "沪金2506", "delivery"),
        ("114", "大连商品交易所", "DCE", "m", "豆粕", "1", "mm", "豆粕主连", "main_continuous"),
        ("115", "郑州商品交易所", "CZCE", "CF", "棉花", "3", "CF705", "棉花705", "delivery"),
        ("142", "上海国际能源交易中心", "INE", "sc", "原油", "1", "scm", "原油主连", "main_continuous"),
        ("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lc2705", "碳酸锂2705", "delivery"),
    ],
)
def test_all_five_exchanges_resolve(stub, market_id, mktname, exchange, vcode, vname, vtype, code, name, contract_type):
    stub.routes["msgid", market_id] = [_commodity_row(market_id, mktname, exchange, vcode, vname, vtype)]
    stub.routes["msgid", f"{market_id}_{vtype}"] = [
        _contract_row(market_id, mktname, exchange, vcode, vname, vtype, code, name)
    ]
    stub.routes["secid", f"{market_id}.{code}"] = _quote_payload(f43=100.5, f57=code, f58=name)
    item = futures_sources.fetch_commodity_price(f"{market_id}.{code.lower()}")
    assert item.exchange == exchange and item.contract_type == contract_type
    assert item.commodity_code == vcode and item.commodity_name == vname
    assert item.contract_name == name
    assert item.symbol == f"{market_id}.{code}"


def test_sparse_vtype_uses_real_catalog_vtype(stub):
    stub.routes["msgid", "114"] = [
        _commodity_row("114", "大连商品交易所", "DCE", "a", "豆一", "1"),
        _commodity_row("114", "大连商品交易所", "DCE", "c", "玉米", "5"),
        _commodity_row("114", "大连商品交易所", "DCE", "rr", "粳米", "23"),
    ]
    stub.routes["msgid", "114_23"] = [
        _contract_row("114", "大连商品交易所", "DCE", "rr", "粳米", "23", "rrm", "粳米主连")
    ]
    stub.routes["secid", "114.rrm"] = _quote_payload(f43=3500.0, f57="rrm", f58="粳米主连")
    item = futures_sources.fetch_commodity_price("114.rrm")
    assert item.commodity_code == "rr" and item.commodity_name == "粳米"
    assert stub.calls == [
        (_CATALOG_URL, {"msgid": "114"}),
        (_CATALOG_URL, {"msgid": "114_23"}),
        (_QUOTE_URL, {"secid": "114.rrm", "fltt": "2", "fields": "f43,f57,f58,f59,f86"}),
    ]


@pytest.mark.parametrize(
    "symbol",
    [
        "1.600519", "0.000001", "600519.SH", "220.IFM", "225.lcs", "225.LCS", "225.lc",
        "225.lc27055", "225.lc2m70", "225..lcm", "225/cum", "225.lcm ", "225.lc;",
        "225.lcm;DROP", "600519", "", "225.lc2705x", "113.cum-2506",
    ],
)
def test_invalid_or_unsupported_symbols_never_reach_upstream(stub, symbol):
    with pytest.raises(ValueError) as error:
        futures_sources.fetch_commodity_price(symbol)
    assert "东财商品期货标识" in str(error.value)
    assert stub.calls == []


def test_absent_commodity_is_invalid_argument_not_source_error(stub):
    with pytest.raises(ValueError) as error:
        futures_sources.fetch_commodity_price("225.sjm")
    assert "当前目录未提供" in str(error.value)
    assert len(stub.calls) == 1


def test_absent_contract_is_invalid_argument_not_source_error(stub):
    with pytest.raises(ValueError) as error:
        futures_sources.fetch_commodity_price("225.lc9999")
    assert "当前目录未提供" in str(error.value)
    assert len(stub.calls) == 2


def test_ambiguous_suffix_decomposition_is_source_error(stub):
    stub.routes["msgid", "225"] = [
        *_GFX_COMMODITIES,
        _commodity_row("225", "广期所", "GFEX", "lc2", "示例品种", "9"),
    ]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lc2705")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, "", "-", "--"])
async def test_confirmed_contract_without_price_is_available_false(server, stub, empty):
    stub.routes["secid", "225.lcm"] = _quote_payload(f43=empty)
    _, result = await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
    assert result["available"] is False and result["items"] == []
    assert result["reason"] == "commodity_price_unavailable:225.lcm"
    assert result["domain"] == "futures"
    assert len(stub.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [(0, "0"), (-5, "-5"), ("0", "0"), (-0.5, "-0.5")])
async def test_zero_and_negative_prices_are_preserved(server, stub, raw, expected):
    stub.routes["secid", "225.lcm"] = _quote_payload(f43=raw)
    _, result = await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
    assert result["available"] is True
    assert result["items"][0]["price"] == expected


@pytest.mark.asyncio
async def test_price_keeps_source_precision_beyond_28_digits(server, stub):
    raw = "9007199254740993.123456789012345678901234"
    stub.routes["secid", "225.lcm"] = _quote_payload(f43=raw)
    _, result = await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
    assert result["items"][0]["price"] == raw


@pytest.mark.asyncio
async def test_decimal_precision_quote_round_trip(server, stub):
    stub.routes["secid", "225.lcm"] = _quote_payload(f43=854.8, f59=2)
    _, result = await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
    assert result["items"][0]["price"] == "854.8"


@pytest.mark.parametrize(
    "value",
    [True, False, "NaN", "Infinity", "-Infinity", float("inf"), float("nan"), "乱码", [], {}],
)
def test_adapter_rejects_invalid_price_values(stub, value):
    stub.routes["secid", "225.lcm"] = _quote_payload(f43=value)
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"


def test_missing_update_time_stays_null(stub):
    quote = _quote_payload()
    del quote["data"]["f86"]
    stub.routes["secid", "225.lcm"] = quote
    assert futures_sources.fetch_commodity_price("225.lcm").source_time is None
    for empty in (None, "", "-", "--"):
        stub.routes["secid", "225.lcm"] = _quote_payload(f86=empty)
        assert futures_sources.fetch_commodity_price("225.lcm").source_time is None


def test_source_time_is_shanghai_wall_clock(stub):
    stub.routes["secid", "225.lcm"] = _quote_payload(f86="1794464936")
    item = futures_sources.fetch_commodity_price("225.lcm")
    assert item.source_time == datetime.fromtimestamp(1794464936, _SHANGHAI)
    assert item.source_time.isoformat() == "2026-11-12T14:28:56+08:00"


@pytest.mark.parametrize(
    "value",
    [True, "abc", 1.5, -1, "-5", "1794464936.5", 99999999999999, float("nan")],
)
def test_invalid_update_time_is_source_error(stub, value):
    stub.routes["secid", "225.lcm"] = _quote_payload(f86=value)
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False


@pytest.mark.parametrize(
    "changes",
    [
        {"mktid": "114"}, {"mktid": True}, {"mktshort": "DCE"}, {"vcode": ""},
        {"vcode": 123}, {"vname": ""}, {"vtype": "x"}, {"vtype": 2},
    ],
)
def test_commodity_catalog_identity_mismatch_is_source_error(stub, changes):
    stub.routes["msgid", "225"] = [
        _commodity_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", changes)
    ]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"
    assert len(stub.calls) == 1


@pytest.mark.parametrize("rows", [[], "not-a-list", {}, ["bad-row"], [None]])
def test_empty_or_malformed_commodity_catalog_is_source_error(stub, rows):
    stub.routes["msgid", "225"] = rows
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"mktid": 114}, {"mktshort": "SHFE"}, {"vcode": "si"}, {"vname": "工业硅"},
        {"code": ""}, {"code": 123}, {"name": ""}, {"name": None},
    ],
)
def test_contract_catalog_identity_mismatch_is_source_error(stub, changes):
    stub.routes["msgid", "225_2"] = [
        _contract_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lcm", "碳酸锂主连", changes)
    ]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"
    assert len(stub.calls) == 2


@pytest.mark.parametrize("rows", [[], "not-a-list", {}, ["bad-row"], [None]])
def test_empty_or_malformed_contract_catalog_is_source_error(stub, rows):
    stub.routes["msgid", "225_2"] = rows
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"
    assert len(stub.calls) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"rc": 1, "data": {}},
        {},
        {"rc": 0},
        {"rc": 0, "data": None},
        {"rc": 0, "data": "bad-data"},
        {"rc": 0, "data": {"f57": "cu", "f43": 1.0, "f59": 0}},
        {"rc": 0, "data": {"f57": "lcm", "f59": 0, "f86": 1}},
        {"rc": 0, "data": {"f57": 225, "f43": 1.0}},
    ],
)
def test_quote_container_and_identity_errors(stub, payload):
    stub.routes["secid", "225.lcm"] = payload
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False
    assert error.value.code != "invalid_argument"
    assert len(stub.calls) == 3


def test_identical_duplicate_catalog_rows_are_deduplicated(stub):
    stub.routes["msgid", "225"] = [dict(_GFX_COMMODITIES[1]), *_GFX_COMMODITIES]
    stub.routes["msgid", "225_2"] = [dict(_LC_CONTRACTS[1]), _LC_CONTRACTS[1], _LC_CONTRACTS[0]]
    item = futures_sources.fetch_commodity_price("225.lcm")
    assert item.symbol == "225.lcm" and item.contract_name == "碳酸锂主连"


@pytest.mark.parametrize("changes", [{"vname": "碳酸锂新"}, {"vtype": "9"}])
def test_conflicting_duplicate_commodity_is_source_error(stub, changes):
    stub.routes["msgid", "225"] = [
        _GFX_COMMODITIES[1],
        _commodity_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", changes),
    ]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False


@pytest.mark.parametrize("changes", [{"name": "碳酸锂主连2"}, {"code": "LCM"}])
def test_conflicting_duplicate_contract_is_source_error(stub, changes):
    stub.routes["msgid", "225_2"] = [
        _LC_CONTRACTS[1],
        _contract_row("225", "广期所", "GFEX", "lc", "碳酸锂", "2", "lcm", "碳酸锂主连", changes),
    ]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert error.value.source == "eastmoney" and error.value.retryable is False


def test_error_messages_do_not_expose_upstream_bodies(stub):
    stub.routes["msgid", "225"] = [{"secret": "response-secret"}]
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert "response-secret" not in str(error.value)
    stub.routes["msgid", "225"] = copy.deepcopy(_GFX_COMMODITIES)
    stub.routes["secid", "225.lcm"] = {"rc": 0, "data": {"f43": "response-secret"}}
    with pytest.raises(source_requests.SourceError) as error:
        futures_sources.fetch_commodity_price("225.lcm")
    assert "response-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_server_rejects_schema_invalid_arguments_before_upstream(server, stub):
    for bad in ("225.lcs", "1.600519", "220.ifm", "225.lc", "225.lc27055", ""):
        with pytest.raises(ToolError) as error:
            await server.call_tool("get_commodity_price", {"symbol": bad})
        details = json.loads(str(error.value))["error"]
        assert details["code"] == "invalid_argument" and details["retryable"] is False
    with pytest.raises(ToolError):
        await server.call_tool("get_commodity_price", {})
    with pytest.raises(ToolError):
        await server.call_tool("get_commodity_price", {"symbol": "225.lcm", "limit": 1})
    assert stub.calls == []


@pytest.mark.asyncio
async def test_server_reports_missing_commodity_and_contract_as_invalid_argument(server, stub):
    for bad, calls in (("225.sjm", 1), ("225.lc9999", 2)):
        before = len(stub.calls)
        with pytest.raises(ToolError) as error:
            await server.call_tool("get_commodity_price", {"symbol": bad})
        details = json.loads(str(error.value))["error"]
        assert details["code"] == "invalid_argument" and details["retryable"] is False
        assert len(stub.calls) - before == calls


@pytest.mark.asyncio
async def test_server_wraps_adapter_results_into_envelope(server, stub):
    _, result = await server.call_tool("get_commodity_price", {"symbol": "225.LCM"})
    assert result["available"] is True and result["reason"] == ""
    assert result["domain"] == "futures" and result["as_of"]
    (item,) = result["items"]
    assert item["symbol"] == "225.lcm" and item["contract_code"] == "lcm"
    assert item["exchange"] == "GFEX" and item["contract_type"] == "main_continuous"
    assert item["price"] == "88880.0"
    assert item["currency"] is None and item["price_unit"] is None
    assert item["source_time"] == "2026-11-12T14:28:56+08:00"
    assert item["source"] == "eastmoney"
    assert futures.DOMAIN == "futures"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses,body,code,retryable,waits",
    [
        ([503, 200, 200, 200], None, None, None, [1]),
        ([429, 200, 200, 200], None, None, None, [4]),
        ([503, 503, 503], None, "upstream_unavailable", True, [1, 2]),
        ([403], None, "permission_denied", False, []),
        ([200], "response-secret: not JSON", "format", False, []),
    ],
    ids=["503-recovery", "429-retry-after", "503-exhaustion", "403-no-retry", "invalid-json"],
)
async def test_http_uses_shared_retry_and_safe_mcp_errors(
    server, monkeypatch, caplog, statuses, body, code, retryable, waits
):
    attempts, actual_waits = [], []
    monkeypatch.setenv("MCP_SOURCE_MAX_RETRIES", "2")
    monkeypatch.setenv("MCP_CALL_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr(
        source_requests.CallBudget, "wait", lambda self, delay: actual_waits.append(delay)
    )
    monkeypatch.setattr(source_requests.random, "uniform", lambda *_: 0)

    def handle(request):
        index = len(attempts)
        attempts.append(request)
        status = statuses[index - 1] if index <= len(statuses) else 200
        if status != 200:
            return httpx.Response(
                status,
                text="response-secret",
                headers={"Retry-After": "4"} if status == 429 else {},
            )
        if body is not None:
            return httpx.Response(200, text=body)
        params = dict(request.url.params)
        key = ("msgid", params["msgid"]) if "msgid" in params else ("secid", params["secid"])
        return httpx.Response(200, content=json.dumps(_ROUTES[key]).encode())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(httpx, "get", client.get)
        if code is None:
            _, result = await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
            assert result["available"] is True
            assert result["items"][0]["symbol"] == "225.lcm"
            assert result["items"][0]["price"] == "88880.0"
            assert len(attempts) == len(statuses)
        else:
            with pytest.raises(ToolError) as error:
                await server.call_tool("get_commodity_price", {"symbol": "225.lcm"})
            details = json.loads(str(error.value))["error"]
            assert details["source"] == "eastmoney"
            assert details["retryable"] is retryable
            assert details["attempt_count"] == len(statuses)
            if code == "format":
                assert details["code"] != "invalid_argument"
            else:
                # 失败的目录不得伪装成“无此合约”。
                assert details["code"] == code
            assert "response-secret" not in str(error.value)
    assert actual_waits == waits
    assert "response-secret" not in caplog.text
