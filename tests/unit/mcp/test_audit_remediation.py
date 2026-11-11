import asyncio
import json
import time
from concurrent.futures import CancelledError
from datetime import UTC
from threading import Event

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from src.mcp.common.server import StrictToolsMCP
from src.mcp.knowledge import client as knowledge_client
from src.mcp.knowledge import server as knowledge
from src.queries import content_http, infoway_source
from src.queries import source_requests as requests


@pytest.fixture
def no_retry_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(requests.CallBudget, "wait", lambda self, delay: waits.append(delay))
    monkeypatch.setattr(requests.random, "uniform", lambda *_: 0)
    return waits


@pytest.mark.asyncio
async def test_knowledge_publishes_result_schema_and_read_only_hint():
    server = FastMCP("audit")
    knowledge.register_tools(server)
    for tool in await server.list_tools():
        assert tool.outputSchema
        assert tool.annotations.readOnlyHint is True
        assert all(p.get("description") for p in tool.inputSchema["properties"].values())


def test_content_request_recovers_from_transient_failure(no_retry_wait):
    attempts = []

    def handle(request):
        attempts.append(request)
        return httpx.Response(503 if len(attempts) == 1 else 200, text="ready")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert content_http.fetch_text(client, "https://example.test", "eastmoney") == "ready"
    assert len(attempts) == 2


def test_content_request_exhaustion_is_machine_readable(no_retry_wait):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        with pytest.raises(RuntimeError) as error:
            content_http.fetch_text(client, "https://example.test", "eastmoney")
    assert error.value.code == "upstream_unavailable"
    assert error.value.attempt_count == 3
    assert error.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["search_research", "search_policy"])
@pytest.mark.parametrize("patch", [{"page": "1"}, {"page": True}, {"unexpected": 1}])
async def test_knowledge_rejects_invalid_types_before_upstream(name, patch, monkeypatch):
    monkeypatch.setattr(
        knowledge, "get_client", lambda: pytest.fail("invalid request reached upstream")
    )
    server = StrictToolsMCP("audit")
    knowledge.register_tools(server)
    with pytest.raises(ToolError) as error:
        await server.call_tool(name, {"query": "x", **patch})
    assert json.loads(str(error.value))["error"]["code"] == "invalid_argument"


@pytest.mark.parametrize(
    "status,code",
    [(401, "permission_denied"), (403, "permission_denied"), (422, "invalid_argument")],
)
def test_permanent_failures_do_not_retry(status, code, no_retry_wait):
    attempts = []

    def handle(request):
        attempts.append(1)
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(requests.SourceError) as error:
            requests.get("https://example.test", client=client, source="eastmoney")
    assert len(attempts) == 1
    assert error.value.code == code and not error.value.retryable
    assert no_retry_wait == []


def test_rate_limit_honors_retry_after(no_retry_wait):
    attempts = []

    def handle(request):
        attempts.append(1)
        return (
            httpx.Response(429, headers={"Retry-After": "4"})
            if len(attempts) == 1
            else httpx.Response(200)
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        requests.get("https://example.test", client=client, source="sina")
    assert no_retry_wait == [4] and len(attempts) == 2


def test_retry_cannot_exceed_call_deadline(no_retry_wait):
    token = requests.call_budget.set(requests.CallBudget(time.monotonic() + 0.1))
    attempts = []
    try:

        def fail(timeout):
            attempts.append(1)
            raise httpx.ReadTimeout("sensitive original exception")

        with pytest.raises(requests.SourceError) as error:
            requests.execute(fail, source="eastmoney")
    finally:
        requests.call_budget.reset(token)
    assert len(attempts) == 1 and no_retry_wait == []
    assert "sensitive" not in str(error.value)


@pytest.mark.asyncio
async def test_async_cancellation_stops_retries():
    started = asyncio.Event()
    attempts = []

    async def fail(timeout):
        attempts.append(1)
        started.set()
        raise httpx.ReadTimeout("offline")

    task = asyncio.create_task(requests.execute_async(fail, source="knowledge-api", timeout=1))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_mcp_cancellation_wakes_sync_retry_worker():
    started, stopped = Event(), Event()
    attempts = []

    def fail(timeout):
        attempts.append(1)
        started.set()
        raise httpx.ReadTimeout("offline")

    def worker():
        try:
            requests.execute(fail, source="eastmoney")
        except CancelledError:
            stopped.set()
            raise

    async def tool() -> str:
        return await asyncio.to_thread(worker)

    server = StrictToolsMCP("audit")
    server.add_tool(tool)
    task = asyncio.create_task(server.call_tool("tool", {}))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(stopped.wait, 2)
    assert len(attempts) == 1


@pytest.mark.parametrize("ret,expected", [(514, 1), (429, 3), (502, 1), (503, 3)])
def test_infoway_sdk_attempts_are_bounded(ret, expected, no_retry_wait, monkeypatch):
    attempts = []

    def handle(request):
        attempts.append(1)
        return httpx.Response(200, json={"ret": ret, "msg": "sensitive", "data": None})

    original = httpx.Client
    monkeypatch.setenv("INFOWAY_API_KEY", "offline-key")
    monkeypatch.setattr(infoway_source, "_next_request", 0)
    monkeypatch.setattr(infoway_source.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    from infoway import SymbolType

    with pytest.raises(requests.SourceError) as error:
        infoway_source.call_infoway(lambda c: c.basic.get_symbols(SymbolType.STOCK_CN))
    assert len(attempts) == expected
    assert error.value.attempt_count == expected
    assert "sensitive" not in str(error.value)


def test_retry_after_http_date_and_invalid_values():
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    future = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
    assert 18 <= requests.retry_after(future) <= 20
    assert requests.retry_after("not-a-date") is None
    assert requests.retry_after("NaN") is None


@pytest.mark.asyncio
async def test_total_deadline_returns_machine_readable_error(monkeypatch):
    monkeypatch.setenv("MCP_CALL_TIMEOUT_SECONDS", "1")

    async def slow() -> str:
        await asyncio.sleep(10)
        return "late"

    server = StrictToolsMCP("audit")
    server.add_tool(slow)
    start = time.monotonic()
    with pytest.raises(ToolError) as error:
        await server.call_tool("slow", {})
    assert time.monotonic() - start < 3
    assert json.loads(str(error.value))["error"]["code"] == "upstream_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_knowledge_http_retry_recovery_and_exhaustion(monkeypatch, recover):
    attempts, waits = [], []

    def handle(request):
        attempts.append(1)
        if recover and len(attempts) == 2:
            return httpx.Response(200, json={"items": []})
        return httpx.Response(429, headers={"Retry-After": "4"}, text="sensitive")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handle)),
    )

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(requests.asyncio, "sleep", sleep)
    client = knowledge_client.KnowledgeAPIClient(base_url="https://example.test", token="test")
    if recover:
        assert await client.search("policy", {"query": "x"}) == {"items": []}
        assert len(attempts) == 2 and waits == [4]
    else:
        with pytest.raises(requests.SourceError) as error:
            await client.search("policy", {"query": "x"})
        assert len(attempts) == 3 and waits == [4, 4]
        assert error.value.code == "rate_limited" and error.value.attempt_count == 3
        assert "sensitive" not in str(error.value)
