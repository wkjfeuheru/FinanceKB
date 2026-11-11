"""Bounded retries and safe failure metadata for read-only source requests."""

import asyncio
import logging
import math
import os
import random
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Event
from typing import TypeVar

import httpx

T = TypeVar("T")
logger = logging.getLogger("uvicorn.error.source_requests")


class SourceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "upstream_unavailable",
        source: str = "upstream",
        retryable: bool = False,
        attempt_count: int = 1,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.source = source
        self.retryable = retryable
        self.attempt_count = attempt_count
        self.retry_after = retry_after

    def payload(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "source": self.source,
            "retryable": self.retryable,
            "attempt_count": self.attempt_count,
        }


def setting(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 超出配置范围")
    return value


@dataclass
class CallBudget:
    deadline: float
    cancelled: Event = field(default_factory=Event)

    def remaining(self) -> float:
        if self.cancelled.is_set():
            raise CancelledError()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise SourceError("工具调用已达到总时限", code="upstream_timeout", retryable=True)
        return remaining

    def wait(self, delay: float) -> None:
        if delay >= self.remaining():
            raise SourceError("剩余调用时间不足以等待重试", code="upstream_timeout", retryable=True)
        if self.cancelled.wait(delay):
            raise CancelledError()


call_budget: ContextVar[CallBudget | None] = ContextVar("source_call_budget", default=None)


def retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def failure(exc: httpx.HTTPError | SourceError, source: str, attempt: int) -> SourceError:
    if isinstance(exc, SourceError):
        exc.source = source
        exc.attempt_count = attempt
        return exc
    code, retryable, delay = "upstream_unavailable", False, None
    if isinstance(exc, httpx.TimeoutException):
        code, retryable = "upstream_timeout", True
    elif isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError)):
        retryable = True
    elif isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            code = "permission_denied"
        elif status in (400, 422):
            code = "invalid_argument"
        elif status == 429:
            code, retryable = "rate_limited", True
            delay = retry_after(exc.response.headers.get("Retry-After"))
        else:
            retryable = status in (502, 503, 504)
    return SourceError(
        "上游请求失败",
        code=code,
        source=source,
        retryable=retryable,
        attempt_count=attempt,
        retry_after=delay,
    )


class RetryPlan:
    def __init__(self, source: str, timeout: float):
        self.source = source
        self.request_timeout = timeout
        retries = setting("MCP_SOURCE_MAX_RETRIES", 2, 0, 5)
        if not retries.is_integer():
            raise ValueError("MCP_SOURCE_MAX_RETRIES 必须是整数")
        self.attempts = int(retries) + 1
        self.budget = call_budget.get() or CallBudget(
            time.monotonic() + setting("MCP_CALL_TIMEOUT_SECONDS", 120, 1, 600)
        )

    def timeout(self) -> float:
        return min(self.request_timeout, self.budget.remaining())

    def delay(self, exc: httpx.HTTPError | SourceError, attempt: int) -> float:
        error = failure(exc, self.source, attempt)
        logger.info(
            "source_request_failed source=%s attempt_count=%s code=%s retryable=%s",
            self.source,
            attempt,
            error.code,
            error.retryable,
        )
        if not error.retryable or attempt >= self.attempts:
            raise error from None
        delay = max(2 ** (attempt - 1) + random.uniform(0, 0.25), error.retry_after or 0)
        if delay >= self.budget.remaining():
            raise error from None
        return delay


def execute(operation: Callable[[float], T], *, source: str, timeout: float = 8) -> T:
    plan = RetryPlan(source, timeout)
    for attempt in range(1, plan.attempts + 1):
        try:
            result = operation(plan.timeout())
            plan.budget.remaining()
            logger.info("source_request_succeeded source=%s attempt_count=%s", source, attempt)
            return result
        except (httpx.HTTPError, SourceError) as exc:
            delay = plan.delay(exc, attempt)
        plan.budget.wait(delay)
    raise AssertionError("unreachable")


async def execute_async(
    operation: Callable[[float], Awaitable[T]],
    *,
    source: str,
    timeout: float,
) -> T:
    plan = RetryPlan(source, timeout)
    for attempt in range(1, plan.attempts + 1):
        try:
            async with asyncio.timeout(plan.budget.remaining()):
                result = await operation(plan.timeout())
                logger.info("source_request_succeeded source=%s attempt_count=%s", source, attempt)
                return result
        except TimeoutError:
            raise SourceError(
                "工具调用已达到总时限",
                code="upstream_timeout",
                source=source,
                retryable=True,
                attempt_count=attempt,
            ) from None
        except (httpx.HTTPError, SourceError) as exc:
            delay = plan.delay(exc, attempt)
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def get(
    url: str, *, source: str, timeout: float = 8, client: httpx.Client | None = None, **kwargs
) -> httpx.Response:
    def request(remaining: float) -> httpx.Response:
        response = (client.get if client else httpx.get)(url, timeout=remaining, **kwargs)
        response.raise_for_status()
        return response

    return execute(request, source=source, timeout=timeout)


def log_fallback(source: str, target: str, error: Exception) -> None:
    logger.info(
        "source_fallback source=%s target=%s code=%s attempt_count=%s fallback=true",
        source,
        target,
        getattr(error, "code", "invalid_response"),
        getattr(error, "attempt_count", 1),
    )
