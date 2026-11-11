"""Infoway Python SDK 的请求配置、进程内限频与脱敏错误。"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from threading import Lock
from typing import TypeVar

from infoway import InfowayClient
from infoway.exceptions import (
    InfowayAPIError,
    InfowayAuthError,
    InfowayIoError,
    InfowayRateLimitError,
    InfowayTimeoutError,
)

from .source_requests import SourceError, call_budget, execute

T = TypeVar("T")
_lock = Lock()
_next_request = 0.0


DataSourceError = SourceError


def call_infoway(operation: Callable[[InfowayClient], T]) -> T:
    api_key = os.getenv("INFOWAY_API_KEY", "").strip()
    if not api_key:
        raise DataSourceError(
            "Infoway 未配置 INFOWAY_API_KEY", code="permission_denied", source="infoway"
        )

    def request(timeout: float) -> T:
        global _next_request
        try:
            with _lock:
                delay = max(0.0, _next_request - time.monotonic())
                budget = call_budget.get()
                if budget:
                    budget.wait(delay)
                    timeout = min(timeout, budget.remaining())
                else:
                    time.sleep(delay)
                _next_request = time.monotonic() + 1.05
                # SDK counts initial request; retries are owned by execute().
                with InfowayClient(api_key=api_key, timeout=timeout, max_retries=1) as client:
                    return operation(client)
        except (InfowayAPIError, InfowayIoError, InfowayTimeoutError) as exc:
            ret = getattr(exc, "ret", None)
            code = "upstream_unavailable"
            retryable = ret in (502, 503, 504) or isinstance(exc, InfowayIoError)
            if isinstance(exc, InfowayTimeoutError):
                code, retryable = "upstream_timeout", True
            elif isinstance(exc, InfowayRateLimitError):
                # SDK business code 502 denotes a daily quota, unlike HTTP 502.
                code, retryable = "rate_limited", ret != 502
            elif isinstance(exc, InfowayAuthError) or ret in (401, 403, 514):
                code, retryable = "permission_denied", False
            suffix = f"（错误码 {ret}）" if isinstance(ret, int) else ""
            raise DataSourceError(
                f"Infoway 请求失败{suffix}",
                code=code,
                retryable=retryable,
                source="infoway",
            ) from None

    return execute(request, source="infoway")
