"""下游数据接口的静态 Bearer Token 鉴权。

重点：未配置令牌时必须拒绝（不能因为「没配」而放行），错误统一 401，
且逐个 token 比较不短路。
"""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api import dependencies


def _credentials(token: str, scheme: str = "Bearer") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def _call(credentials, tokens):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(dependencies, "DATA_API_TOKENS", tokens)
        return asyncio.run(dependencies.require_data_api_token(credentials))


def test_missing_configuration_rejects_every_request():
    with pytest.raises(HTTPException) as excinfo:
        _call(_credentials("anything"), ())
    assert excinfo.value.status_code == 401
    assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"


def test_missing_or_non_bearer_header_is_401():
    with pytest.raises(HTTPException) as excinfo:
        _call(None, ("token-a",))
    assert excinfo.value.status_code == 401
    with pytest.raises(HTTPException):
        _call(_credentials("token-a", scheme="Basic"), ("token-a",))


def test_wrong_token_is_401():
    with pytest.raises(HTTPException) as excinfo:
        _call(_credentials("token-b"), ("token-a",))
    assert excinfo.value.status_code == 401


def test_any_configured_token_passes():
    assert _call(_credentials("token-b"), ("token-a", "token-b")) == "data-api"


def test_empty_header_value_is_401():
    with pytest.raises(HTTPException):
        _call(_credentials(""), ("token-a",))
