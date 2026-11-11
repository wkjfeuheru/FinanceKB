from __future__ import annotations

from starlette.testclient import TestClient

from src.mcp.common.auth import BearerAuthMiddleware


async def _echo(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def _client(tokens: tuple[str, ...]) -> TestClient:
    app = BearerAuthMiddleware(
        _echo,
        tokens,
        health_paths=("/healthz", "/mcp/market/healthz"),
    )
    return TestClient(app)


def test_healthz_is_public() -> None:
    client = _client(("secret",))
    assert client.get("/healthz").status_code == 200
    assert client.get("/mcp/market/healthz").status_code == 200


def test_missing_or_invalid_token_is_rejected() -> None:
    client = _client(("secret",))
    response = client.post("/mcp/market")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert client.post("/mcp/market", headers={"Authorization": "secret"}).status_code == 401
    assert client.post("/mcp/market", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/mcp/market", headers={"Authorization": "Bearer "}).status_code == 401


def test_any_configured_token_is_accepted() -> None:
    client = _client(("first", "second"))
    for token in ("first", "second"):
        response = client.post("/mcp/market", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200


def test_non_ascii_bearer_value_is_rejected() -> None:
    client = _client(("secret",))
    response = client.post("/mcp/market", headers=[(b"authorization", b"Bearer caf\xe9")])
    assert response.status_code == 401


def test_empty_configuration_rejects_everything() -> None:
    client = _client(())
    response = client.post("/mcp/market", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 401
