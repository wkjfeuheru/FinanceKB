from fastapi.testclient import TestClient

import src.main as main


def test_spa_fallback_serves_index_for_history_routes(tmp_path, monkeypatch):
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    # TrustedHostMiddleware 只放行配置的主机名，默认的 testserver 会被拒绝
    client = TestClient(main.app, base_url="http://localhost")

    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "spa" in resp.text
    assert client.get("/documents").status_code == 200


def test_spa_fallback_keeps_api_and_missing_files_404(tmp_path, monkeypatch):
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    client = TestClient(main.app, base_url="http://localhost")

    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/missing.js").status_code == 404
