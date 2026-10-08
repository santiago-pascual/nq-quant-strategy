import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.paper.analytics_db import PaperAnalyticsStore
from src.paper.monitoring_api import create_monitoring_server


def _request(url, *, token=None, method="GET"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(url, headers=headers, method=method)
    try:
        response = urlopen(request, timeout=3)
    except HTTPError as exc:
        response = exc
    return response.status, json.loads(response.read())


def test_monitor_api_is_authenticated_get_only_and_read_only(tmp_path):
    PaperAnalyticsStore(tmp_path / "paper_analytics.sqlite3").close()
    server = create_monitoring_server(tmp_path, token="test-token-that-is-at-least-24", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = _request(base + "/v1/positions")
        assert status == 401
        assert body["data"]["error"] == "unauthorized"

        status, body = _request(base + "/v1/positions", token="test-token-that-is-at-least-24")
        assert status == 200
        assert body["schema_version"] == "1.0"
        assert body["data"] == []

        status, body = _request(base + "/v1/orders", token="test-token-that-is-at-least-24")
        assert status == 404
        status, body = _request(base + "/v1/positions", token="test-token-that-is-at-least-24", method="POST")
        assert status == 405

        reader = PaperAnalyticsStore(tmp_path / "paper_analytics.sqlite3")
        assert reader._db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        reader.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_monitor_api_requires_strong_token(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="24 characters"):
        create_monitoring_server(tmp_path, token="short", port=0)
    with pytest.raises(ValueError, match="loopback only"):
        create_monitoring_server(tmp_path, token="test-token-that-is-at-least-24", host="0.0.0.0", port=0)
