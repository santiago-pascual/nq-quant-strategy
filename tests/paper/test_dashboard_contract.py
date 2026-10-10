import json
from pathlib import Path
import secrets
import shutil
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.paper.monitoring_api import create_monitoring_server


TOKEN = "dashboard-test-token-at-least-24-chars"


def _get(url, *, token=TOKEN, method="GET"):
    request = Request(url, method=method, headers={"Authorization": f"Bearer {token}"})
    try:
        response = urlopen(request, timeout=3)
    except HTTPError as exc:
        response = exc
    return response.status, json.loads(response.read())


def test_dashboard_empty_replay_scope_is_unavailable_not_fabricated():
    output = Path.cwd() / f".dashboard_scope_{secrets.token_hex(5)}"
    output.mkdir()
    scope = {
        "start_utc_inclusive": "2026-08-27T00:00:00Z",
        "replay_end_utc_exclusive": "2026-10-08T13:03:00Z",
    }
    (output / "oos_run_scope.json").write_text(json.dumps(scope), encoding="utf-8")
    server = create_monitoring_server(output, token=TOKEN, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(f"http://127.0.0.1:{server.server_port}/v1/dashboard")
        data = body["data"]
        assert status == 200
        assert data["run"]["kind"] == "HISTORICAL_REPLAY"
        assert data["run"]["status"] == "UNAVAILABLE"
        assert data["run"]["scope"] == scope
        assert data["run"]["processed_bars"] is None
        assert data["run"]["last_processed_bar"] is None
        assert data["run"]["progress_pct"] is None
        assert data["available"] is False
        assert data["account"] is None
        assert data["equity_curve"] == []
        assert data["system"]["latest_event"] is None

        status, _ = _get(f"http://127.0.0.1:{server.server_port}/v1/dashboard", method="POST")
        assert status == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(output, ignore_errors=True)


def test_dashboard_requires_authentication():
    output = Path.cwd() / f".dashboard_auth_{secrets.token_hex(5)}"
    output.mkdir()
    server = create_monitoring_server(output, token=TOKEN, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _get(f"http://127.0.0.1:{server.server_port}/v1/dashboard", token="")
        assert status == 401
        assert body["data"]["error"] == "unauthorized"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(output, ignore_errors=True)


def test_streamlit_dashboard_has_pages_refresh_and_no_order_controls():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "paper_dashboard" / "app.py").read_text(encoding="utf-8")
    for page in ("Command Center", "Performance", "Strategies", "Trade Explorer",
                 "Positions & Orders", "Risk Monitor", "System Observatory"):
        assert page in source
    assert "@st.fragment(run_every=\"30s\")" in source
    assert "max-width:900px" in source
    assert '[data-testid="stColumn"]' in source
    assert "/v1/dashboard" in source
    assert "place_order" not in source
    assert "submit_order" not in source


def test_streamlit_dashboard_has_empty_state_and_unsupported_metrics_copy():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "paper_dashboard" / "app.py").read_text(encoding="utf-8")
    assert "Unavailable" in source
    assert "no activity or P&L is estimated" in source
    assert "No state semantics are inferred or aligned" in source


def test_dashboard_mobile_listener_is_tailscale_only_and_api_stays_loopback():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "start_paper_dashboard.ps1").read_text(encoding="utf-8")
    app = (root / "paper_dashboard" / "app.py").read_text(encoding="utf-8")
    assert '[string]$DashboardAddress = "100.114.250.67"' in launcher
    assert "--server.address $DashboardAddress" in launcher
    assert "--server.address 0.0.0.0" not in launcher
    assert 'host="127.0.0.1"' in app
    assert "submit_order" not in app
    assert "place_order" not in app
