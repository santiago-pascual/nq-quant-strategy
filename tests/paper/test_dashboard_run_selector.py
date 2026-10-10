import json
from pathlib import Path
import secrets
import shutil

from paper_dashboard.run_selector import classify_run, default_run, discover_runs


def _write(path: Path, name: str, value: dict) -> None:
    (path / name).write_text(json.dumps(value), encoding="utf-8")


def test_default_prefers_active_delayed_paper_to_historical_replay():
    root = Path.cwd() / f".dashboard_run_selector_{secrets.token_hex(5)}"
    historical = root / "oos_mnqv0_20260827_20261008_1302"
    delayed = root / "delayed_mnqz6_paper_accepted_20261008_1303_r3"
    try:
        historical.mkdir(parents=True)
        delayed.mkdir()
        _write(historical, "oos_run_scope.json", {"start_utc_inclusive": "2026-08-27T00:00:00Z"})
        _write(historical, "status.json", {"system": {"state": "ERROR"}})
        _write(delayed, "delayed_paper_run.json", {"mode": "DELAYED_IBKR_PAPER"})
        _write(delayed, "status.json", {"mode": "PAPER", "system": {
            "state": "RUNNING", "feed_health": {"source": "ibkr_delayed_recoverable_paper"}
        }})

        options = discover_runs(root, preferred=historical)
        assert {option.kind for option in options} == {"HISTORICAL_REPLAY", "DELAYED_PAPER"}
        assert default_run(options, historical).path == delayed.resolve()
        assert "DELAYED PAPER" in default_run(options, historical).label
        assert classify_run(historical).kind == "HISTORICAL_REPLAY"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_selector_handles_incomplete_status_without_inventing_run():
    root = Path.cwd() / f".dashboard_run_incomplete_{secrets.token_hex(5)}"
    try:
        incomplete = root / "initializing"
        incomplete.mkdir(parents=True)
        _write(incomplete, "delayed_paper_run.json", {"mode": "DELAYED_IBKR_PAPER"})
        selected = default_run(discover_runs(root))
        assert selected is not None
        assert selected.kind == "DELAYED_PAPER"
        assert selected.status == "UNAVAILABLE"
    finally:
        shutil.rmtree(root, ignore_errors=True)
