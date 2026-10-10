from pathlib import Path

import pytest

from paper_dashboard.research_analytics import (
    ResearchIntegrityError,
    cumulative_r_series,
    equity_drawdown_r_series,
    load_validated_research,
    moving_block_expectancy_band,
    reconstruct_paper_outcomes,
    rolling_metrics,
    summarize_r,
)


RESEARCH_DIR = Path("src/research/results/portfolio/independent_reproduction")


def test_frozen_research_provenance_and_oos_metrics_are_verified():
    result = load_validated_research(Path.cwd())
    report = result["report"]["portfolio_metrics"]
    assert result["available"] is True
    assert result["provenance"]["trade_rows"] == 3255
    assert result["provenance"]["daily_rows"] == 1486
    assert report["total_R"] == pytest.approx(289.6619012127312)
    assert report["expectancy_R"] == pytest.approx(0.0889898314)
    assert result["provenance"]["trades_sha256"] == "950f95bba2ed74bb11fc5840a85d720b4f197aac5c72debcd0f6624f401c81ea"
    assert result["trades"] == sorted(result["trades"], key=lambda row: (row["entry_timestamp_utc"], row["strategy"], row["source_row"]))


def test_partial_research_csv_is_rejected_without_mutating_source(tmp_path):
    source = Path.cwd() / RESEARCH_DIR
    target = tmp_path / "independent_reproduction"
    target.mkdir()
    for name in ("independent_reproduction_report.json", "independent_reproduction_oos_trades.csv",
                 "independent_reproduction_daily.csv"):
        (target / name).write_bytes((source / name).read_bytes())
    with (target / "independent_reproduction_oos_trades.csv").open("a", encoding="utf-8") as stream:
        stream.write('"partial,MRL1,')
    with pytest.raises(ResearchIntegrityError, match="hash mismatch"):
        load_validated_research(Path.cwd(), target.relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else target)


def test_paper_outcomes_are_normalized_and_duplicate_ids_fail_closed():
    rows = [
        {"trade_id": "later", "strategy": "ORB", "exit_timestamp_utc": "2026-10-08T16:47:00Z", "net_pnl": -239.22, "realized_r": -1.005126},
        {"trade_id": "earlier", "strategy": "MRL1", "exit_timestamp_utc": "2026-10-08T15:47:00Z", "net_pnl": 100, "realized_r": 0.5},
        {"trade_id": "partial", "exit_timestamp_utc": None, "net_pnl": 1, "realized_r": 1},
    ]
    normalized = reconstruct_paper_outcomes(rows)
    assert normalized["available"] is True
    assert normalized["trade_count"] == 2
    assert normalized["invalid_rows"] == 1
    assert [row["trade_id"] for row in normalized["trades"]] == ["earlier", "later"]
    duplicate = reconstruct_paper_outcomes([rows[0], rows[0]])
    assert duplicate["available"] is False
    assert "Duplicate" in duplicate["reason"]


def test_rolling_metrics_are_chronological_and_use_complete_windows():
    rows = [
        {"timestamp": "2026-01-03T00:00:00Z", "r": -1},
        {"timestamp": "2026-01-01T00:00:00Z", "r": 1},
        {"timestamp": "2026-01-02T00:00:00Z", "r": 0.5},
    ]
    output = rolling_metrics(rows, value_key="r", time_key="timestamp", window=2)
    assert len(output) == 2
    assert output[0]["timestamp_utc"].startswith("2026-01-02")
    assert output[0]["expectancy_r"] == pytest.approx(0.75)
    assert output[1]["timestamp_utc"].startswith("2026-01-03")
    assert output[1]["profit_factor"] == pytest.approx(0.5)


def test_moving_block_band_is_deterministic_and_marks_insufficient_sample():
    unavailable = moving_block_expectancy_band([1, -1], window=3)
    assert unavailable["available"] is False
    values = [(-1.0 if index % 3 == 0 else .25) for index in range(40)]
    first = moving_block_expectancy_band(values, window=10, block_length=3, samples=200)
    second = moving_block_expectancy_band(values, window=10, block_length=3, samples=200)
    assert first == second
    assert first["available"] is True
    assert first["lower"] <= first["upper"]


def test_native_r_normalization_and_transition_gap_have_no_synthetic_points():
    assert summarize_r([.5, -.25])["total_r"] == pytest.approx(.25)
    curve = cumulative_r_series([
        {"exit": "2020-06-19T20:00:00Z", "realized_r": 1},
        {"exit": "2026-10-08T16:47:00Z", "realized_r": -.5},
    ], value_key="realized_r", time_key="exit")
    assert len(curve) == 2
    assert curve[-1]["cumulative_r"] == pytest.approx(.5)
    # Daily drawdown is reconstructed only from recorded daily observations.
    dd = equity_drawdown_r_series([{"timestamp_utc": "2026-01-01T00:00:00Z", "daily_r": 1},
                                   {"timestamp_utc": "2026-01-03T00:00:00Z", "daily_r": -2}])
    assert len(dd) == 2
    assert dd[-1]["drawdown_r"] == pytest.approx(-2)


def test_empty_and_missing_paper_records_are_not_reported_as_zero_outcomes():
    assert reconstruct_paper_outcomes(None)["available"] is False
    empty = reconstruct_paper_outcomes([])
    assert empty["available"] is True and empty["trade_count"] == 0
    assert rolling_metrics([], value_key="r", time_key="timestamp", window=25) == []
