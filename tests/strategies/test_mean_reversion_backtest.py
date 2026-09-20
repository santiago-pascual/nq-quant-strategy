from __future__ import annotations

import pytest

from src.strategies.mean_reversion import (
    MRL1_CONFIG,
    MRS2_CONFIG,
)

from src.strategies.mean_reversion.backtest import (
    MeanReversionBacktestAdapter,
)


# =============================================================================
# HELPERS
# =============================================================================


def make_bar(
    *,
    close: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    hmm_state: int = 2,
    vol_percentile: float = 90.0,
    zscore: float = 2.5,
) -> dict:
    return {
        "timestamp": "2026-01-01T14:30:00+00:00",
        "symbol": "MNQ",
        "open": 100.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
        "hmm_state": hmm_state,
        "vol_percentile": vol_percentile,
        "zscore": zscore,
    }


# =============================================================================
# CONSTRUCTION
# =============================================================================


def test_mrs2_adapter_constructs():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    assert adapter.config == MRS2_CONFIG
    assert adapter.active_trade is None


def test_mrl1_adapter_constructs():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    assert adapter.config == MRL1_CONFIG
    assert adapter.active_trade is None


# =============================================================================
# ENTRY
# =============================================================================


def test_mrs2_valid_signal_opens_short_trade():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    assert adapter.active_trade is not None
    assert adapter.active_trade.side == "SHORT"
    assert adapter.active_trade.entry_price == pytest.approx(100.0)
    assert adapter.active_trade.bars_elapsed == 0


def test_mrl1_valid_signal_opens_long_trade():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    assert adapter.active_trade is not None
    assert adapter.active_trade.side == "LONG"
    assert adapter.active_trade.entry_price == pytest.approx(100.0)


def test_invalid_context_does_not_open_trade():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            hmm_state=1,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    assert adapter.active_trade is None


# =============================================================================
# MRS2 EXITS
# =============================================================================


def test_mrs2_short_target_exit():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    # Entry.
    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    # Target = 72.5.
    trade = adapter.process_bar(
        make_bar(
            close=80.0,
            high=101.0,
            low=72.5,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.side == "SHORT"
    assert trade.exit_reason == "target"
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(72.5)
    assert trade.r_multiple == pytest.approx(1.10)

    assert adapter.active_trade is None


def test_mrs2_short_stop_exit():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    trade = adapter.process_bar(
        make_bar(
            close=120.0,
            high=125.0,
            low=99.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(125.0)
    assert trade.r_multiple == pytest.approx(-1.0)


def test_mrs2_same_bar_conflict_uses_stop():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    trade = adapter.process_bar(
        make_bar(
            close=100.0,
            high=130.0,
            low=70.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "stop"
    assert trade.r_multiple == pytest.approx(-1.0)


# =============================================================================
# MRL1 EXITS
# =============================================================================


def test_mrl1_long_target_exit():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    # Target = 125.
    trade = adapter.process_bar(
        make_bar(
            close=120.0,
            high=125.0,
            low=99.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.side == "LONG"
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(125.0)
    assert trade.r_multiple == pytest.approx(25.0 / 37.5)


def test_mrl1_long_stop_exit():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    trade = adapter.process_bar(
        make_bar(
            close=80.0,
            high=105.0,
            low=62.5,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(62.5)
    assert trade.r_multiple == pytest.approx(-1.0)


def test_mrl1_same_bar_conflict_uses_stop():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    trade = adapter.process_bar(
        make_bar(
            close=100.0,
            high=130.0,
            low=50.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "stop"
    assert trade.r_multiple == pytest.approx(-1.0)


# =============================================================================
# HORIZON
# =============================================================================


def test_mrs2_horizon_exit():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    # Bars 1-29: no exit.
    for _ in range(29):
        trade = adapter.process_bar(
            make_bar(
                close=99.0,
                high=101.0,
                low=99.0,
                hmm_state=0,
                vol_percentile=0.0,
                zscore=0.0,
            )
        )

        assert trade is None

    # Bar 30: timeout.
    trade = adapter.process_bar(
        make_bar(
            close=98.0,
            high=101.0,
            low=97.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "timeout"
    assert trade.bars_elapsed == 30
    assert trade.r_multiple == pytest.approx(0.08)


def test_mrl1_horizon_exit():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    adapter.process_bar(
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-3.0,
        )
    )

    for _ in range(7):
        trade = adapter.process_bar(
            make_bar(
                close=101.0,
                high=102.0,
                low=99.0,
                hmm_state=0,
                vol_percentile=0.0,
                zscore=0.0,
            )
        )

        assert trade is None

    trade = adapter.process_bar(
        make_bar(
            close=103.75,
            high=104.0,
            low=102.0,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        )
    )

    assert trade is not None
    assert trade.exit_reason == "timeout"
    assert trade.bars_elapsed == 8
    assert trade.r_multiple == pytest.approx(0.10)


# =============================================================================
# RESET / RUN
# =============================================================================


def test_reset_closes_active_tracking_without_creating_trade():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    adapter.process_bar(
        make_bar(
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        )
    )

    assert adapter.active_trade is not None

    adapter.reset()

    assert adapter.active_trade is None


def test_run_returns_completed_trades():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    bars = [
        make_bar(
            close=100.0,
            high=101.0,
            low=99.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        ),
        make_bar(
            close=80.0,
            high=101.0,
            low=72.5,
            hmm_state=0,
            vol_percentile=0.0,
            zscore=0.0,
        ),
    ]

    trades = adapter.run(bars)

    assert len(trades) == 1
    assert trades[0].candidate_id == "MRS2_NEW"
    assert trades[0].strategy_name == "MRS2"
    assert trades[0].side == "SHORT"
    assert trades[0].exit_reason == "target"
