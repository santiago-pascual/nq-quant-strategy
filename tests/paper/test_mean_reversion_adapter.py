from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.base import StrategySignal
from src.strategies.mean_reversion.backtest import MeanReversionBacktestAdapter
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG


def _bar(
    timestamp: str,
    *,
    close: float,
    high: float,
    low: float,
    hmm_state: int,
    vol_percentile: float,
    zscore: float,
) -> dict:
    return {
        "timestamp": pd.Timestamp(timestamp),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "hmm_state": hmm_state,
        "vol_percentile": vol_percentile,
        "zscore": zscore,
    }


def test_mrl1_adapter_enters_and_exits_at_target():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0
    target = entry_price + MRL1_CONFIG.target_points

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=entry_price,
        high=entry_price,
        low=entry_price,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    exit_bar = _bar(
        "2026-01-01 10:01:00",
        close=target,
        high=target + 1.0,
        low=entry_price,
        hmm_state=0,
        vol_percentile=50.0,
        zscore=0.0,
    )

    first_result = adapter.process_bar(entry_bar)

    assert first_result is None
    assert adapter.active_trade is not None
    assert adapter.active_trade.side == "LONG"
    assert adapter.active_trade.entry_price == pytest.approx(entry_price)
    assert adapter.active_trade.bars_elapsed == 0

    completed = adapter.process_bar(exit_bar)

    assert completed is not None
    assert completed.strategy_name == MRL1_CONFIG.name
    assert completed.candidate_id == MRL1_CONFIG.candidate_id
    assert completed.side == "LONG"
    assert completed.entry_timestamp == entry_bar["timestamp"]
    assert completed.entry_price == pytest.approx(entry_price)
    assert completed.exit_price == pytest.approx(target)
    assert completed.exit_reason == "target"
    assert completed.bars_elapsed == 1
    assert completed.r_multiple == pytest.approx(MRL1_CONFIG.rr)

    assert adapter.active_trade is None


def test_mrs2_adapter_enters_and_exits_at_stop():
    adapter = MeanReversionBacktestAdapter(MRS2_CONFIG)

    entry_price = 100.0
    stop = entry_price + MRS2_CONFIG.stop_points

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=entry_price,
        high=entry_price,
        low=entry_price,
        hmm_state=MRS2_CONFIG.hmm_state,
        vol_percentile=90.0,
        zscore=MRS2_CONFIG.zscore_threshold,
    )

    exit_bar = _bar(
        "2026-01-01 10:01:00",
        close=stop,
        high=stop + 1.0,
        low=entry_price,
        hmm_state=0,
        vol_percentile=50.0,
        zscore=0.0,
    )

    first_result = adapter.process_bar(entry_bar)

    assert first_result is None
    assert adapter.active_trade is not None
    assert adapter.active_trade.side == "SHORT"
    assert adapter.active_trade.entry_price == pytest.approx(entry_price)
    assert adapter.active_trade.bars_elapsed == 0

    completed = adapter.process_bar(exit_bar)

    assert completed is not None
    assert completed.strategy_name == MRS2_CONFIG.name
    assert completed.candidate_id == MRS2_CONFIG.candidate_id
    assert completed.side == "SHORT"
    assert completed.entry_timestamp == entry_bar["timestamp"]
    assert completed.entry_price == pytest.approx(entry_price)
    assert completed.exit_price == pytest.approx(stop)
    assert completed.exit_reason == "stop"
    assert completed.bars_elapsed == 1
    assert completed.r_multiple == pytest.approx(-1.0)

    assert adapter.active_trade is None


def test_entry_bar_is_not_evaluated_for_exit():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0
    target = entry_price + MRL1_CONFIG.target_points
    stop = entry_price - MRL1_CONFIG.stop_points

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=entry_price,
        high=target + 10.0,
        low=stop - 10.0,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    result = adapter.process_bar(entry_bar)

    assert result is None
    assert adapter.active_trade is not None
    assert adapter.active_trade.bars_elapsed == 0


def test_adapter_holds_active_trade_when_no_exit_occurs():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=entry_price,
        high=entry_price,
        low=entry_price,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    holding_bar = _bar(
        "2026-01-01 10:01:00",
        close=entry_price,
        high=entry_price + 1.0,
        low=entry_price - 1.0,
        hmm_state=0,
        vol_percentile=50.0,
        zscore=0.0,
    )

    assert adapter.process_bar(entry_bar) is None
    assert adapter.active_trade is not None

    result = adapter.process_bar(holding_bar)

    assert result is None
    assert adapter.active_trade is not None
    assert adapter.active_trade.entry_price == pytest.approx(entry_price)
    assert adapter.active_trade.side == "LONG"
    assert adapter.active_trade.bars_elapsed == 1


def test_adapter_ignores_strategy_signal_while_trade_is_active():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=entry_price,
        high=entry_price,
        low=entry_price,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    second_signal_bar = _bar(
        "2026-01-01 10:01:00",
        close=entry_price,
        high=entry_price + 1.0,
        low=entry_price - 1.0,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    assert adapter.process_bar(entry_bar) is None
    assert adapter.active_trade is not None

    result = adapter.process_bar(second_signal_bar)

    assert result is None
    assert adapter.active_trade is not None
    assert adapter.active_trade.entry_price == pytest.approx(entry_price)
    assert adapter.active_trade.bars_elapsed == 1


def test_adapter_reset_closes_internal_state():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_bar = _bar(
        "2026-01-01 10:00:00",
        close=100.0,
        high=100.0,
        low=100.0,
        hmm_state=MRL1_CONFIG.hmm_state,
        vol_percentile=30.0,
        zscore=-MRL1_CONFIG.zscore_threshold,
    )

    assert adapter.process_bar(entry_bar) is None
    assert adapter.active_trade is not None

    adapter.reset()

    assert adapter.active_trade is None


def test_adapter_run_returns_completed_trades_only():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0
    target = entry_price + MRL1_CONFIG.target_points

    bars = [
        _bar(
            "2026-01-01 10:00:00",
            close=entry_price,
            high=entry_price,
            low=entry_price,
            hmm_state=MRL1_CONFIG.hmm_state,
            vol_percentile=30.0,
            zscore=-MRL1_CONFIG.zscore_threshold,
        ),
        _bar(
            "2026-01-01 10:01:00",
            close=target,
            high=target + 1.0,
            low=entry_price,
            hmm_state=0,
            vol_percentile=50.0,
            zscore=0.0,
        ),
    ]

    trades = adapter.run(bars)

    assert len(trades) == 1

    trade = trades[0]

    assert trade.strategy_name == MRL1_CONFIG.name
    assert trade.side == "LONG"
    assert trade.exit_reason == "target"
    assert trade.r_multiple == pytest.approx(MRL1_CONFIG.rr)
    assert adapter.active_trade is None


def test_adapter_run_does_not_artificially_close_open_trade():
    adapter = MeanReversionBacktestAdapter(MRL1_CONFIG)

    entry_price = 100.0

    bars = [
        _bar(
            "2026-01-01 10:00:00",
            close=entry_price,
            high=entry_price,
            low=entry_price,
            hmm_state=MRL1_CONFIG.hmm_state,
            vol_percentile=30.0,
            zscore=-MRL1_CONFIG.zscore_threshold,
        ),
    ]

    trades = adapter.run(bars)

    assert trades == []
    assert adapter.active_trade is not None
    assert adapter.active_trade.entry_price == pytest.approx(entry_price)
    assert adapter.active_trade.side == "LONG"
