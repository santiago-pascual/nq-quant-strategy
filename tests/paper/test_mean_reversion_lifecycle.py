from __future__ import annotations

import pytest

from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.lifecycle import (
    MeanReversionExitReason,
    MeanReversionLifecycle,
    MeanReversionTradeState,
)


def test_long_target():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
    )

    target = 100.0 + MRL1_CONFIG.target_points

    result = lifecycle.evaluate_bar(
        state,
        high=target + 1.0,
        low=100.0,
        close=target,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TARGET
    assert result.exit_price == pytest.approx(target)
    assert result.r_multiple == pytest.approx(MRL1_CONFIG.rr)
    assert result.bars_elapsed == 1


def test_long_stop():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
    )

    stop = 100.0 - MRL1_CONFIG.stop_points

    result = lifecycle.evaluate_bar(
        state,
        high=100.0,
        low=stop - 1.0,
        close=99.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(stop)
    assert result.r_multiple == pytest.approx(-1.0)
    assert result.bars_elapsed == 1


def test_short_target():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="SHORT",
    )

    target = 100.0 - MRS2_CONFIG.target_points

    result = lifecycle.evaluate_bar(
        state,
        high=100.0,
        low=target - 1.0,
        close=target,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TARGET
    assert result.exit_price == pytest.approx(target)
    assert result.r_multiple == pytest.approx(MRS2_CONFIG.rr)
    assert result.bars_elapsed == 1


def test_short_stop():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="SHORT",
    )

    stop = 100.0 + MRS2_CONFIG.stop_points

    result = lifecycle.evaluate_bar(
        state,
        high=stop + 1.0,
        low=100.0,
        close=101.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(stop)
    assert result.r_multiple == pytest.approx(-1.0)
    assert result.bars_elapsed == 1


def test_long_same_bar_target_and_stop_stop_wins():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
    )

    target = 100.0 + MRL1_CONFIG.target_points
    stop = 100.0 - MRL1_CONFIG.stop_points

    result = lifecycle.evaluate_bar(
        state,
        high=target + 1.0,
        low=stop - 1.0,
        close=100.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(stop)
    assert result.r_multiple == pytest.approx(-1.0)
    assert result.bars_elapsed == 1


def test_short_same_bar_target_and_stop_stop_wins():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="SHORT",
    )

    target = 100.0 - MRS2_CONFIG.target_points
    stop = 100.0 + MRS2_CONFIG.stop_points

    result = lifecycle.evaluate_bar(
        state,
        high=stop + 1.0,
        low=target - 1.0,
        close=100.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(stop)
    assert result.r_multiple == pytest.approx(-1.0)
    assert result.bars_elapsed == 1


def test_long_trade_remains_open_before_horizon():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
        bars_elapsed=0,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=100.0 + MRL1_CONFIG.target_points - 1.0,
        low=100.0 - MRL1_CONFIG.stop_points + 1.0,
        close=100.0,
    )

    assert result is None


def test_short_trade_remains_open_before_horizon():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="SHORT",
        bars_elapsed=0,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=100.0 + MRS2_CONFIG.stop_points - 1.0,
        low=100.0 - MRS2_CONFIG.target_points + 1.0,
        close=100.0,
    )

    assert result is None


def test_long_timeout_uses_close():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
        bars_elapsed=MRL1_CONFIG.horizon_bars - 1,
    )

    close = 108.0

    result = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=99.0,
        close=close,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TIMEOUT
    assert result.exit_price == pytest.approx(close)
    assert result.bars_elapsed == MRL1_CONFIG.horizon_bars
    assert result.r_multiple == pytest.approx((close - 100.0) / MRL1_CONFIG.stop_points)


def test_short_timeout_uses_close():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="SHORT",
        bars_elapsed=MRS2_CONFIG.horizon_bars - 1,
    )

    close = 92.0

    result = lifecycle.evaluate_bar(
        state,
        high=101.0,
        low=95.0,
        close=close,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TIMEOUT
    assert result.exit_price == pytest.approx(close)
    assert result.bars_elapsed == MRS2_CONFIG.horizon_bars
    assert result.r_multiple == pytest.approx((100.0 - close) / MRS2_CONFIG.stop_points)


def test_invalid_side_is_rejected():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="INVALID",
    )

    with pytest.raises(ValueError, match="Unsupported Mean Reversion side"):
        lifecycle.evaluate_bar(
            state,
            high=101.0,
            low=99.0,
            close=100.0,
        )


def test_invalid_high_low_is_rejected():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=100.0,
        side="LONG",
    )

    with pytest.raises(ValueError, match="high cannot be below low"):
        lifecycle.evaluate_bar(
            state,
            high=99.0,
            low=101.0,
            close=100.0,
        )


def test_invalid_entry_price_is_rejected():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = MeanReversionTradeState(
        entry_price=0.0,
        side="LONG",
    )

    with pytest.raises(ValueError, match="entry_price must be positive"):
        lifecycle.evaluate_bar(
            state,
            high=101.0,
            low=99.0,
            close=100.0,
        )
