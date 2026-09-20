from __future__ import annotations

import pytest

from src.strategies.mean_reversion import (
    MRL1_CONFIG,
    MRS2_CONFIG,
)

from src.strategies.mean_reversion.lifecycle import (
    MeanReversionExitReason,
    MeanReversionLifecycle,
    MeanReversionTradeState,
)


# =============================================================================
# HELPERS
# =============================================================================


def make_state(
    *,
    entry_price: float = 100.0,
    side: str = "LONG",
    bars_elapsed: int = 0,
) -> MeanReversionTradeState:
    return MeanReversionTradeState(
        entry_price=entry_price,
        side=side,
        bars_elapsed=bars_elapsed,
    )


# =============================================================================
# MRS2
# =============================================================================


def test_mrs2_short_target():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="SHORT",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=101.0,
        low=72.5,
        close=80.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TARGET
    assert result.exit_price == pytest.approx(72.5)
    assert result.r_multiple == pytest.approx(1.10)
    assert result.bars_elapsed == 1


def test_mrs2_short_stop():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="SHORT",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=125.0,
        low=99.0,
        close=120.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(125.0)
    assert result.r_multiple == pytest.approx(-1.0)


def test_mrs2_same_bar_target_and_stop_uses_stop():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="SHORT",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=130.0,
        low=70.0,
        close=100.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.r_multiple == pytest.approx(-1.0)


def test_mrs2_does_not_exit_before_horizon():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="SHORT",
        bars_elapsed=10,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=95.0,
        close=99.0,
    )

    assert result is None


def test_mrs2_times_out_at_30_bars():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="SHORT",
        bars_elapsed=29,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=101.0,
        low=99.0,
        close=98.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TIMEOUT
    assert result.exit_price == pytest.approx(98.0)
    assert result.r_multiple == pytest.approx(0.08)
    assert result.bars_elapsed == 30


# =============================================================================
# MRL1
# =============================================================================


def test_mrl1_long_target():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="LONG",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=125.0,
        low=99.0,
        close=120.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TARGET
    assert result.exit_price == pytest.approx(125.0)
    assert result.r_multiple == pytest.approx(25.0 / 37.5)


def test_mrl1_long_stop():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="LONG",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=62.5,
        close=80.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.exit_price == pytest.approx(62.5)
    assert result.r_multiple == pytest.approx(-1.0)


def test_mrl1_same_bar_target_and_stop_uses_stop():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="LONG",
    )

    result = lifecycle.evaluate_bar(
        state,
        high=130.0,
        low=50.0,
        close=100.0,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.STOP
    assert result.r_multiple == pytest.approx(-1.0)


def test_mrl1_does_not_exit_before_horizon():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="LONG",
        bars_elapsed=3,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=95.0,
        close=101.0,
    )

    assert result is None


def test_mrl1_times_out_at_8_bars():
    lifecycle = MeanReversionLifecycle(MRL1_CONFIG)

    state = make_state(
        entry_price=100.0,
        side="LONG",
        bars_elapsed=7,
    )

    result = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=95.0,
        close=103.75,
    )

    assert result is not None
    assert result.reason is MeanReversionExitReason.TIMEOUT
    assert result.exit_price == pytest.approx(103.75)
    assert result.r_multiple == pytest.approx(3.75 / 37.5)
    assert result.bars_elapsed == 8


# =============================================================================
# VALIDATION
# =============================================================================


def test_invalid_side_is_rejected():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        side="INVALID",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Mean Reversion side",
    ):
        lifecycle.evaluate_bar(
            state,
            high=101.0,
            low=99.0,
            close=100.0,
        )


def test_invalid_high_low_is_rejected():
    lifecycle = MeanReversionLifecycle(MRS2_CONFIG)

    state = make_state(
        side="SHORT",
    )

    with pytest.raises(
        ValueError,
        match="high cannot be below low",
    ):
        lifecycle.evaluate_bar(
            state,
            high=99.0,
            low=101.0,
            close=100.0,
        )
