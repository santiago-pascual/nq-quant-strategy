from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.execution.engine import ExecutionEngine
from src.execution.types import (
    ExecutionIntent,
    Fill,
    StrategyAction,
    StrategySignal,
)


# ============================================================================
# Paths / frozen model constants
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRADES_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "independent_reproduction"
    / "independent_reproduction_oos_trades.csv"
)

OOS_START = pd.Timestamp("2020-06-23 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="UTC")

EXPECTED_TOTAL_TRADES = 3255

EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL_R = 289.661901213


# ============================================================================
# Helpers
# ============================================================================


def utc_timestamp(value: str | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.to_pydatetime()


def signal_from_side(side: str) -> StrategySignal:
    side = side.upper()

    if side == "LONG":
        return StrategySignal.LONG

    if side == "SHORT":
        return StrategySignal.SHORT

    raise ValueError(f"Unknown side: {side}")


def opposite_signal(side: str) -> StrategySignal:
    side = side.upper()

    if side == "LONG":
        return StrategySignal.SHORT

    if side == "SHORT":
        return StrategySignal.LONG

    raise ValueError(f"Unknown side: {side}")


def make_entry_intent(
    *,
    strategy_name: str,
    side: str,
    timestamp: datetime,
) -> ExecutionIntent:

    return ExecutionIntent(
        strategy_name=strategy_name,
        strategy_version="frozen-oos-parity",
        signal=signal_from_side(side),
        action=StrategyAction.ENTER,
        timestamp=timestamp,
        reason="backtest_parity_entry",
    )


def make_exit_intent(
    *,
    strategy_name: str,
    timestamp: datetime,
) -> ExecutionIntent:

    return ExecutionIntent(
        strategy_name=strategy_name,
        strategy_version="frozen-oos-parity",
        signal=StrategySignal.FLAT,
        action=StrategyAction.EXIT,
        timestamp=timestamp,
        reason="backtest_parity_exit",
    )


def make_fill(
    *,
    order_id: str,
    strategy_name: str,
    side: StrategySignal,
    quantity: int,
    price: float,
    timestamp: datetime,
) -> Fill:

    return Fill(
        fill_id=str(uuid4()),
        order_id=order_id,
        strategy_name=strategy_name,
        side=side,
        quantity=quantity,
        price=float(price),
        timestamp=timestamp,
    )


def get_exit_timestamp(row) -> datetime:
    value = getattr(row, "exit_timestamp", None)

    if value is not None and not pd.isna(value):
        return utc_timestamp(value)

    return utc_timestamp(row.entry_timestamp)


def finite_price(value) -> bool:
    return value is not None and not pd.isna(value)


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture(scope="module")
def frozen_trades() -> pd.DataFrame:
    assert TRADES_PATH.exists(), f"Frozen OOS trade artifact not found: {TRADES_PATH}"

    df = pd.read_csv(TRADES_PATH)

    required_columns = {
        "strategy_name",
        "side",
        "entry_timestamp",
        "entry_price",
        "exit_price",
        "r_multiple",
    }

    missing = required_columns - set(df.columns)

    assert not missing, (
        f"Frozen OOS artifact is missing required columns: {sorted(missing)}"
    )

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
    )

    if "exit_timestamp" in df.columns:
        df["exit_timestamp"] = pd.to_datetime(
            df["exit_timestamp"],
            utc=True,
        )

    df = df.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


# ============================================================================
# Frozen artifact integrity
# ============================================================================


def test_frozen_oos_trade_count(
    frozen_trades: pd.DataFrame,
):
    assert len(frozen_trades) == EXPECTED_TOTAL_TRADES


def test_frozen_oos_strategy_counts(
    frozen_trades: pd.DataFrame,
):
    counts = frozen_trades["strategy_name"].value_counts().to_dict()

    assert counts == EXPECTED_COUNTS


def test_frozen_oos_boundaries(
    frozen_trades: pd.DataFrame,
):
    assert frozen_trades["entry_timestamp"].min() >= OOS_START
    assert frozen_trades["entry_timestamp"].max() <= OOS_END


# ============================================================================
# Executable trade subset
# ============================================================================


def test_frozen_executable_prices_are_identified(
    frozen_trades: pd.DataFrame,
):
    """
    The frozen artifact contains 3255 trades, but some reporting rows do not
    contain executable entry/exit prices.

    Those rows remain part of the frozen artifact and aggregate statistics,
    but cannot be replayed as actual execution fills.
    """

    executable = frozen_trades[
        frozen_trades["entry_price"].notna() & frozen_trades["exit_price"].notna()
    ]

    assert len(executable) > 0
    assert len(executable) <= EXPECTED_TOTAL_TRADES


# ============================================================================
# Full execution lifecycle
# ============================================================================


def test_every_frozen_executable_trade_completes_full_execution_lifecycle(
    frozen_trades: pd.DataFrame,
):
    """
    Replay every frozen trade with finite entry and exit prices.

    Non-executable reporting rows with NaN prices are intentionally excluded
    from this lifecycle test because no valid Fill can be constructed from
    them.
    """

    executable = frozen_trades[
        frozen_trades["entry_price"].notna() & frozen_trades["exit_price"].notna()
    ]

    engine = ExecutionEngine()

    processed = 0

    for row in executable.itertuples(index=False):
        strategy_name = str(row.strategy_name)
        side = str(row.side).upper()

        entry_price = float(row.entry_price)
        exit_price = float(row.exit_price)

        entry_time = utc_timestamp(row.entry_timestamp)
        exit_time = get_exit_timestamp(row)

        # ------------------------------------------------------------------
        # ENTRY
        # ------------------------------------------------------------------

        entry_intent = make_entry_intent(
            strategy_name=strategy_name,
            side=side,
            timestamp=entry_time,
        )

        entry_order = engine.submit_entry(
            entry_intent,
            quantity=1,
        )

        processed_entry = engine.process_entry_fill(
            entry_order.order_id,
            entry_price,
            entry_time,
        )

        assert processed_entry is not None

        position = engine.get_position(strategy_name)

        assert position is not None
        assert position.quantity == 1
        assert position.entry_price == pytest.approx(entry_price)

        # ------------------------------------------------------------------
        # EXIT
        # ------------------------------------------------------------------

        exit_intent = make_exit_intent(
            strategy_name=strategy_name,
            timestamp=exit_time,
        )

        exit_order = engine.submit_exit(exit_intent)

        processed_exit = engine.process_exit_fill(
            exit_order.order_id,
            exit_price,
            exit_time,
        )

        assert isinstance(processed_exit, tuple)
        assert len(processed_exit) == 2

        processed_fill, closed_position = processed_exit

        assert processed_fill is not None
        assert closed_position is not None

        assert closed_position.entry_price == pytest.approx(entry_price)
        assert closed_position.exit_price == pytest.approx(exit_price)

        assert engine.get_position(strategy_name) is None

        processed += 1

    assert processed == len(executable)


# ============================================================================
# Position cleanup
# ============================================================================


def test_no_position_remains_open_after_full_replay(
    frozen_trades: pd.DataFrame,
):
    """
    Every executable replayed trade must finish flat.
    """

    executable = frozen_trades[
        frozen_trades["entry_price"].notna() & frozen_trades["exit_price"].notna()
    ]

    engine = ExecutionEngine()

    for row in executable.itertuples(index=False):
        strategy_name = str(row.strategy_name)
        side = str(row.side).upper()

        entry_time = utc_timestamp(row.entry_timestamp)
        exit_time = get_exit_timestamp(row)

        entry_intent = make_entry_intent(
            strategy_name=strategy_name,
            side=side,
            timestamp=entry_time,
        )

        entry_order = engine.submit_entry(
            entry_intent,
            quantity=1,
        )

        engine.process_entry_fill(
            entry_order.order_id,
            float(row.entry_price),
            entry_time,
        )

        exit_intent = make_exit_intent(
            strategy_name=strategy_name,
            timestamp=exit_time,
        )

        exit_order = engine.submit_exit(exit_intent)

        engine.process_exit_fill(
            exit_order.order_id,
            float(row.exit_price),
            exit_time,
        )

        assert engine.get_position(strategy_name) is None


# ============================================================================
# Entry / exit price parity
# ============================================================================


def test_entry_and_exit_prices_match_frozen_backtest(
    frozen_trades: pd.DataFrame,
):
    """
    The execution layer must preserve exact frozen entry/exit prices for
    trades with valid executable prices.
    """

    executable = frozen_trades[
        frozen_trades["entry_price"].notna() & frozen_trades["exit_price"].notna()
    ]

    engine = ExecutionEngine()

    for row in executable.itertuples(index=False):
        strategy_name = str(row.strategy_name)
        side = str(row.side).upper()

        entry_price = float(row.entry_price)
        exit_price = float(row.exit_price)

        entry_time = utc_timestamp(row.entry_timestamp)
        exit_time = get_exit_timestamp(row)

        # ENTRY
        entry_intent = make_entry_intent(
            strategy_name=strategy_name,
            side=side,
            timestamp=entry_time,
        )

        entry_order = engine.submit_entry(
            entry_intent,
            quantity=1,
        )

        engine.process_entry_fill(
            entry_order.order_id,
            entry_price,
            entry_time,
        )

        position = engine.get_position(strategy_name)

        assert position is not None
        assert position.quantity == 1
        assert position.entry_price == pytest.approx(entry_price)

        # EXIT
        exit_intent = make_exit_intent(
            strategy_name=strategy_name,
            timestamp=exit_time,
        )

        exit_order = engine.submit_exit(exit_intent)

        _, closed_position = engine.process_exit_fill(
            exit_order.order_id,
            exit_price,
            exit_time,
        )

        assert closed_position.entry_price == pytest.approx(entry_price)
        assert closed_position.exit_price == pytest.approx(exit_price)


# ============================================================================
# Result-direction parity
# ============================================================================


def test_r_multiple_direction_is_preserved(
    frozen_trades: pd.DataFrame,
):
    """
    Validate result direction for trades with finite entry, exit and R.

    This checks only the sign, not R magnitude.
    """

    checked = 0

    for row in frozen_trades.itertuples(index=False):
        side = str(row.side).upper()

        entry_price = row.entry_price
        exit_price = row.exit_price
        frozen_r = row.r_multiple

        if not (
            finite_price(entry_price)
            and finite_price(exit_price)
            and finite_price(frozen_r)
        ):
            continue

        entry_price = float(entry_price)
        exit_price = float(exit_price)
        frozen_r = float(frozen_r)

        if side == "LONG":
            price_delta = exit_price - entry_price

        elif side == "SHORT":
            price_delta = entry_price - exit_price

        else:
            pytest.fail(f"Unknown frozen trade side: {side}")

        if frozen_r > 0:
            assert price_delta > 0

        elif frozen_r < 0:
            assert price_delta < 0

        else:
            assert price_delta == pytest.approx(0.0)

        checked += 1

    assert checked > 0


# ============================================================================
# Strategy-level parity
# ============================================================================


@pytest.mark.parametrize(
    "strategy_name",
    [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ],
)
def test_strategy_trade_counts_preserved(
    frozen_trades: pd.DataFrame,
    strategy_name: str,
):
    strategy_trades = frozen_trades[frozen_trades["strategy_name"] == strategy_name]

    assert len(strategy_trades) == EXPECTED_COUNTS[strategy_name]


# ============================================================================
# Aggregate R parity
# ============================================================================


def test_frozen_oos_total_r_is_unchanged(
    frozen_trades: pd.DataFrame,
):
    total_r = float(frozen_trades["r_multiple"].dropna().sum())

    assert total_r == pytest.approx(
        EXPECTED_TOTAL_R,
        abs=1e-6,
    )
