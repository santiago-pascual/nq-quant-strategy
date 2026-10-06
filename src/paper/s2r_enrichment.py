"""Paper-side, post-execution S26 analytical enrichment for S2R trades."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.strategies.s2r.config import S2RConfig
from src.strategies.s2r.recovery import (
    RecoveryConfig,
    RecoveryModel,
    RecoveryState,
)


S4_EARLY_ADVERSE_THRESHOLD_R = 0.75
S4_DECISION_BAR = 8

ENRICHMENT_COLUMNS = (
    "s2r_enrichment_eligible",
    "s2r_recovery_state",
    "s2r_recovery_exit_type",
    "s2r_mae_r",
    "s2r_max_mae_r",
    "s2r_mae_bar",
    "s2r_recovery_bar",
    "s2r_exit_bar",
    "s2r_strategy_R",
)


def _empty_enrichment() -> dict[str, Any]:
    return {
        "s2r_enrichment_eligible": False,
        "s2r_recovery_state": "NO_RECOVERY_ENRICHMENT",
        "s2r_recovery_exit_type": "ORIGINAL_S2",
        "s2r_mae_r": np.nan,
        "s2r_max_mae_r": np.nan,
        "s2r_mae_bar": np.nan,
        "s2r_recovery_bar": np.nan,
        "s2r_exit_bar": np.nan,
        "s2r_strategy_R": np.nan,
    }


def enrich_s2r_trade_from_bars(
    trade: Mapping[str, Any],
    market_bars: pd.DataFrame,
    *,
    config: S2RConfig | None = None,
) -> dict[str, Any]:
    """Compute Research-faithful S4/S26 metadata from Paper's own OHLC bars.

    The Paper trade supplies only its own entry timestamp and fill price. The
    executable exit fields are deliberately not read. Analysis uses the entry
    bar plus up to the next 20 bars in that trading session, so it can finish
    after the Paper position has closed.
    """

    config = config or S2RConfig()
    required = {"timestamp", "high", "close"}
    missing = required - set(market_bars.columns)
    if missing:
        raise ValueError(f"S2R enrichment market data lacks columns: {sorted(missing)}")
    if market_bars.empty:
        return _empty_enrichment()

    timestamps = pd.to_datetime(market_bars["timestamp"], utc=True, errors="raise")
    local_timestamps = timestamps.dt.tz_convert("America/New_York")
    session_ids = (local_timestamps - pd.Timedelta(hours=18)).dt.date.astype(str)
    entry_timestamp = pd.Timestamp(trade["entry_timestamp"])
    if entry_timestamp.tzinfo is None:
        entry_timestamp = entry_timestamp.tz_localize("UTC")
    else:
        entry_timestamp = entry_timestamp.tz_convert("UTC")
    entry_positions = np.flatnonzero(timestamps.eq(entry_timestamp).to_numpy())
    if len(entry_positions) != 1:
        raise ValueError(
            "Paper S2R entry must map to exactly one replay bar: "
            f"{entry_timestamp} (matches={len(entry_positions)})"
        )

    entry_position = int(entry_positions[0])
    session = market_bars.loc[session_ids.eq(session_ids.iloc[entry_position])]
    session_positions = np.flatnonzero(
        timestamps.loc[session.index].eq(entry_timestamp).to_numpy()
    )
    if len(session_positions) != 1:
        raise ValueError(f"S2R entry is not unique within its Research session: {entry_timestamp}")
    local_entry_position = int(session_positions[0])
    future = session.iloc[
        local_entry_position : local_entry_position + config.horizon_bars + 1
    ]

    if len(future) < 2:
        return _empty_enrichment()

    entry_price = float(trade["entry_price"])
    stop_points = float(config.stop_points)
    highs = pd.to_numeric(future["high"], errors="raise").to_numpy(dtype=float)
    closes = pd.to_numeric(future["close"], errors="raise").to_numpy(dtype=float)

    # S3 numbers bars from the entry (bar 0) and excludes bar 0 from MAE.
    # A zero-valued entry element lets the shared RecoveryModel use the same
    # bar numbers and recovery-after-MAE convention.
    post_entry_mae_r = (highs[1:] - entry_price) / stop_points
    cumulative_mae_r = np.maximum.accumulate(post_entry_mae_r)
    post_entry_close_r = (entry_price - closes[1:]) / stop_points

    result = _empty_enrichment()
    baseline_r = trade.get("r_multiple")
    if baseline_r is not None and pd.notna(baseline_r):
        result["s2r_strategy_R"] = float(baseline_r)
    early_mae_r = (
        float(cumulative_mae_r[S4_DECISION_BAR - 1])
        if len(cumulative_mae_r) >= S4_DECISION_BAR
        else np.nan
    )
    if not np.isfinite(early_mae_r) or early_mae_r < S4_EARLY_ADVERSE_THRESHOLD_R:
        result["s2r_max_mae_r"] = (
            float(np.max(cumulative_mae_r)) if len(cumulative_mae_r) else np.nan
        )
        return result

    # RecoveryModel uses sequence indices as bar numbers. Insert bar 0 so its
    # MAE, recovery, and deadline indices match the Research S3/S26 convention.
    decision = RecoveryModel(
        RecoveryConfig(
            mae_threshold_r=config.mae_threshold_r,
            recovery_level_r=config.recovery_level_r,
            deadline_bars=config.recovery_deadline_bars,
        )
    ).evaluate(
        [0.0, *post_entry_close_r.tolist()],
        [0.0, *post_entry_mae_r.tolist()],
    )
    if decision.state is RecoveryState.INITIAL or decision.mae_bar is None:
        raise RuntimeError("S4-eligible S2R path did not cross the S26 MAE threshold.")

    state_to_exit_type = {
        RecoveryState.RECOVERED: "RECOVERY_EXIT",
        RecoveryState.FAILED_TO_RECOVER: "FAILURE_EXIT",
    }
    if decision.state not in state_to_exit_type or decision.exit_bar is None:
        raise RuntimeError(f"Unexpected S26 recovery result: {decision}")

    result.update(
        {
            "s2r_enrichment_eligible": True,
            "s2r_recovery_state": decision.state.name,
            "s2r_recovery_exit_type": state_to_exit_type[decision.state],
            "s2r_mae_r": float(cumulative_mae_r[decision.mae_bar - 1]),
            "s2r_max_mae_r": float(np.max(cumulative_mae_r)),
            "s2r_mae_bar": int(decision.mae_bar),
            "s2r_recovery_bar": (
                np.nan
                if decision.recovery_bar is None
                else int(decision.recovery_bar)
            ),
            "s2r_exit_bar": int(decision.exit_bar),
            "s2r_strategy_R": float(post_entry_close_r[decision.exit_bar - 1]),
        }
    )
    return result


def enrich_paper_s2r_trades(
    paper_trades: pd.DataFrame,
    market_bars: pd.DataFrame,
    *,
    config: S2RConfig | None = None,
) -> pd.DataFrame:
    """Attach S26 analytics to Paper-generated trades without changing exits."""

    enriched = paper_trades.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in enriched:
            enriched[column] = pd.Series(
                [None] * len(enriched), index=enriched.index, dtype=object
            )
    if enriched.empty or "strategy_name" not in enriched:
        return enriched

    mask = enriched["strategy_name"].eq("S2R")
    for index, trade in enriched.loc[mask].iterrows():
        metadata = enrich_s2r_trade_from_bars(
            trade.to_dict(),
            market_bars,
            config=config,
        )
        for column, value in metadata.items():
            enriched.at[index, column] = value
    return enriched
