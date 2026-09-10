"""
08AE — LOCAL PARAMETER PERTURBATION ROBUSTNESS

Purpose
-------
Validate the robustness of the NEW mean-reversion parameter branch.

IMPORTANT
---------
- The historical 5/2 branch is NOT evaluated.
- The frozen event populations come directly from 08AA.
- Parameter perturbations are evaluated on the EXACT SAME event populations
  as the frozen candidates.
- This script does NOT re-select events.
- This script does NOT optimize parameters.
- This script performs a predefined local sensitivity test.

Candidates
----------
MRS2_NEW:
    SHORT
    HMM state 2
    VOL 80-100
    Z >= 2.0
    TP 27.5
    SL 25.0
    H 30

MRL1_NEW:
    LONG
    HMM state 1
    VOL 20-40
    Z <= -2.5
    TP 25.0
    SL 37.5
    H 8

Outputs
-------
results/research_08ae_parameter_perturbation.csv
results/research_08ae_parameter_perturbation_windows.csv
results/research_08ae_local_sensitivity.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

RESEARCH_07_EVENTS = CACHE_DIR / "research_07_event_metadata.csv"
FROZEN_08AA = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"


# =============================================================================
# OUTPUTS
# =============================================================================

OUTPUT_SUMMARY = RESULTS_DIR / "research_08ae_parameter_perturbation.csv"
OUTPUT_WINDOWS = RESULTS_DIR / "research_08ae_parameter_perturbation_windows.csv"
OUTPUT_SENSITIVITY = RESULTS_DIR / "research_08ae_local_sensitivity.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

FROZEN = {
    "MRS2_NEW": {
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "VOL80-100",
        "zscore_threshold": 2.0,
        "tp": 27.5,
        "sl": 25.0,
        "horizon": 30,
    },
    "MRL1_NEW": {
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "VOL20-40",
        "zscore_threshold": 2.5,
        "tp": 25.0,
        "sl": 37.5,
        "horizon": 8,
    },
}


# =============================================================================
# LOCAL PERTURBATION GRID
# =============================================================================
#
# This is deliberately local around the NEW branch.
#
# MRS2:
#   TP 25 / 27.5 / 30
#   SL 22.5 / 25 / 27.5
#   H 20 / 30
#
# MRL1:
#   TP 20 / 25 / 30
#   SL 30 / 37.5 / 45
#   H 6 / 8 / 10
#
# The frozen configuration is included in each neighborhood.
#
# NO optimization is performed.
# =============================================================================

PERTURATIONS = {
    "MRS2_NEW": {
        "tp": [25.0, 27.5, 30.0],
        "sl": [22.5, 25.0, 27.5],
        "horizon": [20, 30],
    },
    "MRL1_NEW": {
        "tp": [20.0, 25.0, 30.0],
        "sl": [30.0, 37.5, 45.0],
        "horizon": [6, 8, 10],
    },
}


# =============================================================================
# CONSTANTS
# =============================================================================

TOLERANCE = 1e-9


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def status(label: str, passed: bool) -> None:
    print(f"{label:<55}: {'PASS' if passed else 'FAIL'}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_market() -> pd.DataFrame:
    """
    Load the canonical market dataset through the same loader used by the
    Research 07 pipeline.
    """

    from src.data_loader import load_data

    market = load_data()

    if not isinstance(market, pd.DataFrame):
        raise TypeError(
            f"load_data() returned {type(market)}, expected pandas DataFrame."
        )

    required = {
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(market.columns)

    if missing:
        raise ValueError(f"Canonical market data missing columns: {sorted(missing)}")

    print(f"Market rows: {len(market):,}")

    return market


# =============================================================================
# RTH PREPARATION
# =============================================================================


def prepare_rth(market: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the Research 07 RTH preparation logic.

    Research 07 uses market_period == RTH.
    """

    df = market.copy()

    if "market_period" not in df.columns:
        raise ValueError("Canonical market data does not contain 'market_period'.")

    df["timestamp ET"] = pd.to_datetime(
        df["timestamp ET"],
        errors="coerce",
    )

    df = df[df["market_period"] == "RTH"].copy()

    df = df.sort_values("timestamp ET").reset_index(drop=True)

    return df


# =============================================================================
# EVENT -> RTH MAPPING
# =============================================================================


def validate_event_mapping(
    events: pd.DataFrame,
    rth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the Research 07 event -> canonical RTH mapping.

    IMPORTANT
    ---------
    Research 07 event timestamps are UTC.
    Canonical `timestamp ET` values are New York local time.

    The authoritative mapping is `data_index`, which points directly to the
    corresponding row in the RTH dataframe.

    Timestamp and close are then independently validated after timezone
    normalization.
    """

    required = {
        "event_id",
        "data_index",
        "timestamp",
        "close",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(f"Research 07 metadata missing columns: {sorted(missing)}")

    # -------------------------------------------------------------------------
    # Identity checks
    # -------------------------------------------------------------------------

    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id values found.")

    if events["data_index"].duplicated().any():
        raise ValueError("Duplicate data_index values found.")

    if events["data_index"].min() < 0:
        raise ValueError("Negative data_index detected.")

    if events["data_index"].max() >= len(rth):
        raise ValueError("Some data_index values fall outside the RTH dataframe.")

    mapped = events.copy()

    mapped["rth_position"] = mapped["data_index"].astype(int)

    # -------------------------------------------------------------------------
    # TIMESTAMP VALIDATION
    # -------------------------------------------------------------------------
    #
    # Research 07:
    #     timestamp -> UTC
    #
    # Canonical market:
    #     timestamp ET -> America/New_York
    #
    # Convert both to UTC before comparison.
    # -------------------------------------------------------------------------

    event_ts = pd.to_datetime(
        mapped["timestamp"],
        errors="coerce",
        utc=True,
    )

    rth_ts_et = pd.to_datetime(
        rth["timestamp ET"],
        errors="coerce",
    )

    # If the canonical ET timestamps are naive, they represent New York time.
    if rth_ts_et.dt.tz is None:
        rth_ts_utc = rth_ts_et.dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT",
        ).dt.tz_convert("UTC")

    else:
        rth_ts_utc = rth_ts_et.dt.tz_convert("UTC")

    expected_ts = rth_ts_utc.iloc[mapped["rth_position"].to_numpy()].reset_index(
        drop=True
    )

    actual_ts = event_ts.reset_index(drop=True)

    timestamp_match = actual_ts.eq(expected_ts).to_numpy()

    # Ignore rows where either side is NaT only for the purpose of producing
    # the diagnostic. They are still treated as mismatches below.
    timestamp_match = (
        timestamp_match & actual_ts.notna().to_numpy() & expected_ts.notna().to_numpy()
    )

    status(
        "Research 07 event -> RTH mapping",
        bool(timestamp_match.all()),
    )

    if not timestamp_match.all():
        bad = np.flatnonzero(~timestamp_match)

        first = int(bad[0])

        print()
        print("First timestamp mismatch:")
        print(f"  event_id       : {mapped.iloc[first]['event_id']}")
        print(f"  data_index     : {mapped.iloc[first]['data_index']}")
        print(f"  event timestamp: {actual_ts.iloc[first]}")
        print(f"  RTH timestamp  : {expected_ts.iloc[first]}")

        raise ValueError(
            f"Event/RTH timestamp mismatch detected. First mismatch position: {first}"
        )

    # -------------------------------------------------------------------------
    # CLOSE VALIDATION
    # -------------------------------------------------------------------------

    expected_close = (
        pd.to_numeric(
            rth["close"],
            errors="coerce",
        )
        .iloc[mapped["rth_position"].to_numpy()]
        .reset_index(drop=True)
        .to_numpy(dtype=float)
    )

    actual_close = (
        pd.to_numeric(
            mapped["close"],
            errors="coerce",
        )
        .reset_index(drop=True)
        .to_numpy(dtype=float)
    )

    close_match = np.isclose(
        actual_close,
        expected_close,
        atol=TOLERANCE,
        rtol=0.0,
        equal_nan=False,
    )

    status(
        "Research 07 event close -> RTH close",
        bool(close_match.all()),
    )

    if not close_match.all():
        bad = np.flatnonzero(~close_match)

        first = int(bad[0])

        print()
        print("First close mismatch:")
        print(f"  event_id        : {mapped.iloc[first]['event_id']}")
        print(f"  data_index      : {mapped.iloc[first]['data_index']}")
        print(f"  event close     : {actual_close[first]}")
        print(f"  RTH close       : {expected_close[first]}")

        raise ValueError(
            f"Event/RTH close mismatch detected. First mismatch position: {first}"
        )

    return mapped


# =============================================================================
# FUTURE OHLC
# =============================================================================


def build_future_ohlc(
    rth: pd.DataFrame,
    max_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build future high/low/close matrices.

    Row i corresponds exactly to RTH row i.

    Column 0 = bar +1
    Column 1 = bar +2
    ...
    """

    n = len(rth)

    high = pd.to_numeric(
        rth["high"],
        errors="coerce",
    ).to_numpy(dtype=float)

    low = pd.to_numeric(
        rth["low"],
        errors="coerce",
    ).to_numpy(dtype=float)

    close = pd.to_numeric(
        rth["close"],
        errors="coerce",
    ).to_numpy(dtype=float)

    future_high = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    future_low = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    future_close = np.full(
        (n, max_horizon),
        np.nan,
        dtype=float,
    )

    for h in range(1, max_horizon + 1):
        future_high[:-h, h - 1] = high[h:]
        future_low[:-h, h - 1] = low[h:]
        future_close[:-h, h - 1] = close[h:]

    return future_high, future_low, future_close


# =============================================================================
# RESULT LOGIC
# =============================================================================


def evaluate_trade(
    side: str,
    entry: float,
    tp: float,
    sl: float,
    horizon: int,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
) -> Tuple[str, float, int]:
    """
    Evaluate one trade using intrabar OHLC.

    Barrier convention
    ------------------
    If TP and SL are both touched on the same bar:
        STOP FIRST

    Timeout convention
    ------------------
    At the final horizon close:

        final_r > 0  -> TIMEOUT_WIN
        final_r < 0  -> TIMEOUT_LOSS
        final_r == 0  -> UNRESOLVED

    R normalization
    ----------------
    +TP/SL for winners
    -1 for SL losses
    timeout = final close movement / SL
    """

    if not np.isfinite(entry):
        return "UNRESOLVED", 0.0, 0

    if sl <= 0 or tp <= 0:
        raise ValueError("TP and SL must be positive.")

    if horizon <= 0:
        raise ValueError("Horizon must be positive.")

    available = min(
        horizon,
        len(future_high),
        len(future_low),
        len(future_close),
    )

    if available == 0:
        return "UNRESOLVED", 0.0, 0

    if side == "LONG":
        tp_price = entry + tp
        sl_price = entry - sl

        for i in range(available):
            hi = future_high[i]
            lo = future_low[i]

            if not np.isfinite(hi) or not np.isfinite(lo):
                continue

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

            # Same-bar conflict: STOP FIRST.
            if hit_tp and hit_sl:
                return "LOSS", -1.0, i + 1

            if hit_sl:
                return "LOSS", -1.0, i + 1

            if hit_tp:
                return "WIN", tp / sl, i + 1

        final_close = future_close[available - 1]

        if not np.isfinite(final_close):
            return "UNRESOLVED", 0.0, available

        final_r = (final_close - entry) / sl

    elif side == "SHORT":
        tp_price = entry - tp
        sl_price = entry + sl

        for i in range(available):
            hi = future_high[i]
            lo = future_low[i]

            if not np.isfinite(hi) or not np.isfinite(lo):
                continue

            hit_tp = lo <= tp_price
            hit_sl = hi >= sl_price

            # Same-bar conflict: STOP FIRST.
            if hit_tp and hit_sl:
                return "LOSS", -1.0, i + 1

            if hit_sl:
                return "LOSS", -1.0, i + 1

            if hit_tp:
                return "WIN", tp / sl, i + 1

        final_close = future_close[available - 1]

        if not np.isfinite(final_close):
            return "UNRESOLVED", 0.0, available

        final_r = (entry - final_close) / sl

    else:
        raise ValueError(f"Unknown side: {side}")

    if final_r > TOLERANCE:
        return "TIMEOUT_WIN", float(final_r), available

    if final_r < -TOLERANCE:
        return "TIMEOUT_LOSS", float(final_r), available

    return "UNRESOLVED", 0.0, available


# =============================================================================
# METRICS
# =============================================================================


def profit_factor(r_values: np.ndarray) -> float:
    wins = r_values[r_values > 0]
    losses = r_values[r_values < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss <= TOLERANCE:
        if gross_profit > TOLERANCE:
            return np.inf
        return np.nan

    return float(gross_profit / gross_loss)


def max_drawdown(r_values: np.ndarray) -> float:
    if len(r_values) == 0:
        return 0.0

    equity = np.cumsum(r_values)

    running_max = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_metrics(
    r_values: np.ndarray,
) -> Dict[str, float]:

    r_values = np.asarray(r_values, dtype=float)

    n = len(r_values)

    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "unresolved": 0,
            "win_rate": np.nan,
            "total_R": 0.0,
            "expectancy_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    wins = int((r_values > 0).sum())
    losses = int((r_values < 0).sum())
    unresolved = int((np.abs(r_values) <= TOLERANCE).sum())

    total_r = float(r_values.sum())
    mean_r = float(r_values.mean())

    pf = profit_factor(r_values)
    dd = max_drawdown(r_values)

    # Win rate is based on positive outcomes among resolved trades.
    resolved = wins + losses

    if resolved > 0:
        wr = wins / resolved
    else:
        wr = np.nan

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "timeouts": 0,
        "unresolved": unresolved,
        "win_rate": wr,
        "total_R": total_r,
        "expectancy_R": mean_r,
        "profit_factor": pf,
        "max_drawdown_R": dd,
    }


# =============================================================================
# FROZEN REPRODUCTION
# =============================================================================


def reproduce_frozen_candidate(
    candidate_id: str,
    frozen_trades: pd.DataFrame,
    event_map: pd.DataFrame,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
) -> pd.DataFrame:
    """
    Reconstruct the exact frozen 08AA candidate.

    This is a hard integrity gate.

    If the frozen 08AA trades cannot be reproduced exactly, the perturbation
    analysis is aborted.
    """

    frozen = FROZEN[candidate_id]

    df = frozen_trades[frozen_trades["candidate_id"] == candidate_id].copy()

    if df.empty:
        raise ValueError(f"No frozen trades found for candidate {candidate_id}.")

    rows = []

    event_lookup = event_map.set_index("event_id")

    for _, trade in df.iterrows():
        event_id = int(trade["event_id"])

        if event_id not in event_lookup.index:
            raise ValueError(f"event_id {event_id} not found in Research 07 metadata.")

        event = event_lookup.loc[event_id]

        rth_position = int(event["rth_position"])

        entry = float(event["close"])

        result, r_value, bars = evaluate_trade(
            side=frozen["side"],
            entry=entry,
            tp=frozen["tp"],
            sl=frozen["sl"],
            horizon=frozen["horizon"],
            future_high=future_high[rth_position],
            future_low=future_low[rth_position],
            future_close=future_close[rth_position],
        )

        rows.append(
            {
                "candidate_id": candidate_id,
                "event_id": event_id,
                "stored_result": str(trade["result"]),
                "reconstructed_result": result,
                "stored_r": float(trade["r"]),
                "reconstructed_r": float(r_value),
                "stored_bars": int(trade["bars"]),
                "reconstructed_bars": int(bars),
            }
        )

    reproduction = pd.DataFrame(rows)

    result_match = reproduction["stored_result"] == reproduction["reconstructed_result"]

    r_match = np.isclose(
        reproduction["stored_r"].to_numpy(dtype=float),
        reproduction["reconstructed_r"].to_numpy(dtype=float),
        atol=TOLERANCE,
        rtol=0.0,
    )

    bars_match = reproduction["stored_bars"].to_numpy(dtype=int) == reproduction[
        "reconstructed_bars"
    ].to_numpy(dtype=int)

    all_match = result_match & r_match & bars_match

    print()
    print(f"{candidate_id} frozen reproduction")
    print(f"  Trades             : {len(reproduction):,}")
    print(f"  Result mismatches  : {int((~result_match).sum())}")
    print(f"  R mismatches       : {int((~r_match).sum())}")
    print(f"  Bars mismatches    : {int((~bars_match).sum())}")
    print(f"  ALL MATCH          : {int(all_match.sum())}/{len(all_match)}")

    if not all_match.all():
        print()
        print("First reproduction mismatches:")

        print(reproduction.loc[~all_match].head(10).to_string(index=False))

        raise RuntimeError(f"{candidate_id} frozen reproduction FAILED.")

    print(f"  {candidate_id}: PASS")

    return reproduction


# =============================================================================
# EVALUATE PERTURBATION
# =============================================================================


def evaluate_configuration(
    candidate_id: str,
    tp: float,
    sl: float,
    horizon: int,
    frozen_events: pd.DataFrame,
    event_map: pd.DataFrame,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    frozen = FROZEN[candidate_id]

    event_lookup = event_map.set_index("event_id")

    rows = []

    for _, event_trade in frozen_events.iterrows():
        event_id = int(event_trade["event_id"])

        event = event_lookup.loc[event_id]

        rth_position = int(event["rth_position"])

        entry = float(event["close"])

        result, r_value, bars = evaluate_trade(
            side=frozen["side"],
            entry=entry,
            tp=tp,
            sl=sl,
            horizon=horizon,
            future_high=future_high[rth_position],
            future_low=future_low[rth_position],
            future_close=future_close[rth_position],
        )

        rows.append(
            {
                "candidate_id": candidate_id,
                "event_id": event_id,
                "timestamp": event_trade["timestamp"],
                "window": int(event_trade["window"]),
                "side": frozen["side"],
                "tp": tp,
                "sl": sl,
                "rr": tp / sl,
                "horizon": horizon,
                "result": result,
                "r": r_value,
                "bars": bars,
            }
        )

    trades = pd.DataFrame(rows)

    r_values = trades["r"].to_numpy(dtype=float)

    metrics = calculate_metrics(r_values)

    # Add result-category counts.
    metrics["timeouts"] = int(
        trades["result"].astype(str).str.startswith("TIMEOUT").sum()
    )

    metrics["unresolved"] = int((trades["result"] == "UNRESOLVED").sum())

    # Window metrics.
    window_rows = []

    for window, group in trades.groupby("window", sort=True):
        wr = group["r"].to_numpy(dtype=float)

        wm = calculate_metrics(wr)

        window_rows.append(
            {
                "candidate_id": candidate_id,
                "tp": tp,
                "sl": sl,
                "rr": tp / sl,
                "horizon": horizon,
                "window": int(window),
                "trades": wm["trades"],
                "wins": wm["wins"],
                "losses": wm["losses"],
                "timeouts": int(
                    group["result"].astype(str).str.startswith("TIMEOUT").sum()
                ),
                "unresolved": int((group["result"] == "UNRESOLVED").sum()),
                "total_R": wm["total_R"],
                "expectancy_R": wm["expectancy_R"],
                "profit_factor": wm["profit_factor"],
                "win_rate": wm["win_rate"],
            }
        )

    windows = pd.DataFrame(window_rows)

    positive_windows = int((windows["total_R"] > 0).sum())

    active_windows = int(len(windows))

    if active_windows > 0:
        positive_window_rate = positive_windows / active_windows
    else:
        positive_window_rate = np.nan

    metrics["positive_windows"] = positive_windows
    metrics["active_windows"] = active_windows
    metrics["positive_window_rate"] = positive_window_rate

    return trades, windows, metrics


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("08AE — LOCAL PARAMETER PERTURBATION ROBUSTNESS")

    print()
    print("NEW BRANCH ONLY")
    print("Old 5/2 parameter branch is NOT evaluated.")

    # -------------------------------------------------------------------------
    # Load frozen 08AA
    # -------------------------------------------------------------------------

    banner("LOADING FROZEN 08AA")

    if not FROZEN_08AA.exists():
        raise FileNotFoundError(f"Missing frozen 08AA file:\n{FROZEN_08AA}")

    frozen_trades = pd.read_csv(FROZEN_08AA)

    print(f"08AA trades: {len(frozen_trades):,}")

    required_frozen = {
        "candidate_id",
        "event_id",
        "timestamp",
        "window",
        "side",
        "tp",
        "sl",
        "rr",
        "horizon",
        "result",
        "r",
        "bars",
    }

    missing = required_frozen - set(frozen_trades.columns)

    if missing:
        raise ValueError(f"08AA file missing columns: {sorted(missing)}")

    # -------------------------------------------------------------------------
    # Load Research 07 event metadata
    # -------------------------------------------------------------------------

    banner("LOADING RESEARCH 07 EVENT METADATA")

    if not RESEARCH_07_EVENTS.exists():
        raise FileNotFoundError(
            f"Missing Research 07 event metadata:\n{RESEARCH_07_EVENTS}"
        )

    events = pd.read_csv(RESEARCH_07_EVENTS)

    print(f"Research 07 events: {len(events):,}")

    # -------------------------------------------------------------------------
    # Load canonical market
    # -------------------------------------------------------------------------

    banner("LOADING CANONICAL MARKET")

    market = load_market()

    rth = prepare_rth(market)

    print(f"RTH rows: {len(rth):,}")

    # -------------------------------------------------------------------------
    # Event mapping
    # -------------------------------------------------------------------------

    event_map = validate_event_mapping(
        events,
        rth,
    )

    # -------------------------------------------------------------------------
    # Validate candidate populations
    # -------------------------------------------------------------------------

    banner("VALIDATING FROZEN EVENT POPULATIONS")

    for candidate_id in FROZEN:
        df = frozen_trades[frozen_trades["candidate_id"] == candidate_id].copy()

        expected = FROZEN[candidate_id]

        if len(df) == 0:
            raise ValueError(f"No frozen trades for {candidate_id}.")

        side_ok = (df["side"] == expected["side"]).all()

        tp_ok = np.isclose(
            df["tp"].astype(float),
            expected["tp"],
            atol=TOLERANCE,
            rtol=0.0,
        ).all()

        sl_ok = np.isclose(
            df["sl"].astype(float),
            expected["sl"],
            atol=TOLERANCE,
            rtol=0.0,
        ).all()

        horizon_ok = (df["horizon"].astype(int) == expected["horizon"]).all()

        status(
            f"{candidate_id} side",
            bool(side_ok),
        )

        status(
            f"{candidate_id} TP",
            bool(tp_ok),
        )

        status(
            f"{candidate_id} SL",
            bool(sl_ok),
        )

        status(
            f"{candidate_id} horizon",
            bool(horizon_ok),
        )

        if not (side_ok and tp_ok and sl_ok and horizon_ok):
            raise RuntimeError(f"{candidate_id} frozen parameter definition mismatch.")

    # -------------------------------------------------------------------------
    # Maximum horizon
    # -------------------------------------------------------------------------

    max_horizon = max(max(PERTURATIONS[c]["horizon"]) for c in PERTURATIONS)

    print()
    print(f"Maximum perturbation horizon: {max_horizon}")

    # -------------------------------------------------------------------------
    # Future OHLC
    # -------------------------------------------------------------------------

    banner("BUILDING FUTURE OHLC")

    future_high, future_low, future_close = build_future_ohlc(
        rth,
        max_horizon,
    )

    print(f"Future high shape : {future_high.shape}")
    print(f"Future low shape  : {future_low.shape}")
    print(f"Future close shape: {future_close.shape}")

    # -------------------------------------------------------------------------
    # Frozen reproduction gate
    # -------------------------------------------------------------------------

    banner("FROZEN REPRODUCTION GATE")

    for candidate_id in FROZEN:
        reproduce_frozen_candidate(
            candidate_id=candidate_id,
            frozen_trades=frozen_trades,
            event_map=event_map,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
        )

    print()
    print("Frozen reproduction gate: PASS")
    print("Proceeding to local parameter perturbations.")

    # -------------------------------------------------------------------------
    # Perturbation evaluation
    # -------------------------------------------------------------------------

    banner("RUNNING LOCAL PARAMETER PERTURBATIONS")

    summary_rows = []
    all_window_rows = []
    all_trade_rows = []

    for candidate_id in FROZEN:
        candidate_trades = frozen_trades[
            frozen_trades["candidate_id"] == candidate_id
        ].copy()

        grid = PERTURATIONS[candidate_id]

        total_configs = len(grid["tp"]) * len(grid["sl"]) * len(grid["horizon"])

        print()
        print(f"{candidate_id}")
        print(f"TP values      : {grid['tp']}")
        print(f"SL values      : {grid['sl']}")
        print(f"Horizon values : {grid['horizon']}")
        print(f"Configurations : {total_configs}")

        completed = 0

        for tp in grid["tp"]:
            for sl in grid["sl"]:
                for horizon in grid["horizon"]:
                    completed += 1

                    trades, windows, metrics = evaluate_configuration(
                        candidate_id=candidate_id,
                        tp=float(tp),
                        sl=float(sl),
                        horizon=int(horizon),
                        frozen_events=candidate_trades,
                        event_map=event_map,
                        future_high=future_high,
                        future_low=future_low,
                        future_close=future_close,
                    )

                    row = {
                        "candidate_id": candidate_id,
                        "strategy_name": candidate_id,
                        "tp": float(tp),
                        "sl": float(sl),
                        "rr": float(tp / sl),
                        "horizon": int(horizon),
                        **metrics,
                    }

                    summary_rows.append(row)

                    all_window_rows.append(windows)

                    # Keep trade-level output for reproducibility.
                    all_trade_rows.append(trades)

        print(f"Completed {completed}/{total_configs}")

    summary = pd.DataFrame(summary_rows)

    windows = pd.concat(
        all_window_rows,
        ignore_index=True,
    )

    trade_results = pd.concat(
        all_trade_rows,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Integrity checks
    # -------------------------------------------------------------------------

    banner("PERTURBATION INTEGRITY CHECKS")

    expected_configs = sum(
        len(v["tp"]) * len(v["sl"]) * len(v["horizon"]) for v in PERTURATIONS.values()
    )

    actual_configs = len(summary)

    status(
        "Expected configuration count",
        actual_configs == expected_configs,
    )

    if actual_configs != expected_configs:
        raise RuntimeError(
            f"Expected {expected_configs} configurations, got {actual_configs}."
        )

    # Check finite core metrics.
    metric_columns = [
        "trades",
        "wins",
        "losses",
        "timeouts",
        "unresolved",
        "win_rate",
        "total_R",
        "expectancy_R",
        "profit_factor",
        "max_drawdown_R",
        "positive_windows",
        "active_windows",
        "positive_window_rate",
    ]

    finite = np.isfinite(
        summary[metric_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    ).all()

    # Profit factor can legitimately be infinite if there are no losses.
    pf_finite_or_inf = (
        np.isfinite(
            summary["profit_factor"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        ).all()
        or np.isinf(summary["profit_factor"].to_numpy(dtype=float)).any()
    )

    status(
        "All non-PF metrics finite",
        bool(
            np.isfinite(
                summary[
                    [
                        "trades",
                        "wins",
                        "losses",
                        "timeouts",
                        "unresolved",
                        "win_rate",
                        "total_R",
                        "expectancy_R",
                        "max_drawdown_R",
                        "positive_windows",
                        "active_windows",
                        "positive_window_rate",
                    ]
                ]
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=float)
            ).all()
        ),
    )

    # -------------------------------------------------------------------------
    # Summary by candidate
    # -------------------------------------------------------------------------

    banner("LOCAL PERTURBATION SUMMARY")

    for candidate_id in FROZEN:
        candidate = summary[summary["candidate_id"] == candidate_id].copy()

        positive = int((candidate["total_R"] > 0).sum())

        pf_above_one = int((candidate["profit_factor"] > 1).sum())

        positive_expectancy = int((candidate["expectancy_R"] > 0).sum())

        frozen = FROZEN[candidate_id]

        frozen_row = candidate[
            np.isclose(
                candidate["tp"],
                frozen["tp"],
            )
            & np.isclose(
                candidate["sl"],
                frozen["sl"],
            )
            & (candidate["horizon"] == frozen["horizon"])
        ]

        print()
        print(candidate_id)
        print("-" * 80)

        print(f"Configurations          : {len(candidate)}")

        print(
            f"Positive total R        : "
            f"{positive}/{len(candidate)} "
            f"({100 * positive / len(candidate):.2f}%)"
        )

        print(
            f"Positive expectancy     : "
            f"{positive_expectancy}/{len(candidate)} "
            f"({100 * positive_expectancy / len(candidate):.2f}%)"
        )

        print(
            f"PF > 1                  : "
            f"{pf_above_one}/{len(candidate)} "
            f"({100 * pf_above_one / len(candidate):.2f}%)"
        )

        if len(frozen_row) == 1:
            fr = frozen_row.iloc[0]

            print()
            print("Frozen configuration")
            print(
                f"  TP / SL / H           : "
                f"{fr['tp']:.2f} / "
                f"{fr['sl']:.2f} / "
                f"{int(fr['horizon'])}"
            )
            print(f"  RR                    : {fr['rr']:.4f}")
            print(f"  Total R               : {fr['total_R']:.4f}")
            print(f"  Expectancy            : {fr['expectancy_R']:.6f}R")
            print(f"  PF                    : {fr['profit_factor']:.4f}")
            print(f"  Max DD                : {fr['max_drawdown_R']:.4f}R")
            print(
                f"  Positive windows      : "
                f"{int(fr['positive_windows'])}/"
                f"{int(fr['active_windows'])}"
            )

    # -------------------------------------------------------------------------
    # Local sensitivity
    # -------------------------------------------------------------------------

    banner("LOCAL SENSITIVITY")

    sensitivity_rows = []

    for candidate_id in FROZEN:
        candidate = summary[summary["candidate_id"] == candidate_id].copy()

        frozen = FROZEN[candidate_id]

        frozen_mask = (
            np.isclose(
                candidate["tp"],
                frozen["tp"],
            )
            & np.isclose(
                candidate["sl"],
                frozen["sl"],
            )
            & (candidate["horizon"] == frozen["horizon"])
        )

        frozen_row = candidate.loc[frozen_mask]

        if len(frozen_row) != 1:
            raise RuntimeError(
                f"Could not uniquely identify frozen configuration for {candidate_id}."
            )

        base = frozen_row.iloc[0]

        for _, row in candidate.iterrows():
            delta_tp = float(row["tp"] - frozen["tp"])
            delta_sl = float(row["sl"] - frozen["sl"])
            delta_h = int(row["horizon"] - frozen["horizon"])

            sensitivity_rows.append(
                {
                    "candidate_id": candidate_id,
                    "tp": row["tp"],
                    "sl": row["sl"],
                    "rr": row["rr"],
                    "horizon": row["horizon"],
                    "delta_tp": delta_tp,
                    "delta_sl": delta_sl,
                    "delta_horizon": delta_h,
                    "total_R": row["total_R"],
                    "expectancy_R": row["expectancy_R"],
                    "profit_factor": row["profit_factor"],
                    "max_drawdown_R": row["max_drawdown_R"],
                    "positive_windows": row["positive_windows"],
                    "active_windows": row["active_windows"],
                    "positive_window_rate": row["positive_window_rate"],
                    "delta_expectancy_vs_frozen": (
                        row["expectancy_R"] - base["expectancy_R"]
                    ),
                    "delta_total_R_vs_frozen": (row["total_R"] - base["total_R"]),
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    banner("SAVING OUTPUTS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    sensitivity.to_csv(
        OUTPUT_SENSITIVITY,
        index=False,
    )

    print(f"Summary saved      : {OUTPUT_SUMMARY}")

    print(f"Windows saved      : {OUTPUT_WINDOWS}")

    print(f"Sensitivity saved  : {OUTPUT_SENSITIVITY}")

    # -------------------------------------------------------------------------
    # Final table
    # -------------------------------------------------------------------------

    banner("FINAL CONFIGURATION TABLE")

    display_columns = [
        "candidate_id",
        "tp",
        "sl",
        "rr",
        "horizon",
        "trades",
        "win_rate",
        "total_R",
        "expectancy_R",
        "profit_factor",
        "max_drawdown_R",
        "positive_windows",
        "active_windows",
        "positive_window_rate",
    ]

    print(
        summary[display_columns]
        .sort_values(
            [
                "candidate_id",
                "expectancy_R",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Final integrity summary
    # -------------------------------------------------------------------------

    banner("08AE COMPLETE")

    print(f"Total configurations evaluated : {len(summary):,}")

    print(f"Total window results            : {len(windows):,}")

    print(f"Total trade results             : {len(trade_results):,}")

    print()
    print("Frozen reproduction: PASS")
    print("Parameter branch: NEW ONLY")
    print("Old 5/2 branch: NOT EVALUATED")
    print("Optimization: NONE")
    print()
    print("08AE finished successfully.")


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
