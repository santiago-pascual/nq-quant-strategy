import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
08AA — MODULAR MEAN REVERSION REPRODUCTION

Purpose
-------
Reproduce the frozen Mean Reversion candidates through the new modular
strategy + lifecycle + backtest architecture.

Important
---------
- No costs.
- No slippage.
- Frozen parameters only.
- Full historical RTH market path.
- Intrabar OHLC lifecycle.
- Causal volatility context.
- Research 07 event context is the canonical signal context.
- Candidate event count is NOT assumed to equal completed trade count.
- No overlapping trades are allowed by the modular adapter.

Frozen candidates
-----------------

MRS2:
    SHORT
    HMM 2
    VOL 80-100
    Z >= +2.0
    TP 27.5
    SL 25.0
    H 30
    RR 1.10

MRL1:
    LONG
    HMM 1
    VOL 20-40
    Z <= -2.5
    TP 25.0
    SL 37.5
    H 8
    RR 0.6667
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

RESEARCH_07_EVENTS = CACHE_DIR / "research_07_event_metadata.csv"

RESEARCH_08B_HMM = CACHE_DIR / "research_08b_causal_hmm_states.csv"


# =============================================================================
# OUTPUTS
# =============================================================================

OUTPUT_DIR = RESULTS_DIR

OUTPUT_TRADES = OUTPUT_DIR / "research_08aa_modular_reproduction_trades.csv"

OUTPUT_SUMMARY = OUTPUT_DIR / "research_08aa_modular_reproduction_summary.csv"

OUTPUT_EVENTS = OUTPUT_DIR / "research_08aa_modular_reproduction_events.csv"

OUTPUT_AUDIT = OUTPUT_DIR / "research_08aa_modular_reproduction_audit.csv"


# =============================================================================
# EXPECTED RESEARCH CONTEXT
# =============================================================================

EXPECTED_EVENT_COUNT = 825_717

EXPECTED_RTH_ROWS = 825_746

EXPECTED_VALID_REALIZED_VOL = 2_577_631

EXPECTED_MISSING_REALIZED_VOL = 30

EXPECTED_CANDIDATE_EVENTS = {
    "MRS2": 2_253,
    "MRL1": 840,
}


# =============================================================================
# IMPORT PROJECT COMPONENTS
# =============================================================================

from src.databento_loader import load_databento_mnq

from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)

# IMPORTANT:
# Keep this import pointing to the actual project location that already
# worked in the previous run.
from src.session_engine import add_session_information

from src.strategies.mean_reversion import (
    MRL1_CONFIG,
    MRS2_CONFIG,
    MeanReversionBacktestRunner,
)


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_timestamp(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize timestamps to timezone-aware UTC.
    """

    ts = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if ts.isna().any():
        raise ValueError(
            f"Timestamp normalization produced {ts.isna().sum()} NaT values."
        )

    return ts


def assert_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# =============================================================================
# LOAD RESEARCH EVENT CONTEXT
# =============================================================================


def load_research_context() -> pd.DataFrame:
    banner("LOADING RESEARCH EVENT CONTEXT")

    if not RESEARCH_07_EVENTS.exists():
        raise FileNotFoundError(
            f"Research 07 event file not found:\n{RESEARCH_07_EVENTS}"
        )

    if not RESEARCH_08B_HMM.exists():
        raise FileNotFoundError(f"Research 08B HMM file not found:\n{RESEARCH_08B_HMM}")

    events = pd.read_csv(RESEARCH_07_EVENTS)

    hmm = pd.read_csv(RESEARCH_08B_HMM)

    print(f"Research 07 events: {len(events):,}")

    print(f"HMM rows:          {len(hmm):,}")

    if len(events) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            "Unexpected Research 07 event count: "
            f"{len(events):,} != "
            f"{EXPECTED_EVENT_COUNT:,}"
        )

    assert_columns(
        events,
        [
            "event_id",
            "data_index",
            "window",
            "timestamp",
            "close",
            "zscore_30",
        ],
        "Research 07 events",
    )

    assert_columns(
        hmm,
        [
            "event_id",
            "window",
            "timestamp",
            "hmm_state",
        ],
        "Research 08B HMM",
    )

    events["timestamp"] = normalize_timestamp(events["timestamp"])

    hmm["timestamp"] = normalize_timestamp(hmm["timestamp"])

    events = events.sort_values("event_id").reset_index(drop=True)

    hmm = hmm.sort_values("event_id").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Structural identity checks
    # -------------------------------------------------------------------------

    expected_ids = np.arange(len(events))

    if not np.array_equal(
        events["event_id"].to_numpy(),
        expected_ids,
    ):
        raise ValueError("Research 07 event_id is not a contiguous 0..N-1 index.")

    if not np.array_equal(
        hmm["event_id"].to_numpy(),
        events["event_id"].to_numpy(),
    ):
        raise ValueError("Research 08B event_id does not align with Research 07.")

    events["hmm_state"] = hmm["hmm_state"].astype(int)

    missing_hmm = events["hmm_state"].isna().sum()

    missing_zscore = events["zscore_30"].isna().sum()

    print(f"Missing HMM:   {missing_hmm:,}")

    print(f"Missing zscore: {missing_zscore:,}")

    if missing_hmm:
        raise ValueError("Missing HMM states detected.")

    if missing_zscore:
        raise ValueError("Missing zscore values detected.")

    return events


# =============================================================================
# LOAD CANONICAL MARKET DATA
# =============================================================================


def load_rth_market() -> pd.DataFrame:
    """
    Load the full canonical market dataframe.

    Feature construction happens BEFORE RTH filtering.

    This ordering is critical because realized_vol_30 is based on the full
    market path and therefore preserves the canonical volatility history.
    """

    banner("LOADING DATABENTO MARKET DATA")

    market = load_databento_mnq()

    print(f"Market rows: {len(market):,}")

    required = [
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert_columns(
        market,
        required,
        "Databento market data",
    )

    # -------------------------------------------------------------------------
    # Canonical return features
    # -------------------------------------------------------------------------

    print("Building canonical return features...")

    market = add_return_features(market)

    # -------------------------------------------------------------------------
    # Canonical volatility features
    # -------------------------------------------------------------------------

    print("Building canonical volatility features...")

    market = add_volatility_features(market)

    if "realized_vol_30" not in market.columns:
        raise ValueError(
            "Canonical volatility pipeline did not produce 'realized_vol_30'."
        )

    valid_vol = int(market["realized_vol_30"].notna().sum())

    missing_vol = int(market["realized_vol_30"].isna().sum())

    print(f"Valid realized_vol_30: {valid_vol:,}")

    print(f"Missing realized_vol_30: {missing_vol:,}")

    if valid_vol != EXPECTED_VALID_REALIZED_VOL:
        raise ValueError(
            "Unexpected realized_vol_30 count: "
            f"{valid_vol:,} != "
            f"{EXPECTED_VALID_REALIZED_VOL:,}"
        )

    if missing_vol != EXPECTED_MISSING_REALIZED_VOL:
        raise ValueError(
            "Unexpected realized_vol_30 missing count: "
            f"{missing_vol:,} != "
            f"{EXPECTED_MISSING_REALIZED_VOL:,}"
        )

    # -------------------------------------------------------------------------
    # Session information
    # -------------------------------------------------------------------------

    market = add_session_information(market)

    if "market_period" not in market.columns:
        raise ValueError("Session pipeline did not produce 'market_period'.")

    # -------------------------------------------------------------------------
    # Timestamp normalization
    # -------------------------------------------------------------------------

    market["timestamp"] = normalize_timestamp(market["timestamp ET"])

    # -------------------------------------------------------------------------
    # RTH filter AFTER feature construction
    # -------------------------------------------------------------------------

    market = (
        market.loc[market["market_period"].eq("RTH")]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"RTH rows: {len(market):,}")

    if len(market) != EXPECTED_RTH_ROWS:
        raise ValueError(
            f"Unexpected RTH row count: {len(market):,} != {EXPECTED_RTH_ROWS:,}"
        )

    return market


# =============================================================================
# CAUSAL VOLATILITY PERCENTILE
# =============================================================================


class FenwickTree:
    """
    Fenwick / Binary Indexed Tree.

    Used to calculate the causal percentile efficiently.

    Complexity:
        Coordinate compression: O(N log N)
        Each observation:       O(log N)

    This replaces the previous O(N^2 log N) implementation.
    """

    def __init__(
        self,
        size: int,
    ) -> None:
        self.size = int(size)

        self.tree = np.zeros(
            self.size + 1,
            dtype=np.int64,
        )

    def update(
        self,
        index: int,
        value: int = 1,
    ) -> None:
        i = int(index)

        while i <= self.size:
            self.tree[i] += value
            i += i & -i

    def query(
        self,
        index: int,
    ) -> int:
        result = 0
        i = int(index)

        while i > 0:
            result += int(self.tree[i])
            i -= i & -i

        return result


def build_causal_percentile(
    values: pd.Series,
) -> pd.Series:
    """
    Build a causal expanding percentile.

    At time t, only observations strictly before t are used.

    The current observation is queried BEFORE it is inserted into the
    Fenwick tree.

    Result:
        [0, 1]

    First valid observation:
        NaN
    """

    array = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    n = len(array)

    result = np.full(
        n,
        np.nan,
        dtype=float,
    )

    finite_mask = np.isfinite(array)

    if not finite_mask.any():
        return pd.Series(
            result,
            index=values.index,
            name="vol_percentile",
        )

    # -------------------------------------------------------------------------
    # Coordinate compression
    # -------------------------------------------------------------------------

    unique_values = np.unique(array[finite_mask])

    tree = FenwickTree(len(unique_values))

    previous_count = 0

    # -------------------------------------------------------------------------
    # Causal expanding percentile
    # -------------------------------------------------------------------------

    for i, value in enumerate(array):
        if not np.isfinite(value):
            continue

        # Search position of current value.
        rank_index = np.searchsorted(
            unique_values,
            value,
            side="right",
        )

        # Query ONLY prior observations.
        if previous_count > 0:
            count_leq = tree.query(rank_index)

            result[i] = count_leq / previous_count

        # Insert current observation AFTER querying.
        insert_index = (
            np.searchsorted(
                unique_values,
                value,
                side="left",
            )
            + 1
        )

        tree.update(insert_index)

        previous_count += 1

    return pd.Series(
        result,
        index=values.index,
        name="vol_percentile",
    )


# =============================================================================
# VOLATILITY BUCKETS
# =============================================================================


def assign_volatility_bucket(
    percentile: pd.Series,
) -> pd.Series:
    """
    Convert causal percentile [0,1] into canonical 5 buckets.
    """

    result = pd.Series(
        "UNKNOWN",
        index=percentile.index,
        dtype="object",
    )

    valid = percentile.notna()

    result.loc[valid & (percentile < 0.20)] = "VOL0-20"

    result.loc[valid & (percentile >= 0.20) & (percentile < 0.40)] = "VOL20-40"

    result.loc[valid & (percentile >= 0.40) & (percentile < 0.60)] = "VOL40-60"

    result.loc[valid & (percentile >= 0.60) & (percentile < 0.80)] = "VOL60-80"

    result.loc[valid & (percentile >= 0.80)] = "VOL80-100"

    return result


# =============================================================================
# BUILD EVENT-LEVEL MARKET CONTEXT
# =============================================================================


def build_event_context(
    events: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    banner("BUILDING CAUSAL VOLATILITY CONTEXT")

    market = market.copy()
    events = events.copy()

    # -------------------------------------------------------------------------
    # Causal percentile
    #
    # IMPORTANT:
    # build_causal_percentile returns 0..1.
    # -------------------------------------------------------------------------

    market["vol_percentile_raw"] = build_causal_percentile(market["realized_vol_30"])

    # -------------------------------------------------------------------------
    # Canonical bucket is based on 0..1 percentile.
    # -------------------------------------------------------------------------

    market["vol_bucket"] = assign_volatility_bucket(market["vol_percentile_raw"])

    # -------------------------------------------------------------------------
    # The modular MeanReversionConfig expects volatility percentiles in
    # 0..100 scale.
    #
    # Keep the raw value for audit and expose the strategy-facing value as
    # 0..100.
    # -------------------------------------------------------------------------

    market["vol_percentile"] = market["vol_percentile_raw"] * 100.0

    # -------------------------------------------------------------------------
    # Normalize timestamps
    # -------------------------------------------------------------------------

    events["timestamp"] = normalize_timestamp(events["timestamp"])

    market["timestamp"] = normalize_timestamp(market["timestamp"])

    # -------------------------------------------------------------------------
    # Event-to-market lookup
    # -------------------------------------------------------------------------

    market_lookup = market[
        [
            "timestamp",
            "close",
            "realized_vol_30",
            "vol_percentile_raw",
            "vol_percentile",
            "vol_bucket",
        ]
    ].copy()

    event_context = events.merge(
        market_lookup,
        on="timestamp",
        how="left",
        suffixes=(
            "_event",
            "_market",
        ),
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Timestamp mapping validation
    # -------------------------------------------------------------------------

    mapped = int(event_context["close_market"].notna().sum())

    print(f"Event timestamp == RTH timestamp: {mapped:,}/{len(events):,}")

    if mapped != len(events):
        raise ValueError("Not every Research 07 event mapped to an RTH market bar.")

    # -------------------------------------------------------------------------
    # Close validation
    # -------------------------------------------------------------------------

    close_equal = np.isclose(
        event_context["close_event"].to_numpy(dtype=float),
        event_context["close_market"].to_numpy(dtype=float),
        equal_nan=False,
    )

    print(f"Event close == market close: {close_equal.sum():,}/{len(close_equal):,}")

    if not close_equal.all():
        raise ValueError(
            "Research 07 event closes do not match canonical market closes."
        )

    # -------------------------------------------------------------------------
    # Complete event context
    # -------------------------------------------------------------------------

    complete = (
        event_context["hmm_state"].notna()
        & event_context["zscore_30"].notna()
        & event_context["vol_bucket"].ne("UNKNOWN")
    )

    print(f"\nComplete events: {complete.sum():,}")

    event_context = event_context.loc[complete].copy()

    # -------------------------------------------------------------------------
    # Canonical names used by modular strategy
    # -------------------------------------------------------------------------

    event_context["zscore"] = event_context["zscore_30"]

    # -------------------------------------------------------------------------
    # Attach canonical OHLCV
    # -------------------------------------------------------------------------

    ohlcv = market[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    # Remove the event-side close before merging.
    event_context = event_context.drop(
        columns=["close"],
        errors="ignore",
    )

    event_context = event_context.merge(
        ohlcv,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    event_context["close"] = event_context["close"].astype(float)

    event_context = event_context.sort_values("event_id").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Final context validation
    # -------------------------------------------------------------------------

    required = [
        "event_id",
        "data_index",
        "window",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hmm_state",
        "zscore",
        "realized_vol_30",
        "vol_percentile_raw",
        "vol_percentile",
        "vol_bucket",
    ]

    assert_columns(
        event_context,
        required,
        "Event context",
    )

    return event_context


# =============================================================================
# CANDIDATE MASK
# =============================================================================


def candidate_event_mask(
    events: pd.DataFrame,
    config,
) -> pd.Series:
    """
    Return Research-level signal qualification.

    This is NOT the same thing as completed trades.

    The modular adapter controls position state and therefore prevents
    overlapping positions.
    """

    mask = events["hmm_state"].eq(config.hmm_state) & events["vol_bucket"].eq(
        f"VOL{int(config.volatility_low)}-{int(config.volatility_high)}"
    )

    # -------------------------------------------------------------------------
    # config.side is a string in the actual MeanReversionConfig.
    #
    # Be robust in case an Enum is used later.
    # -------------------------------------------------------------------------

    side = config.side

    if hasattr(side, "value"):
        side = side.value

    side = str(side).upper()

    if side == "LONG":
        mask &= events["zscore"] <= -float(config.zscore_threshold)

    elif side == "SHORT":
        mask &= events["zscore"] >= float(config.zscore_threshold)

    else:
        raise ValueError(f"Unsupported candidate side: {config.side}")

    return mask


# =============================================================================
# RUN MODULAR CANDIDATE
# =============================================================================


def run_full_candidate(
    events: pd.DataFrame,
    config,
):
    """
    Run one frozen candidate through the modular architecture.

    The COMPLETE event path is passed to the runner.

    We do NOT pass only candidate events because lifecycle evaluation needs
    every subsequent OHLC bar while a trade is active.

    Therefore:

        candidate events != completed trades
    """

    # -------------------------------------------------------------------------
    # Correct runner constructor.
    # -------------------------------------------------------------------------

    runner = MeanReversionBacktestRunner(config=config)

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hmm_state",
        "vol_percentile",
        "zscore",
    ]

    assert_columns(
        events,
        required_columns,
        f"{config.name} modular input",
    )

    # -------------------------------------------------------------------------
    # Research-level candidate mask.
    # -------------------------------------------------------------------------

    candidate_mask = candidate_event_mask(
        events,
        config,
    )

    # Keep this explicit for audit/debugging.
    candidate_events = events.loc[candidate_mask].copy()

    # -------------------------------------------------------------------------
    # Run COMPLETE market path.
    # -------------------------------------------------------------------------

    trades = runner.run(events[required_columns].copy())

    trades = trades.copy()

    # -------------------------------------------------------------------------
    # Empty case.
    # -------------------------------------------------------------------------

    if trades.empty:
        trades["candidate_id"] = pd.Series(dtype=str)

        trades["strategy_name"] = pd.Series(dtype=str)

        trades["entry_event_id"] = pd.Series(dtype="int64")

        return (
            trades,
            candidate_mask,
        )

    # -------------------------------------------------------------------------
    # The modular runner now exposes entry_timestamp.
    # -------------------------------------------------------------------------

    if "entry_timestamp" not in trades.columns:
        raise RuntimeError(
            "MeanReversionBacktestRunner did not "
            "return an 'entry_timestamp' column.\n"
            f"Returned columns: "
            f"{list(trades.columns)}"
        )

    trades["entry_timestamp"] = normalize_timestamp(trades["entry_timestamp"])

    # -------------------------------------------------------------------------
    # Build event lookup.
    #
    # Every modular entry should correspond to one Research 07 event.
    # -------------------------------------------------------------------------

    lookup = (
        events[
            [
                "event_id",
                "timestamp",
            ]
        ]
        .drop_duplicates(subset=["timestamp"])
        .copy()
    )

    lookup["timestamp"] = normalize_timestamp(lookup["timestamp"])

    # -------------------------------------------------------------------------
    # Map modular entry timestamp to Research 07 event.
    # -------------------------------------------------------------------------

    trades = trades.merge(
        lookup,
        left_on="entry_timestamp",
        right_on="timestamp",
        how="left",
        validate="many_to_one",
    )

    # The right-side timestamp is now redundant.
    trades = trades.drop(columns=["timestamp"])

    trades = trades.rename(
        columns={
            "event_id": "entry_event_id",
        }
    )

    # -------------------------------------------------------------------------
    # Mapping validation.
    # -------------------------------------------------------------------------

    missing_event_ids = int(trades["entry_event_id"].isna().sum())

    if missing_event_ids:
        raise RuntimeError(
            f"{config.name}: "
            f"{missing_event_ids} modular trades "
            "could not be mapped back to "
            "Research 07 events."
        )

    trades["entry_event_id"] = trades["entry_event_id"].astype(int)

    # -------------------------------------------------------------------------
    # Candidate metadata.
    # -------------------------------------------------------------------------

    trades["candidate_id"] = config.candidate_id

    trades["strategy_name"] = config.name

    # -------------------------------------------------------------------------
    # Add original entry context.
    # -------------------------------------------------------------------------

    entry_context = events[
        [
            "event_id",
            "window",
            "hmm_state",
            "vol_bucket",
            "zscore",
        ]
    ].copy()

    entry_context = entry_context.rename(
        columns={
            "event_id": "entry_event_id",
            "hmm_state": "entry_hmm_state",
            "vol_bucket": "entry_vol_bucket",
            "zscore": "entry_zscore",
        }
    )

    trades = trades.merge(
        entry_context,
        on="entry_event_id",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------------------
    # Verify every modular entry really belongs to the frozen candidate.
    # -------------------------------------------------------------------------

    candidate_event_ids = set(candidate_events["event_id"].astype(int))

    entered_event_ids = set(trades["entry_event_id"].astype(int))

    unexpected_entries = entered_event_ids - candidate_event_ids

    if unexpected_entries:
        raise RuntimeError(
            f"{config.name}: modular strategy generated "
            f"{len(unexpected_entries)} entries that are "
            "not frozen candidate events."
        )

    return (
        trades,
        candidate_mask,
    )


# =============================================================================
# TRADE NORMALIZATION
# =============================================================================


def normalize_trade_output(
    trades: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Normalize modular runner output into a stable research schema.
    """

    if trades.empty:
        return trades.copy()

    result = trades.copy()

    result["candidate_id"] = config.candidate_id

    result["strategy_name"] = config.name

    # -------------------------------------------------------------------------
    # Lifecycle fields.
    # -------------------------------------------------------------------------

    if "r_multiple" in result.columns:
        result["r"] = pd.to_numeric(
            result["r_multiple"],
            errors="coerce",
        )

    if "bars_elapsed" in result.columns:
        result["bars"] = pd.to_numeric(
            result["bars_elapsed"],
            errors="coerce",
        )

    if "exit_reason" in result.columns:
        result["exit_reason"] = result["exit_reason"].astype(str).str.lower()

    # -------------------------------------------------------------------------
    # Side.
    # -------------------------------------------------------------------------

    if "side" in result.columns:
        result["side"] = result["side"].astype(str).str.upper()

    # -------------------------------------------------------------------------
    # Numeric fields.
    # -------------------------------------------------------------------------

    for column in [
        "entry_price",
        "exit_price",
        "r",
        "bars",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


# =============================================================================
# METRICS
# =============================================================================


def calculate_metrics(
    trades: pd.DataFrame,
    config,
) -> dict:
    """
    Calculate basic trade-level performance metrics.
    """

    if trades.empty:
        return {
            "candidate_id": config.candidate_id,
            "strategy": config.name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": np.nan,
            "net_r": 0.0,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": 0.0,
            "mean_bars": np.nan,
        }

    r = pd.to_numeric(
        trades["r"],
        errors="coerce",
    ).dropna()

    if r.empty:
        return {
            "candidate_id": config.candidate_id,
            "strategy": config.name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": np.nan,
            "net_r": 0.0,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": 0.0,
            "mean_bars": np.nan,
        }

    wins = int((r > 0).sum())

    losses = int((r < 0).sum())

    timeouts = 0

    if "exit_reason" in trades.columns:
        timeouts = int(trades["exit_reason"].eq("timeout").sum())

    gross_profit = float(r[r > 0].sum())

    gross_loss = float(-r[r < 0].sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    else:
        profit_factor = np.inf

    # -------------------------------------------------------------------------
    # Equity and drawdown in R.
    # -------------------------------------------------------------------------

    equity = r.cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = float(drawdown.min())

    return {
        "candidate_id": config.candidate_id,
        "strategy": config.name,
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": (wins / len(r) if len(r) else np.nan),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": (max_drawdown),
        "mean_bars": (
            float(trades["bars"].mean()) if "bars" in trades.columns else np.nan
        ),
    }


# =============================================================================
# EVENT AUDIT
# =============================================================================


def build_event_audit(
    events: pd.DataFrame,
    candidate_masks: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Store candidate-level signal-event counts.

    These are signal observations, not completed trades.
    """

    rows = []

    for candidate_name, mask in candidate_masks.items():
        rows.append(
            {
                "candidate": candidate_name,
                "candidate_events": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("08AA — MODULAR MEAN REVERSION REPRODUCTION")

    print("No costs.")
    print("No slippage.")
    print("Frozen parameters only.")
    print("Full historical market path.")
    print("Intrabar OHLC lifecycle.")

    # =========================================================================
    # LOAD RESEARCH CONTEXT
    # =========================================================================

    events = load_research_context()

    # =========================================================================
    # LOAD MARKET
    # =========================================================================

    market = load_rth_market()

    # =========================================================================
    # BUILD EVENT CONTEXT
    # =========================================================================

    events = build_event_context(
        events,
        market,
    )

    # =========================================================================
    # VOLATILITY DISTRIBUTION
    # =========================================================================

    banner("VOLATILITY BUCKET DISTRIBUTION")

    print(events["vol_bucket"].value_counts().sort_index().to_string())

    # =========================================================================
    # FROZEN CANDIDATES
    # =========================================================================

    configs = [
        MRS2_CONFIG,
        MRL1_CONFIG,
    ]

    candidate_masks: dict[
        str,
        pd.Series,
    ] = {}

    banner("FROZEN CANDIDATE EVENT COUNTS")

    for config in configs:
        mask = candidate_event_mask(
            events,
            config,
        )

        candidate_masks[config.name] = mask

        actual = int(mask.sum())

        expected = EXPECTED_CANDIDATE_EVENTS[config.name]

        print(
            f"{config.name}: "
            f"actual={actual:,} | "
            f"expected={expected:,} | "
            f"match={actual == expected}"
        )

        if actual != expected:
            raise ValueError(
                f"{config.name} candidate "
                "event count mismatch: "
                f"{actual:,} != "
                f"{expected:,}"
            )

    # =========================================================================
    # RUN CANDIDATES
    # =========================================================================

    all_trades = []

    summaries = []

    for config in configs:
        banner(f"RUNNING {config.name}")

        trades, mask = run_full_candidate(
            events,
            config,
        )

        trades = normalize_trade_output(
            trades,
            config,
        )

        metrics = calculate_metrics(
            trades,
            config,
        )

        summaries.append(metrics)

        print(f"Candidate events: {int(mask.sum()):,}")

        print(f"Completed trades: {metrics['trades']:,}")

        print(f"Wins:             {metrics['wins']:,}")

        print(f"Losses:           {metrics['losses']:,}")

        print(f"Timeouts:         {metrics['timeouts']:,}")

        if np.isfinite(metrics["win_rate"]):
            print(f"Win rate:         {metrics['win_rate']:.4f}")

        else:
            print("Win rate:         NaN")

        print(f"Net R:            {metrics['net_r']:.4f}")

        if np.isfinite(metrics["expectancy_r"]):
            print(f"Expectancy R:     {metrics['expectancy_r']:.6f}")

        else:
            print("Expectancy R:     NaN")

        if np.isfinite(metrics["profit_factor"]):
            print(f"Profit factor:    {metrics['profit_factor']:.6f}")

        else:
            print("Profit factor:    inf")

        print(f"Max DD R:         {metrics['max_drawdown_r']:.4f}")

        if np.isfinite(metrics["mean_bars"]):
            print(f"Mean bars:        {metrics['mean_bars']:.4f}")

        else:
            print("Mean bars:        NaN")

        if not trades.empty:
            all_trades.append(trades)

    # =========================================================================
    # COMBINE TRADES
    # =========================================================================

    if all_trades:
        trades_output = pd.concat(
            all_trades,
            ignore_index=True,
        )

    else:
        trades_output = pd.DataFrame()

    summary_output = pd.DataFrame(summaries)

    event_output = build_event_audit(
        events,
        candidate_masks,
    )

    # =========================================================================
    # MODULAR REPRODUCTION AUDIT
    # =========================================================================

    banner("MODULAR REPRODUCTION AUDIT")

    audit_rows = []

    for config in configs:
        candidate_name = config.name

        candidate_mask = candidate_masks[candidate_name]

        candidate_event_ids = set(
            events.loc[
                candidate_mask,
                "event_id",
            ].astype(int)
        )

        if trades_output.empty:
            candidate_trades = pd.DataFrame()

        else:
            candidate_trades = trades_output.loc[
                trades_output["strategy_name"] == candidate_name
            ]

        if candidate_trades.empty:
            entered_events = set()

        else:
            entered_events = set(candidate_trades["entry_event_id"].astype(int))

        unexpected_entries = entered_events - candidate_event_ids

        all_entries_are_valid = len(unexpected_entries) == 0

        audit_rows.append(
            {
                "candidate": candidate_name,
                "candidate_events": len(candidate_event_ids),
                "modular_completed_trades": len(candidate_trades),
                "modular_entries": len(entered_events),
                "unexpected_entries": len(unexpected_entries),
                "all_entries_from_candidate_events": all_entries_are_valid,
            }
        )

        print(
            f"{candidate_name}: "
            f"candidate_events="
            f"{len(candidate_event_ids):,} | "
            f"entries="
            f"{len(entered_events):,} | "
            f"completed_trades="
            f"{len(candidate_trades):,} | "
            f"unexpected_entries="
            f"{len(unexpected_entries):,} | "
            f"PASS="
            f"{all_entries_are_valid}"
        )

        if not all_entries_are_valid:
            raise RuntimeError(
                f"{candidate_name}: modular "
                "strategy generated entries "
                "that were not candidate "
                "signal events."
            )

    audit_output = pd.DataFrame(audit_rows)

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_output.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary_output.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    event_output.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    audit_output.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    banner("08AA — FINAL SUMMARY")

    print(summary_output.to_string(index=False))

    print()

    print(f"Trades saved:  {OUTPUT_TRADES}")

    print(f"Summary saved: {OUTPUT_SUMMARY}")

    print(f"Events saved:  {OUTPUT_EVENTS}")

    print(f"Audit saved:   {OUTPUT_AUDIT}")

    print()

    print("08AA MODULAR REPRODUCTION COMPLETE")


if __name__ == "__main__":
    main()
