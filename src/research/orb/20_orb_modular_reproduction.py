from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv"
)

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

OUTPUT_FILE = RESULTS_DIR / "orb_modular_reproduction_trades.csv"


# =============================================================================
# FROZEN ORB IDENTITY
# =============================================================================

STRATEGY_NAME = "ORB"

CANDIDATE_ID = "ORB_30M_2R_11"

VERSION = "ORB_30M_2R_11_FROZEN"


# =============================================================================
# FROZEN PARAMETERS
# =============================================================================

OPENING_RANGE_MINUTES = 30

RR = 2.0

ENTRY_CUTOFF_ET = "11:00"

RTH_OPEN_ET = "09:30"

RTH_CLOSE_ET = "16:00"


# =============================================================================
# FROZEN VALIDATION WINDOW
# =============================================================================

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="America/New_York",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59",
    tz="America/New_York",
)

EXPECTED_OOS_TRADES = 1442


# =============================================================================
# EXPECTED BASELINE
# =============================================================================

EXPECTED_NET_R = 156.026568

EXPECTED_EXPECTANCY = 0.1082015

EXPECTED_WR = 0.48404993

EXPECTED_PF = 1.255941

EXPECTED_MAX_DD = -14.175719


# =============================================================================
# FROZEN PRICE PARAMETERS
# =============================================================================

# ORB has no universal fixed SL/TP distance because the stop is determined
# by the opening-range width.
#
# Therefore SL/TP should be taken from the frozen reconciliation output
# whenever those columns exist.
#
# The visualizer can reconstruct them from entry + stop_points / target_points
# if those fields are explicitly exported.


# =============================================================================
# UTILITIES
# =============================================================================


def banner(title: str) -> None:

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def first_existing(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def numeric_or_nan(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# =============================================================================
# LOAD FROZEN RECONCILIATION
# =============================================================================


def load_frozen_orb() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing frozen ORB reconciliation file:\n{INPUT_FILE}\n"
        )

    df = pd.read_csv(INPUT_FILE)

    if "entry_timestamp" not in df.columns:
        raise ValueError("ORB reconciliation is missing 'entry_timestamp'.")

    r_column = first_existing(
        df,
        [
            "r_multiple",
            "r",
            "net_R",
        ],
    )

    if r_column is None:
        raise ValueError("ORB reconciliation has no R column.")

    df = df.copy()

    # -------------------------------------------------------------------------
    # TIMESTAMPS
    # -------------------------------------------------------------------------

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    exit_column = first_existing(
        df,
        [
            "exit_timestamp",
            "exit_time",
        ],
    )

    if exit_column is not None:
        df["exit_timestamp"] = pd.to_datetime(
            df[exit_column],
            utc=True,
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # R
    # -------------------------------------------------------------------------

    df["r_multiple"] = numeric_or_nan(df[r_column])

    df = df.dropna(
        subset=[
            "entry_timestamp",
            "r_multiple",
        ]
    )

    # -------------------------------------------------------------------------
    # ENTRY TIME IN NEW YORK
    # -------------------------------------------------------------------------

    df["entry_timestamp_et"] = df["entry_timestamp"].dt.tz_convert("America/New_York")

    # -------------------------------------------------------------------------
    # EXACT OOS
    # -------------------------------------------------------------------------

    oos_mask = (df["entry_timestamp_et"] >= OOS_START) & (
        df["entry_timestamp_et"] <= OOS_END
    )

    oos = (
        df.loc[oos_mask]
        .copy()
        .sort_values(
            "entry_timestamp",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(oos) != EXPECTED_OOS_TRADES:
        raise RuntimeError(
            "\nORB OOS AUDIT FAILED\n"
            f"Expected: {EXPECTED_OOS_TRADES:,}\n"
            f"Actual:   {len(oos):,}\n"
        )

    if oos["entry_timestamp"].duplicated().any():
        raise RuntimeError("ORB OOS contains duplicate entry timestamps.")

    return oos


# =============================================================================
# NORMALIZATION
# =============================================================================


def normalize_orb(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # -------------------------------------------------------------------------
    # STRATEGY IDENTITY
    # -------------------------------------------------------------------------

    result["strategy_name"] = STRATEGY_NAME

    result["candidate_id"] = CANDIDATE_ID

    result["strategy_version"] = VERSION

    # -------------------------------------------------------------------------
    # SIDE
    # -------------------------------------------------------------------------

    side_column = first_existing(
        result,
        [
            "side",
            "direction",
        ],
    )

    if side_column is not None:
        result["side"] = result[side_column].astype(str).str.upper().str.strip()

    else:
        result["side"] = ""

    # -------------------------------------------------------------------------
    # ENTRY PRICE
    # -------------------------------------------------------------------------

    entry_price_column = first_existing(
        result,
        [
            "entry_price",
            "entry",
            "entry_px",
        ],
    )

    if entry_price_column is not None:
        result["entry_price"] = numeric_or_nan(result[entry_price_column])

    else:
        result["entry_price"] = np.nan

    # -------------------------------------------------------------------------
    # EXIT PRICE
    # -------------------------------------------------------------------------

    exit_price_column = first_existing(
        result,
        [
            "exit_price",
            "exit",
            "exit_px",
        ],
    )

    if exit_price_column is not None:
        result["exit_price"] = numeric_or_nan(result[exit_price_column])

    else:
        result["exit_price"] = np.nan

    # -------------------------------------------------------------------------
    # R
    # -------------------------------------------------------------------------

    result["r"] = result["r_multiple"]

    # -------------------------------------------------------------------------
    # BARS
    # -------------------------------------------------------------------------

    bars_column = first_existing(
        result,
        [
            "bars",
            "holding_bars",
            "bars_elapsed",
        ],
    )

    if bars_column is not None:
        result["bars"] = numeric_or_nan(result[bars_column])

    else:
        result["bars"] = np.nan

    # -------------------------------------------------------------------------
    # EXIT REASON
    # -------------------------------------------------------------------------

    reason_column = first_existing(
        result,
        [
            "exit_reason",
            "reason",
            "result",
        ],
    )

    if reason_column is not None:
        result["exit_reason"] = (
            result[reason_column].astype(str).str.lower().str.strip()
        )

    else:
        result["exit_reason"] = ""

    # -------------------------------------------------------------------------
    # OPENING RANGE WIDTH
    # -------------------------------------------------------------------------

    or_width_column = first_existing(
        result,
        [
            "or_width",
            "opening_range_width",
            "range_width",
            "orb_width",
        ],
    )

    if or_width_column is not None:
        result["entry_or_width"] = numeric_or_nan(result[or_width_column])

    else:
        result["entry_or_width"] = np.nan

    # -------------------------------------------------------------------------
    # STOP / TARGET
    # -------------------------------------------------------------------------

    for normalized, candidates in {
        "stop_points": [
            "stop_points",
            "risk_points",
            "sl_points",
        ],
        "target_points": [
            "target_points",
            "tp_points",
        ],
    }.items():
        column = first_existing(
            result,
            candidates,
        )

        if column is not None:
            result[normalized] = numeric_or_nan(result[column])

        else:
            result[normalized] = np.nan

    # -------------------------------------------------------------------------
    # ORB CONTEXT
    # -------------------------------------------------------------------------

    result["entry_hmm_state"] = np.nan

    result["entry_zscore"] = np.nan

    result["entry_quality"] = np.nan

    result["entry_vol_percentile"] = np.nan

    result["entry_vol_bucket"] = np.nan

    # ORB is not a volatility-routed MR strategy.
    # Do NOT fabricate an HMM state / z-score / quality value.
    #
    # If the original reconciliation already contains those fields,
    # preserve them.

    mapping = {
        "entry_hmm_state": [
            "entry_hmm_state",
            "hmm_state",
        ],
        "entry_zscore": [
            "entry_zscore",
            "zscore",
        ],
        "entry_quality": [
            "entry_quality",
            "quality",
        ],
        "entry_vol_percentile": [
            "entry_vol_percentile",
            "vol_percentile",
            "entry_vol",
        ],
        "entry_vol_bucket": [
            "entry_vol_bucket",
            "vol_bucket",
        ],
    }

    for normalized, candidates in mapping.items():
        column = first_existing(
            result,
            candidates,
        )

        if column is not None:
            result[normalized] = result[column]

    # -------------------------------------------------------------------------
    # DATE
    # -------------------------------------------------------------------------

    result["date_ny"] = result["entry_timestamp_et"].dt.date

    # -------------------------------------------------------------------------
    # ORB METADATA
    # -------------------------------------------------------------------------

    result["opening_range_minutes"] = OPENING_RANGE_MINUTES

    result["rr"] = RR

    result["entry_cutoff_et"] = ENTRY_CUTOFF_ET

    result["rth_open_et"] = RTH_OPEN_ET

    result["rth_close_et"] = RTH_CLOSE_ET

    # -------------------------------------------------------------------------
    # SOURCE
    # -------------------------------------------------------------------------

    result["source_file"] = INPUT_FILE.name

    # -------------------------------------------------------------------------
    # STABLE ORDER
    # -------------------------------------------------------------------------

    result = result.sort_values(
        "entry_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    result["trade_id"] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# AUDIT
# =============================================================================


def audit(
    trades: pd.DataFrame,
) -> None:

    banner("ORB MODULAR REPRODUCTION AUDIT")

    r = trades["r"]

    wins = int((r > 0).sum())

    losses = int((r < 0).sum())

    net_r = float(r.sum())

    expectancy = float(r.mean())

    gross_profit = float(r[r > 0].sum())

    gross_loss = float(-r[r < 0].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = r.cumsum()

    running_max = equity.cummax()

    dd = equity - running_max

    max_dd = float(dd.min())

    wr = wins / len(r)

    print(f"Strategy             : {STRATEGY_NAME}")

    print(f"Candidate ID         : {CANDIDATE_ID}")

    print(f"Trades               : {len(trades):,}")

    print(f"First trade          : {trades['entry_timestamp'].iloc[0]}")

    print(f"Last trade           : {trades['entry_timestamp'].iloc[-1]}")

    print(f"Wins                 : {wins:,}")

    print(f"Losses               : {losses:,}")

    print(f"Win rate             : {wr:.8%}")

    print(f"Net R                : {net_r:.6f}")

    print(f"Expectancy           : {expectancy:.8f}")

    print(f"Profit factor        : {pf:.6f}")

    print(f"Max DD               : {max_dd:.6f}R")

    # -------------------------------------------------------------------------
    # FROZEN RECONCILIATION CHECK
    # -------------------------------------------------------------------------

    checks = {
        "trade_count": len(trades) == EXPECTED_OOS_TRADES,
        "net_r": np.isclose(
            net_r,
            EXPECTED_NET_R,
            atol=1e-6,
        ),
        "expectancy": np.isclose(
            expectancy,
            EXPECTED_EXPECTANCY,
            atol=1e-6,
        ),
        "win_rate": np.isclose(
            wr,
            EXPECTED_WR,
            atol=1e-8,
        ),
        "profit_factor": np.isclose(
            pf,
            EXPECTED_PF,
            atol=1e-6,
        ),
        "max_dd": np.isclose(
            max_dd,
            EXPECTED_MAX_DD,
            atol=1e-6,
        ),
    }

    print()
    print("FROZEN RECONCILIATION:")

    for name, passed in checks.items():
        print(f"{name:20s}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError("ORB modular reconciliation FAILED.")

    print()
    print("ORB MODULAR REPRODUCTION: PASS")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("ORB MODULAR REPRODUCTION")

    print("FROZEN STRATEGY.")

    print("NO PARAMETER OPTIMIZATION.")

    print("EXACT OOS ONLY.")

    print("PRE-OOS AND POST-OOS ARE EXCLUDED.")

    print()
    print(f"Candidate: {CANDIDATE_ID}")

    print("Opening range: 30 minutes")

    print("RR: 2.0")

    print("Entry cutoff: 11:00 ET")

    df = load_frozen_orb()

    trades = normalize_orb(df)

    audit(trades)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    banner("OUTPUT")

    print(OUTPUT_FILE)

    print()
    print("Modular ORB trade file created.")


if __name__ == "__main__":
    main()
