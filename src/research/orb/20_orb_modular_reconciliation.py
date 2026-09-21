from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from src.databento_loader import load_databento_mnq
from src.strategies.orb import run_frozen_orb


# =============================================================================
# PATHS
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "orb"

BASELINE_PATH = RESULTS_DIR / "orb_reconciliation_trades.csv"

OUTPUT_DIR = RESULTS_DIR / "modular_reconciliation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPARISON_PATH = OUTPUT_DIR / "orb_modular_reconciliation_comparison.csv"

MISMATCH_PATH = OUTPUT_DIR / "orb_modular_reconciliation_mismatches.csv"


# =============================================================================
# CONFIG
# =============================================================================

ET_TZ = "America/New_York"

OOS_START = "2020-06-23"
OOS_END = "2026-06-19"

RTH_START_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60


# =============================================================================
# HELPERS
# =============================================================================


def header(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def parse_timestamp(
    series: pd.Series,
) -> pd.Series:

    return pd.to_datetime(
        series,
        utc=True,
        errors="raise",
    )


def normalize_side(
    value,
) -> str:

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    if value in {
        "LONG",
        "BUY",
    }:
        return "LONG"

    if value in {
        "SHORT",
        "SELL",
    }:
        return "SHORT"

    return value


def normalize_exit_reason(
    value,
) -> str:

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    if value in {
        "INITIAL_STOP",
        "STOP",
        "STOP_AND_TARGET_SAME_BAR_STOP_FIRST",
        "STOP_AND_TARGET_SAME_BAR_TARGET_FIRST",
    }:
        return "stop"

    if value in {
        "TAKE_PROFIT",
        "TARGET",
    }:
        return "target"

    if value == "RTH_CLOSE":
        return "rth_close"

    return value.lower()


def numeric_equal(
    a,
    b,
    atol: float = 1e-9,
) -> bool:

    if pd.isna(a) and pd.isna(b):
        return True

    if pd.isna(a) or pd.isna(b):
        return False

    return bool(
        np.isclose(
            float(a),
            float(b),
            atol=atol,
            rtol=0.0,
        )
    )


# =============================================================================
# MARKET DATA
# =============================================================================


def load_oos_market() -> pd.DataFrame:

    header("LOADING CANONICAL DATABENTO DATA")

    data = load_databento_mnq()

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError("load_databento_mnq() did not return a DataFrame.")

    data = data.copy()

    if "timestamp ET" in data.columns:
        data = data.rename(columns={"timestamp ET": "timestamp"})

    if "timestamp" not in data.columns:
        raise KeyError("Canonical data does not contain 'timestamp'.")

    data["timestamp"] = parse_timestamp(data["timestamp"])

    data["timestamp_et"] = data["timestamp"].dt.tz_convert(ET_TZ)

    print(f"Rows:  {len(data):,}")

    print(f"Start: {data['timestamp_et'].min()}")

    print(f"End:   {data['timestamp_et'].max()}")

    # -------------------------------------------------------------------------
    # EXACT OOS
    # -------------------------------------------------------------------------

    oos_start = pd.Timestamp(
        OOS_START,
        tz=ET_TZ,
    )

    oos_end = pd.Timestamp(
        OOS_END,
        tz=ET_TZ,
    )

    session_dates = data["timestamp_et"].dt.normalize()

    data = data.loc[(session_dates >= oos_start) & (session_dates <= oos_end)].copy()

    # -------------------------------------------------------------------------
    # RTH
    # -------------------------------------------------------------------------

    minutes = data["timestamp_et"].dt.hour * 60 + data["timestamp_et"].dt.minute

    data = data.loc[(minutes >= RTH_START_MINUTE) & (minutes < RTH_END_MINUTE)].copy()

    data = data.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"OOS market rows: {len(data):,}")

    return data


# =============================================================================
# MODULAR ORB
# =============================================================================


def run_modular(
    market: pd.DataFrame,
) -> pd.DataFrame:

    header("RUNNING MODULAR ORB")

    modular_input = market.copy()

    # IMPORTANT:
    #
    # ORBContextBuilder reads timestamp.hour/minute directly.
    # Therefore the modular strategy must receive NY-local timestamps.

    modular_input["timestamp"] = modular_input["timestamp"].dt.tz_convert(ET_TZ)

    trades = run_frozen_orb(modular_input)

    trades = trades.copy()

    print(f"Modular trades: {len(trades):,}")

    return trades


# =============================================================================
# BASELINE
# =============================================================================


def load_baseline(
    market: pd.DataFrame,
) -> pd.DataFrame:

    header("LOADING FROZEN ORB BASELINE")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found:\n{BASELINE_PATH}")

    baseline = pd.read_csv(BASELINE_PATH)

    print(f"Baseline raw rows: {len(baseline):,}")

    print()
    print("Baseline columns:")

    print(list(baseline.columns))

    required = {
        "direction",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "stop_price",
        "target_price",
        "net_R",
        "exit_reason",
    }

    missing = required - set(baseline.columns)

    if missing:
        raise KeyError(f"Baseline is missing required columns: {sorted(missing)}")

    # -------------------------------------------------------------------------
    # TIMESTAMPS
    # -------------------------------------------------------------------------

    baseline["entry_timestamp"] = parse_timestamp(baseline["entry_timestamp"])

    baseline["exit_timestamp"] = parse_timestamp(baseline["exit_timestamp"])

    # -------------------------------------------------------------------------
    # EXACT OOS
    # -------------------------------------------------------------------------

    entry_et = baseline["entry_timestamp"].dt.tz_convert(ET_TZ)

    entry_dates = entry_et.dt.normalize()

    oos_start = pd.Timestamp(
        OOS_START,
        tz=ET_TZ,
    )

    oos_end = pd.Timestamp(
        OOS_END,
        tz=ET_TZ,
    )

    baseline = baseline.loc[
        (entry_dates >= oos_start) & (entry_dates <= oos_end)
    ].copy()

    # -------------------------------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------------------------------

    baseline["direction"] = baseline["direction"].map(normalize_side)

    for column in [
        "entry_price",
        "stop_price",
        "target_price",
        "net_R",
    ]:
        baseline[column] = pd.to_numeric(
            baseline[column],
            errors="raise",
        )

    baseline["exit_reason"] = baseline["exit_reason"].map(normalize_exit_reason)

    # -------------------------------------------------------------------------
    # RECONSTRUCT EXIT PRICE
    # -------------------------------------------------------------------------

    market_close = market[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    market_close["timestamp"] = parse_timestamp(market_close["timestamp"])

    market_close = market_close.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    ).set_index("timestamp")["close"]

    baseline["exit_price"] = np.nan

    stop_mask = baseline["exit_reason"] == "stop"

    target_mask = baseline["exit_reason"] == "target"

    close_mask = baseline["exit_reason"] == "rth_close"

    baseline.loc[
        stop_mask,
        "exit_price",
    ] = baseline.loc[
        stop_mask,
        "stop_price",
    ]

    baseline.loc[
        target_mask,
        "exit_price",
    ] = baseline.loc[
        target_mask,
        "target_price",
    ]

    # Market timestamps are UTC.
    # Baseline timestamps are UTC after parse_timestamp().
    baseline.loc[
        close_mask,
        "exit_price",
    ] = baseline.loc[
        close_mask,
        "exit_timestamp",
    ].map(market_close)

    missing_exit_price = baseline["exit_price"].isna()

    if missing_exit_price.any():
        print()
        print(
            "WARNING: missing reconstructed exit prices:",
            int(missing_exit_price.sum()),
        )

        print(
            baseline.loc[
                missing_exit_price,
                [
                    "entry_timestamp",
                    "exit_timestamp",
                    "exit_reason",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Cannot reconcile baseline: some exit prices could not be reconstructed."
        )

    baseline = baseline.sort_values(
        "entry_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"Baseline OOS trades: {len(baseline):,}")

    print(
        "Baseline reconstructed "
        f"exit prices: "
        f"{baseline['exit_price'].notna().sum()} "
        f"/ {len(baseline)}"
    )

    return baseline


# =============================================================================
# NORMALIZE MODULAR
# =============================================================================


def normalize_modular(
    modular: pd.DataFrame,
) -> pd.DataFrame:

    df = modular.copy()

    df["entry_timestamp"] = parse_timestamp(df["entry_timestamp"])

    df["exit_timestamp"] = parse_timestamp(df["exit_timestamp"])

    df["direction"] = df["side"].map(normalize_side)

    for column in [
        "entry_price",
        "exit_price",
        "stop_price",
        "target_price",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df["net_R"] = pd.to_numeric(
        df["r_multiple"],
        errors="raise",
    )

    df["exit_reason"] = df["exit_reason"].map(normalize_exit_reason)

    return df.sort_values(
        "entry_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# COMPARISON
# =============================================================================


def compare_trades(
    modular: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:

    total = max(
        len(modular),
        len(baseline),
    )

    rows = []

    for i in range(total):
        m = modular.iloc[i] if i < len(modular) else None

        b = baseline.iloc[i] if i < len(baseline) else None

        if m is None or b is None:
            rows.append(
                {
                    "index": i,
                    "trade_match": False,
                    "entry_timestamp_match": False,
                    "exit_timestamp_match": False,
                    "direction_match": False,
                    "entry_price_match": False,
                    "stop_price_match": False,
                    "target_price_match": False,
                    "exit_price_match": False,
                    "r_match": False,
                    "exit_reason_match": False,
                    "entry_timestamp_modular": m["entry_timestamp"]
                    if m is not None
                    else pd.NaT,
                    "entry_timestamp_baseline": b["entry_timestamp"]
                    if b is not None
                    else pd.NaT,
                    "exit_timestamp_modular": m["exit_timestamp"]
                    if m is not None
                    else pd.NaT,
                    "exit_timestamp_baseline": b["exit_timestamp"]
                    if b is not None
                    else pd.NaT,
                    "direction_modular": m["direction"] if m is not None else "",
                    "direction_baseline": b["direction"] if b is not None else "",
                    "entry_price_modular": m["entry_price"]
                    if m is not None
                    else np.nan,
                    "entry_price_baseline": b["entry_price"]
                    if b is not None
                    else np.nan,
                    "stop_price_modular": m["stop_price"] if m is not None else np.nan,
                    "stop_price_baseline": b["stop_price"] if b is not None else np.nan,
                    "target_price_modular": m["target_price"]
                    if m is not None
                    else np.nan,
                    "target_price_baseline": b["target_price"]
                    if b is not None
                    else np.nan,
                    "exit_price_modular": m["exit_price"] if m is not None else np.nan,
                    "exit_price_baseline": b["exit_price"] if b is not None else np.nan,
                    "net_R_modular": m["net_R"] if m is not None else np.nan,
                    "net_R_baseline": b["net_R"] if b is not None else np.nan,
                    "exit_reason_modular": m["exit_reason"] if m is not None else "",
                    "exit_reason_baseline": b["exit_reason"] if b is not None else "",
                }
            )

            continue

        entry_timestamp_match = m["entry_timestamp"] == b["entry_timestamp"]

        exit_timestamp_match = m["exit_timestamp"] == b["exit_timestamp"]

        direction_match = m["direction"] == b["direction"]

        entry_price_match = numeric_equal(
            m["entry_price"],
            b["entry_price"],
        )

        stop_price_match = numeric_equal(
            m["stop_price"],
            b["stop_price"],
        )

        target_price_match = numeric_equal(
            m["target_price"],
            b["target_price"],
        )

        exit_price_match = numeric_equal(
            m["exit_price"],
            b["exit_price"],
        )

        r_match = numeric_equal(
            m["net_R"],
            b["net_R"],
        )

        exit_reason_match = m["exit_reason"] == b["exit_reason"]

        trade_match = all(
            [
                entry_timestamp_match,
                exit_timestamp_match,
                direction_match,
                entry_price_match,
                stop_price_match,
                target_price_match,
                exit_price_match,
                r_match,
                exit_reason_match,
            ]
        )

        rows.append(
            {
                "index": i,
                "trade_match": trade_match,
                "entry_timestamp_match": entry_timestamp_match,
                "exit_timestamp_match": exit_timestamp_match,
                "direction_match": direction_match,
                "entry_price_match": entry_price_match,
                "stop_price_match": stop_price_match,
                "target_price_match": target_price_match,
                "exit_price_match": exit_price_match,
                "r_match": r_match,
                "exit_reason_match": exit_reason_match,
                "entry_timestamp_modular": m["entry_timestamp"],
                "entry_timestamp_baseline": b["entry_timestamp"],
                "exit_timestamp_modular": m["exit_timestamp"],
                "exit_timestamp_baseline": b["exit_timestamp"],
                "direction_modular": m["direction"],
                "direction_baseline": b["direction"],
                "entry_price_modular": m["entry_price"],
                "entry_price_baseline": b["entry_price"],
                "stop_price_modular": m["stop_price"],
                "stop_price_baseline": b["stop_price"],
                "target_price_modular": m["target_price"],
                "target_price_baseline": b["target_price"],
                "exit_price_modular": m["exit_price"],
                "exit_price_baseline": b["exit_price"],
                "net_R_modular": m["net_R"],
                "net_R_baseline": b["net_R"],
                "exit_reason_modular": m["exit_reason"],
                "exit_reason_baseline": b["exit_reason"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# DIAGNOSTICS
# =============================================================================


def print_diagnostics(
    comparison: pd.DataFrame,
    modular: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:

    header("RECONCILIATION RESULTS")

    print(f"Modular trades:  {len(modular):,}")

    print(f"Baseline trades: {len(baseline):,}")

    print(f"Difference: {len(modular) - len(baseline):+,}")

    print()

    print(
        "Trade count:",
        "PASS" if len(modular) == len(baseline) else "FAIL",
    )

    checks = [
        "entry_timestamp_match",
        "exit_timestamp_match",
        "direction_match",
        "entry_price_match",
        "stop_price_match",
        "target_price_match",
        "exit_price_match",
        "r_match",
        "exit_reason_match",
    ]

    print()
    print("FIELD MATCH RATES")

    print("-" * 80)

    for column in checks:
        matched = int(comparison[column].sum())

        total = len(comparison)

        rate = matched / total if total else 0.0

        print(f"{column:<32}{matched:>6}/{total:<6} ({rate:.4%})")

    mismatches = comparison.loc[~comparison["trade_match"]].copy()

    print()
    print(f"Trade mismatches: {len(mismatches):,}")

    if mismatches.empty:
        print()
        print("#" * 80)

        print("# ORB MODULAR RECONCILIATION: PASS")

        print("#" * 80)

        return

    print()
    print("FIRST MISMATCHES")

    print("-" * 80)

    columns = [
        "index",
        "entry_timestamp_modular",
        "entry_timestamp_baseline",
        "exit_timestamp_modular",
        "exit_timestamp_baseline",
        "direction_modular",
        "direction_baseline",
        "entry_price_modular",
        "entry_price_baseline",
        "stop_price_modular",
        "stop_price_baseline",
        "target_price_modular",
        "target_price_baseline",
        "exit_price_modular",
        "exit_price_baseline",
        "net_R_modular",
        "net_R_baseline",
        "exit_reason_modular",
        "exit_reason_baseline",
    ]

    print(mismatches[columns].head(20).to_string(index=False))

    print()
    print("MISMATCH BREAKDOWN")

    print("-" * 80)

    for column in checks:
        count = int((~comparison[column]).sum())

        if count:
            print(f"{column:<32}: {count:,}")

    # -------------------------------------------------------------------------
    # FIRST ENTRY DIVERGENCE
    # -------------------------------------------------------------------------

    divergent = comparison.loc[~comparison["entry_timestamp_match"]]

    print()
    print("FIRST ENTRY TIMESTAMP DIVERGENCE")

    print("-" * 80)

    if divergent.empty:
        print("No entry timestamp divergence.")

    else:
        row = divergent.iloc[0]

        print(f"Index: {int(row['index'])}")

        print(f"Modular: {row['entry_timestamp_modular']}")

        print(f"Baseline: {row['entry_timestamp_baseline']}")

    mismatches.to_csv(
        MISMATCH_PATH,
        index=False,
    )

    print()
    print("Mismatch file:")

    print(MISMATCH_PATH)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    print()
    print("#" * 80)

    print("# ORB MODULAR RECONCILIATION")

    print("#" * 80)

    print()
    print(f"OOS: {OOS_START} -> {OOS_END}")

    # -------------------------------------------------------------------------
    # 1. MARKET
    # -------------------------------------------------------------------------

    market = load_oos_market()

    # -------------------------------------------------------------------------
    # 2. MODULAR
    # -------------------------------------------------------------------------

    modular_raw = run_modular(market)

    # -------------------------------------------------------------------------
    # 3. BASELINE
    # -------------------------------------------------------------------------

    baseline_raw = load_baseline(market)

    # -------------------------------------------------------------------------
    # 4. NORMALIZE
    # -------------------------------------------------------------------------

    modular = normalize_modular(modular_raw)

    baseline = baseline_raw.copy()

    # -------------------------------------------------------------------------
    # 5. COMPARE
    # -------------------------------------------------------------------------

    comparison = compare_trades(
        modular,
        baseline,
    )

    # -------------------------------------------------------------------------
    # 6. SAVE
    # -------------------------------------------------------------------------

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 7. DIAGNOSTICS
    # -------------------------------------------------------------------------

    print_diagnostics(
        comparison,
        modular,
        baseline,
    )

    print()
    print("=" * 80)

    print("OUTPUT")

    print("=" * 80)

    print(f"Full comparison:\n{COMPARISON_PATH}")

    if MISMATCH_PATH.exists():
        print(f"Mismatch file:\n{MISMATCH_PATH}")


if __name__ == "__main__":
    main()
