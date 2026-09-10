from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

EVENTS_FILE = CACHE_DIR / "research_07_event_metadata.csv"

OUTPUT_FILE = RESULTS_DIR / "research_08ac_mae_mfe_failure_analysis.csv"

SUMMARY_FILE = RESULTS_DIR / "research_08ac_failure_summary.csv"


# ============================================================
# NEW PARAMETER BRANCH
# ============================================================

CANDIDATES = {
    "MRS2_NEW": {
        "side": "SHORT",
        "tp": 27.5,
        "sl": 25.0,
        "horizon": 30,
    },
    "MRL1_NEW": {
        "side": "LONG",
        "tp": 25.0,
        "sl": 37.5,
        "horizon": 8,
    },
}


# ============================================================
# HELPERS
# ============================================================


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan

    return float(series.mean())


def safe_median(series: pd.Series) -> float:
    if series.empty:
        return np.nan

    return float(series.median())


def percentile(
    series: pd.Series,
    q: float,
) -> float:

    if series.empty:
        return np.nan

    return float(series.quantile(q))


# ============================================================
# LOAD RESEARCH 07
# ============================================================


def load_research_07():

    path = BASE_DIR / "research" / "07_path_dependent_long_short.py"

    spec = importlib.util.spec_from_file_location(
        "research_07",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Research 07.")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# FUTURE PATH
# ============================================================


def build_future_path(
    rth: pd.DataFrame,
    max_horizon: int,
):

    high = rth["high"].to_numpy(dtype=float)

    low = rth["low"].to_numpy(dtype=float)

    close = rth["close"].to_numpy(dtype=float)

    n = len(rth)

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

    for offset in range(
        1,
        max_horizon + 1,
    ):
        if offset >= n:
            break

        future_high[
            :-offset,
            offset - 1,
        ] = high[offset:]

        future_low[
            :-offset,
            offset - 1,
        ] = low[offset:]

        future_close[
            :-offset,
            offset - 1,
        ] = close[offset:]

    return (
        future_high,
        future_low,
        future_close,
    )


# ============================================================
# REAL EXIT + MAE/MFE
# ============================================================


def analyze_trade_path(
    *,
    entry: float,
    side: str,
    tp: float,
    sl: float,
    horizon: int,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    data_index: int,
):
    """
    Calculate MAE/MFE only until the actual trade exit.

    Exit rules:

        TP
        SL
        SAME BAR -> STOP FIRST
        TIMEOUT -> final horizon close

    R definition:

        1R = SL distance
    """

    fh = future_high[
        data_index,
        :horizon,
    ]

    fl = future_low[
        data_index,
        :horizon,
    ]

    fc = future_close[
        data_index,
        :horizon,
    ]

    valid = np.isfinite(fh) & np.isfinite(fl) & np.isfinite(fc)

    valid_positions = np.flatnonzero(valid)

    if len(valid_positions) == 0:
        return {
            "exit_type": "UNRESOLVED",
            "exit_bar": 0,
            "exit_price": np.nan,
            "mae_points": np.nan,
            "mfe_points": np.nan,
            "mae_r": np.nan,
            "mfe_r": np.nan,
            "mae_bar": np.nan,
            "mfe_bar": np.nan,
        }

    # --------------------------------------------------------
    # FIND REAL EXIT
    # --------------------------------------------------------

    exit_bar = None
    exit_type = None
    exit_price = np.nan

    for i in valid_positions:
        high_i = fh[i]
        low_i = fl[i]

        if side == "LONG":
            tp_hit = high_i >= entry + tp

            sl_hit = low_i <= entry - sl

        else:
            tp_hit = low_i <= entry - tp

            sl_hit = high_i >= entry + sl

        # ----------------------------------------------------
        # SAME BAR -> STOP FIRST
        # ----------------------------------------------------

        if tp_hit and sl_hit:
            exit_bar = i + 1

            exit_type = "SAME_BAR_SL_FIRST"

            if side == "LONG":
                exit_price = entry - sl
            else:
                exit_price = entry + sl

            break

        # ----------------------------------------------------
        # SL
        # ----------------------------------------------------

        if sl_hit:
            exit_bar = i + 1

            exit_type = "SL"

            if side == "LONG":
                exit_price = entry - sl
            else:
                exit_price = entry + sl

            break

        # ----------------------------------------------------
        # TP
        # ----------------------------------------------------

        if tp_hit:
            exit_bar = i + 1

            exit_type = "TP"

            if side == "LONG":
                exit_price = entry + tp
            else:
                exit_price = entry - tp

            break

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if exit_bar is None:
        last_position = valid_positions[-1]

        exit_bar = last_position + 1

        exit_type = "TIMEOUT"

        exit_price = fc[last_position]

    # --------------------------------------------------------
    # PATH ONLY UNTIL EXIT
    # --------------------------------------------------------

    end = exit_bar

    path_high = fh[:end]
    path_low = fl[:end]

    if side == "LONG":
        favorable = path_high - entry

        adverse = entry - path_low

    else:
        favorable = entry - path_low

        adverse = path_high - entry

    mfe_points = max(
        0.0,
        float(np.max(favorable)),
    )

    mae_points = max(
        0.0,
        float(np.max(adverse)),
    )

    # --------------------------------------------------------
    # R NORMALIZATION
    # --------------------------------------------------------

    mae_r = mae_points / sl
    mfe_r = mfe_points / sl

    mfe_bar = int(np.argmax(favorable) + 1)

    mae_bar = int(np.argmax(adverse) + 1)

    return {
        "exit_type": exit_type,
        "exit_bar": exit_bar,
        "exit_price": exit_price,
        "mae_points": mae_points,
        "mfe_points": mfe_points,
        "mae_r": mae_r,
        "mfe_r": mfe_r,
        "mae_bar": mae_bar,
        "mfe_bar": mfe_bar,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AC — REAL-EXIT MAE / MFE + FAILURE ANALYSIS")
    print("=" * 72)

    # ========================================================
    # LOAD TRADES
    # ========================================================

    trades = pd.read_csv(TRADES_FILE)

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
    )

    print(f"\nTrades loaded: {len(trades):,}")

    # ========================================================
    # LOAD RESEARCH 07
    # ========================================================

    research_07 = load_research_07()

    from src.data_loader import load_data

    print("\nLoading canonical market data...")

    market = load_data()

    print(f"Market rows: {len(market):,}")

    print("Reproducing Research 07 RTH...")

    rth = research_07.prepare_rth(market.copy())

    print(f"RTH rows: {len(rth):,}")

    if len(rth) != 825_746:
        raise RuntimeError("RTH dataset does not match Research 07.")

    rth = rth.reset_index(drop=True)

    # ========================================================
    # EVENT METADATA
    # ========================================================

    events = pd.read_csv(EVENTS_FILE)

    events = events.sort_values("event_id").reset_index(drop=True)

    # ========================================================
    # EVENT ID INTEGRITY
    # ========================================================

    expected_event_ids = np.arange(len(events))

    if not np.array_equal(
        events["event_id"].to_numpy(),
        expected_event_ids,
    ):
        raise RuntimeError(
            "Research 07 event_id is not equal to metadata row position."
        )

    # ========================================================
    # FUTURE PATH
    # ========================================================

    max_horizon = max(cfg["horizon"] for cfg in CANDIDATES.values())

    print(f"\nBuilding future path up to {max_horizon} bars...")

    (
        future_high,
        future_low,
        future_close,
    ) = build_future_path(
        rth,
        max_horizon,
    )

    # ========================================================
    # MAE / MFE
    # ========================================================

    print("\nCalculating REAL-EXIT MAE / MFE...")

    output_rows = []

    for _, trade in trades.iterrows():
        candidate = trade["candidate_id"]

        if candidate not in CANDIDATES:
            continue

        cfg = CANDIDATES[candidate]

        event_id = int(trade["event_id"])

        event = events.iloc[event_id]

        data_index = int(event["data_index"])

        # ----------------------------------------------------
        # IMPORTANT:
        # 08AA does NOT contain an "entry" column.
        #
        # Research 07 defines the event entry as the event
        # close stored in event metadata.
        # ----------------------------------------------------

        entry = float(event["close"])

        result = analyze_trade_path(
            entry=entry,
            side=cfg["side"],
            tp=cfg["tp"],
            sl=cfg["sl"],
            horizon=cfg["horizon"],
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            data_index=data_index,
        )

        output_rows.append(
            {
                "candidate_id": candidate,
                "event_id": event_id,
                "timestamp": trade["timestamp"],
                "window": trade["window"],
                "side": cfg["side"],
                "tp": cfg["tp"],
                "sl": cfg["sl"],
                "rr": cfg["tp"] / cfg["sl"],
                "horizon": cfg["horizon"],
                "entry": entry,
                # Original 08AA result
                "result": trade["result"],
                "r": trade["r"],
                "bars": trade["bars"],
                # Reconstructed real exit
                "exit_type": result["exit_type"],
                "exit_bar": result["exit_bar"],
                "exit_price": result["exit_price"],
                # Real-exit MAE/MFE
                "mae_points": result["mae_points"],
                "mae_r": result["mae_r"],
                "mfe_points": result["mfe_points"],
                "mfe_r": result["mfe_r"],
                "mae_bar": result["mae_bar"],
                "mfe_bar": result["mfe_bar"],
            }
        )

    analysis = pd.DataFrame(output_rows)

    print(f"MAE/MFE rows: {len(analysis):,}")

    if len(analysis) != len(trades):
        raise RuntimeError(
            "MAE/MFE analysis did not reproduce the complete 08AA trade set."
        )

    # ========================================================
    # EXIT CONSISTENCY CHECK
    # ========================================================

    print("\nChecking reconstructed exits...")

    exit_mismatch = 0

    for _, row in analysis.iterrows():
        original_r = float(row["r"])

        exit_type = row["exit_type"]

        if exit_type == "TP" and original_r <= 0:
            exit_mismatch += 1

        elif (
            exit_type
            in (
                "SL",
                "SAME_BAR_SL_FIRST",
            )
            and original_r >= 0
        ):
            exit_mismatch += 1

    print(f"Exit/result directional mismatches: {exit_mismatch:,}")

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for candidate in CANDIDATES:
        data = analysis[analysis["candidate_id"] == candidate].copy()

        if data.empty:
            continue

        winners = data[data["r"] > 0]

        losers = data[data["r"] < 0]

        tp_exits = data[data["exit_type"] == "TP"]

        sl_exits = data[data["exit_type"] == "SL"]

        same_bar = data[data["exit_type"] == "SAME_BAR_SL_FIRST"]

        timeouts = data[data["exit_type"] == "TIMEOUT"]

        cfg = CANDIDATES[candidate]

        row = {
            "candidate_id": candidate,
            "side": cfg["side"],
            "tp": cfg["tp"],
            "sl": cfg["sl"],
            "rr": cfg["tp"] / cfg["sl"],
            "horizon": cfg["horizon"],
            "n": len(data),
            "wins": len(winners),
            "losses": len(losers),
            "tp_exits": len(tp_exits),
            "sl_exits": len(sl_exits),
            "same_bar_sl_first": len(same_bar),
            "timeouts": len(timeouts),
            # ------------------------------------------------
            # MAE
            # ------------------------------------------------
            "mae_mean_all": safe_mean(data["mae_r"]),
            "mae_median_all": safe_median(data["mae_r"]),
            "mae_p75": percentile(
                data["mae_r"],
                0.75,
            ),
            "mae_p90": percentile(
                data["mae_r"],
                0.90,
            ),
            "mae_p95": percentile(
                data["mae_r"],
                0.95,
            ),
            # ------------------------------------------------
            # MFE
            # ------------------------------------------------
            "mfe_mean_all": safe_mean(data["mfe_r"]),
            "mfe_median_all": safe_median(data["mfe_r"]),
            "mfe_p75": percentile(
                data["mfe_r"],
                0.75,
            ),
            "mfe_p90": percentile(
                data["mfe_r"],
                0.90,
            ),
            "mfe_p95": percentile(
                data["mfe_r"],
                0.95,
            ),
            # ------------------------------------------------
            # WINNERS / LOSERS
            # ------------------------------------------------
            "mae_mean_winners": safe_mean(winners["mae_r"]),
            "mae_mean_losers": safe_mean(losers["mae_r"]),
            "mae_median_winners": safe_median(winners["mae_r"]),
            "mae_median_losers": safe_median(losers["mae_r"]),
            "mfe_mean_winners": safe_mean(winners["mfe_r"]),
            "mfe_mean_losers": safe_mean(losers["mfe_r"]),
            "mfe_median_winners": safe_median(winners["mfe_r"]),
            "mfe_median_losers": safe_median(losers["mfe_r"]),
            # ------------------------------------------------
            # TIMING
            # ------------------------------------------------
            "mean_bars_to_mae": safe_mean(data["mae_bar"]),
            "mean_bars_to_mfe": safe_mean(data["mfe_bar"]),
            "median_bars_to_mae": safe_median(data["mae_bar"]),
            "median_bars_to_mfe": safe_median(data["mfe_bar"]),
        }

        # ====================================================
        # FAILURE ANALYSIS
        # ====================================================

        if len(losers):
            for threshold in (
                0.5,
                1.0,
                1.5,
                2.0,
            ):
                count = int((losers["mae_r"] <= threshold).sum())

                row[f"losses_mae_le_{threshold:g}r"] = count

                row[f"loss_fraction_mae_le_{threshold:g}r"] = count / len(losers)

        else:
            for threshold in (
                0.5,
                1.0,
                1.5,
                2.0,
            ):
                row[f"losses_mae_le_{threshold:g}r"] = 0

                row[f"loss_fraction_mae_le_{threshold:g}r"] = np.nan

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # ========================================================
    # SAVE
    # ========================================================

    analysis.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 72)
    print("REAL-EXIT MAE / MFE SUMMARY")
    print("=" * 72)

    for _, row in summary.iterrows():
        print(f"\n{row['candidate_id']}")

        print(
            f"  Parameters: "
            f"TP={row['tp']} "
            f"SL={row['sl']} "
            f"RR={row['rr']:.3f} "
            f"H={int(row['horizon'])}"
        )

        print(f"  N: {int(row['n']):,}")

        print(f"  MAE median: {row['mae_median_all']:.3f}R")

        print(f"  MAE P90: {row['mae_p90']:.3f}R")

        print(f"  MFE median: {row['mfe_median_all']:.3f}R")

        print(f"  MFE P90: {row['mfe_p90']:.3f}R")

        print(f"  Winners MAE: {row['mae_mean_winners']:.3f}R")

        print(f"  Losers MAE: {row['mae_mean_losers']:.3f}R")

        print(f"  TP exits: {int(row['tp_exits']):,}")

        print(f"  SL exits: {int(row['sl_exits']):,}")

        print(f"  Same-bar SL first: {int(row['same_bar_sl_first']):,}")

        print(f"  Timeouts: {int(row['timeouts']):,}")

        print(f"  Losses MAE <= 1R: {row['loss_fraction_mae_le_1r']:.2%}")

        print(f"  Losses MAE <= 2R: {row['loss_fraction_mae_le_2r']:.2%}")

    print("\n" + "=" * 72)
    print("08AC COMPLETE")
    print("=" * 72)

    print(f"Trade analysis: {OUTPUT_FILE}")

    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
