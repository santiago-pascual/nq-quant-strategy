from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# 08AE.3 — MRL1 EXACT REPRODUCTION DIAGNOSTIC
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

EVENTS_FILE = CACHE_DIR / "research_07_event_metadata.csv"

OUTPUT_FILE = RESULTS_DIR / "research_08ae3_mrl1_diagnostics.csv"


TP = 25.0
SL = 37.5
HORIZON = 8

MAX_HORIZON = 40


# ============================================================
# EXACT MRL1 EVALUATION
# ============================================================


def evaluate_trade(
    entry,
    future_high,
    future_low,
    future_close,
):
    """
    MRL1:

        LONG
        TP = 25
        SL = 37.5
        H = 8

    Same-bar TP + SL:
        STOP FIRST

    Timeout:
        final close of horizon.
    """

    tp_price = entry + TP
    sl_price = entry - SL

    for bar in range(len(future_close)):
        high = future_high[bar]
        low = future_low[bar]

        hit_tp = high >= tp_price
        hit_sl = low <= sl_price

        if hit_tp and hit_sl:
            return (
                "LOSS",
                -1.0,
                bar + 1,
            )

        if hit_sl:
            return (
                "LOSS",
                -1.0,
                bar + 1,
            )

        if hit_tp:
            return (
                "WIN",
                TP / SL,
                bar + 1,
            )

    final_close = future_close[-1]

    r = (final_close - entry) / SL

    if r > 0:
        return (
            "TIMEOUT_WIN",
            r,
            len(future_close),
        )

    if r < 0:
        return (
            "TIMEOUT_LOSS",
            r,
            len(future_close),
        )

    return (
        "TIMEOUT",
        0.0,
        len(future_close),
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AE.3 — MRL1 REPRODUCTION DIAGNOSTIC")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    trades = pd.read_csv(TRADES_FILE)

    events = pd.read_csv(EVENTS_FILE)

    mrl1 = trades[trades["candidate_id"] == "MRL1_NEW"].copy()

    print(f"\nMRL1 trades: {len(mrl1):,}")

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    print("\nLoading canonical market...")

    market = load_data()

    market["timestamp ET"] = pd.to_datetime(
        market["timestamp ET"],
        utc=True,
    )

    market = market[market["market_period"] == "RTH"].copy()

    market = market.sort_values("timestamp ET").reset_index(drop=True)

    print(f"RTH rows: {len(market):,}")

    # --------------------------------------------------------
    # EVENT → RTH
    # --------------------------------------------------------

    event_positions = events["data_index"].to_numpy(dtype=int)

    close = market["close"].to_numpy(dtype=float)

    high = market["high"].to_numpy(dtype=float)

    low = market["low"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------

    diagnostics = []

    for _, trade in mrl1.iterrows():
        event_id = int(trade["event_id"])

        position = event_positions[event_id]

        entry = close[position]

        future_high = high[position + 1 : position + 1 + HORIZON]

        future_low = low[position + 1 : position + 1 + HORIZON]

        future_close = close[position + 1 : position + 1 + HORIZON]

        # Safety
        if len(future_close) != HORIZON:
            raise RuntimeError(f"Insufficient future data for event {event_id}")

        reconstructed_result, reconstructed_r, bars = evaluate_trade(
            entry=entry,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
        )

        stored_result = str(trade["result"])

        stored_r = float(trade["r"])

        result_match = reconstructed_result == stored_result

        r_match = np.isclose(
            reconstructed_r,
            stored_r,
            atol=1e-10,
        )

        if not result_match:
            # ------------------------------------------------
            # Determine whether any barrier was touched
            # ------------------------------------------------

            tp_price = entry + TP
            sl_price = entry - SL

            tp_hits = future_high >= tp_price

            sl_hits = future_low <= sl_price

            tp_hit_indices = np.where(tp_hits)[0]

            sl_hit_indices = np.where(sl_hits)[0]

            first_tp = int(tp_hit_indices[0] + 1) if len(tp_hit_indices) else np.nan

            first_sl = int(sl_hit_indices[0] + 1) if len(sl_hit_indices) else np.nan

            # ------------------------------------------------
            # Store complete path
            # ------------------------------------------------

            row = {
                "event_id": event_id,
                "window": int(trade["window"]),
                "timestamp": trade["timestamp"],
                "entry": entry,
                "stored_result": stored_result,
                "reconstructed_result": reconstructed_result,
                "stored_r": stored_r,
                "reconstructed_r": reconstructed_r,
                "bars": bars,
                "result_match": result_match,
                "r_match": r_match,
                "first_tp_bar": first_tp,
                "first_sl_bar": first_sl,
                "final_close": future_close[-1],
                "final_r": (future_close[-1] - entry) / SL,
            }

            # Add all 8 future bars
            for i in range(HORIZON):
                row[f"high_{i + 1}"] = future_high[i]

                row[f"low_{i + 1}"] = future_low[i]

                row[f"close_{i + 1}"] = future_close[i]

            diagnostics.append(row)

    # ========================================================
    # OUTPUT
    # ========================================================

    diagnostic_df = pd.DataFrame(diagnostics)

    diagnostic_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 72)

    print("DIAGNOSTIC RESULT")

    print("=" * 72)

    print(f"\nMismatches found: {len(diagnostic_df)}")

    if len(diagnostic_df):
        print("\nStored → reconstructed:")

        print(
            pd.crosstab(
                diagnostic_df["stored_result"],
                diagnostic_df["reconstructed_result"],
            ).to_string()
        )

        print("\nDetailed mismatches:")

        display_columns = [
            "event_id",
            "window",
            "timestamp",
            "entry",
            "stored_result",
            "reconstructed_result",
            "stored_r",
            "reconstructed_r",
            "bars",
            "first_tp_bar",
            "first_sl_bar",
            "final_close",
            "final_r",
        ]

        print(diagnostic_df[display_columns].to_string(index=False))

        print("\nFull OHLC paths saved to:")

        print(OUTPUT_FILE)

    else:
        print("\nNO MISMATCHES.")


if __name__ == "__main__":
    main()
