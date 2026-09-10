from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data


# ============================================================
# 08AE.2 — DIAGNOSTIC OF 08AA BASE REPRODUCTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

EVENTS_FILE = CACHE_DIR / "research_07_event_metadata.csv"

OUTPUT_FILE = RESULTS_DIR / "research_08ae2_reproduction_diagnostics.csv"


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

MAX_HORIZON = 40


# ============================================================
# TRADE EVALUATION
# ============================================================


def evaluate_trade(
    side,
    entry,
    future_high,
    future_low,
    future_close,
    tp,
    sl,
):
    """
    Same intrabar methodology used by the current 08AE
    reconstruction.

    Same-bar TP + SL:
        STOP FIRST

    Timeout:
        final close.
    """

    if side == "LONG":
        tp_price = entry + tp
        sl_price = entry - sl

        for bar in range(len(future_close)):
            high = future_high[bar]
            low = future_low[bar]

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            if hit_tp and hit_sl:
                return "LOSS", -1.0, bar + 1

            if hit_sl:
                return "LOSS", -1.0, bar + 1

            if hit_tp:
                return "WIN", tp / sl, bar + 1

        final_close = future_close[-1]

        r = (final_close - entry) / sl

    elif side == "SHORT":
        tp_price = entry - tp
        sl_price = entry + sl

        for bar in range(len(future_close)):
            high = future_high[bar]
            low = future_low[bar]

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

            if hit_tp and hit_sl:
                return "LOSS", -1.0, bar + 1

            if hit_sl:
                return "LOSS", -1.0, bar + 1

            if hit_tp:
                return "WIN", tp / sl, bar + 1

        final_close = future_close[-1]

        r = (entry - final_close) / sl

    else:
        raise ValueError(side)

    if r > 0:
        return "WIN", r, len(future_close)

    if r < 0:
        return "LOSS", r, len(future_close)

    return "UNRESOLVED", 0.0, len(future_close)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AE.2 — REPRODUCTION DIAGNOSTIC")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD 08AA
    # --------------------------------------------------------

    trades = pd.read_csv(TRADES_FILE)

    events = pd.read_csv(EVENTS_FILE)

    print(f"\n08AA trades: {len(trades):,}")

    print(f"Research 07 events: {len(events):,}")

    print("\n08AA columns:")

    print(list(trades.columns))

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_trade_columns = {
        "candidate_id",
        "event_id",
        "window",
        "timestamp",
        "result",
        "r",
    }

    missing = required_trade_columns - set(trades.columns)

    if missing:
        raise RuntimeError(f"08AA is missing required columns: {sorted(missing)}")

    # --------------------------------------------------------
    # LOAD MARKET
    # --------------------------------------------------------

    print("\nLoading canonical market data...")

    market = load_data()

    market["timestamp ET"] = pd.to_datetime(
        market["timestamp ET"],
        utc=True,
    )

    market = market[market["market_period"] == "RTH"].copy()

    market = market.sort_values("timestamp ET").reset_index(drop=True)

    print(f"RTH rows: {len(market):,}")

    # --------------------------------------------------------
    # EVENT MAPPING
    # --------------------------------------------------------

    data_indices = events["data_index"].to_numpy(dtype=int)

    event_positions = data_indices

    close = market["close"].to_numpy(dtype=float)

    high = market["high"].to_numpy(dtype=float)

    low = market["low"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # BUILD FUTURE OHLC
    # --------------------------------------------------------

    print("\nBuilding future OHLC arrays...")

    n_events = len(events)

    future_high = np.full(
        (n_events, MAX_HORIZON),
        np.nan,
        dtype=float,
    )

    future_low = np.full(
        (n_events, MAX_HORIZON),
        np.nan,
        dtype=float,
    )

    future_close = np.full(
        (n_events, MAX_HORIZON),
        np.nan,
        dtype=float,
    )

    for offset in range(
        1,
        MAX_HORIZON + 1,
    ):
        valid = event_positions + offset < len(market)

        rows = np.where(valid)[0]

        source_idx = event_positions[rows] + offset

        future_high[rows, offset - 1] = high[source_idx]

        future_low[rows, offset - 1] = low[source_idx]

        future_close[rows, offset - 1] = close[source_idx]

    print(f"Future OHLC shape: {future_close.shape}")

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    diagnostics = []

    for candidate, cfg in CANDIDATES.items():
        print("\n" + "-" * 72)

        print(candidate)

        print("-" * 72)

        subset = trades[trades["candidate_id"] == candidate].copy()

        print(f"Trades: {len(subset):,}")

        counts = {
            "entry_mismatch": 0,
            "result_mismatch": 0,
            "r_mismatch": 0,
            "result_and_r_mismatch": 0,
            "all_match": 0,
        }

        for _, trade in subset.iterrows():
            event_id = int(trade["event_id"])

            position = event_positions[event_id]

            # --------------------------------------------
            # Event entry
            # --------------------------------------------

            reconstructed_entry = close[position]

            stored_entry = np.nan

            # 08AA does not necessarily contain entry.
            # If it does, validate it.
            if "entry" in trade.index:
                if pd.notna(trade["entry"]):
                    stored_entry = float(trade["entry"])

            entry_match = np.isnan(stored_entry) or np.isclose(
                reconstructed_entry,
                stored_entry,
                atol=1e-10,
            )

            # --------------------------------------------
            # Reconstruct
            # --------------------------------------------

            result, r, bars = evaluate_trade(
                side=cfg["side"],
                entry=reconstructed_entry,
                future_high=future_high[event_id, : cfg["horizon"]],
                future_low=future_low[event_id, : cfg["horizon"]],
                future_close=future_close[event_id, : cfg["horizon"]],
                tp=cfg["tp"],
                sl=cfg["sl"],
            )

            stored_result = str(trade["result"])

            stored_r = float(trade["r"])

            result_match = result == stored_result

            r_match = np.isclose(
                r,
                stored_r,
                atol=1e-10,
            )

            all_match = entry_match and result_match and r_match

            if all_match:
                counts["all_match"] += 1

                continue

            # --------------------------------------------
            # Classify mismatch
            # --------------------------------------------

            if not result_match and not r_match:
                mismatch_type = "RESULT_AND_R_MISMATCH"

                counts["result_and_r_mismatch"] += 1

            elif not result_match:
                mismatch_type = "RESULT_MISMATCH"

                counts["result_mismatch"] += 1

            elif not r_match:
                mismatch_type = "R_MISMATCH"

                counts["r_mismatch"] += 1

            elif not entry_match:
                mismatch_type = "ENTRY_MISMATCH"

                counts["entry_mismatch"] += 1

            else:
                mismatch_type = "UNKNOWN_MISMATCH"

            diagnostics.append(
                {
                    "candidate_id": candidate,
                    "event_id": event_id,
                    "window": int(trade["window"]),
                    "timestamp": trade["timestamp"],
                    "stored_entry": stored_entry,
                    "reconstructed_entry": reconstructed_entry,
                    "stored_result": stored_result,
                    "reconstructed_result": result,
                    "stored_r": stored_r,
                    "reconstructed_r": r,
                    "reconstructed_bars": bars,
                    "entry_match": entry_match,
                    "result_match": result_match,
                    "r_match": r_match,
                    "all_match": all_match,
                    "mismatch_type": mismatch_type,
                }
            )

        # ----------------------------------------------------
        # Candidate summary
        # ----------------------------------------------------

        print(f"\nALL MATCH: {counts['all_match']:,}/{len(subset):,}")

        print(f"Result mismatches: {counts['result_mismatch']:,}")

        print(f"R mismatches: {counts['r_mismatch']:,}")

        print(f"Result + R mismatches: {counts['result_and_r_mismatch']:,}")

        print(f"Entry mismatches: {counts['entry_mismatch']:,}")

        # ----------------------------------------------------
        # Detailed mismatch output
        # ----------------------------------------------------

        candidate_diag = pd.DataFrame(
            [d for d in diagnostics if d["candidate_id"] == candidate]
        )

        if len(candidate_diag):
            print("\nMismatch types:")

            print(candidate_diag["mismatch_type"].value_counts().to_string())

            print("\nStored → reconstructed result:")

            print(
                pd.crosstab(
                    candidate_diag["stored_result"],
                    candidate_diag["reconstructed_result"],
                ).to_string()
            )

            print("\nFirst 20 mismatches:")

            columns = [
                "event_id",
                "window",
                "timestamp",
                "stored_result",
                "reconstructed_result",
                "stored_r",
                "reconstructed_r",
                "reconstructed_bars",
                "mismatch_type",
            ]

            print(candidate_diag[columns].head(20).to_string(index=False))

    # ========================================================
    # SAVE
    # ========================================================

    diagnostics_df = pd.DataFrame(diagnostics)

    diagnostics_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 72)

    print("08AE.2 DIAGNOSTIC COMPLETE")

    print("=" * 72)

    print(f"\nTotal mismatches: {len(diagnostics_df):,}")

    if len(diagnostics_df):
        print("\nGlobal mismatch types:")

        print(diagnostics_df["mismatch_type"].value_counts().to_string())

    print(f"\nDiagnostic file:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
