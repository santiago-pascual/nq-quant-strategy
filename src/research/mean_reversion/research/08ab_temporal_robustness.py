from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

TRADES_FILE = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

LOO_FILE = RESULTS_DIR / "research_08ab_loo_results.csv"
YEAR_FILE = RESULTS_DIR / "research_08ab_year_results.csv"


# ============================================================
# CANDIDATES
# ============================================================

CANDIDATES = [
    "MRS2_NEW",
    "MRL1_NEW",
]


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(df: pd.DataFrame) -> dict:

    if df.empty:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "win_rate": np.nan,
            "net_r": np.nan,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
        }

    r = df["r"].to_numpy(dtype=float)

    wins = int(np.sum(r > 0))
    losses = int(np.sum(r < 0))
    unresolved = int(np.sum(r == 0))

    gross_profit = float(r[r > 0].sum()) if wins else 0.0
    gross_loss = float(-r[r < 0].sum()) if losses else 0.0

    equity = np.cumsum(r)

    running_max = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]

    drawdown = equity - running_max

    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "n": len(df),
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_rate": wins / len(df),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else np.inf),
        "max_drawdown_r": max_dd,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("08AB — TEMPORAL ROBUSTNESS OF NEW PARAMETER BRANCH")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    trades = pd.read_csv(TRADES_FILE)

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
    )

    print(f"\nTrades loaded: {len(trades):,}")

    # ========================================================
    # LOO — LEAVE ONE OOS WINDOW OUT
    # ========================================================

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OOS-WINDOW-OUT")
    print("=" * 72)

    loo_rows = []

    for candidate in CANDIDATES:
        data = trades[trades["candidate_id"] == candidate].copy()

        windows = sorted(data["window"].dropna().unique())

        print(f"\n{candidate}: {len(data):,} trades, {len(windows)} windows")

        for omitted_window in windows:
            remaining = data[data["window"] != omitted_window]

            metrics = calculate_metrics(remaining)

            loo_rows.append(
                {
                    "candidate_id": candidate,
                    "omitted_window": int(omitted_window),
                    **metrics,
                }
            )

        loo_candidate = pd.DataFrame(
            [row for row in loo_rows if row["candidate_id"] == candidate]
        )

        print(f"  Worst LOO Net R: {loo_candidate['net_r'].min():.2f}R")

        print(f"  Worst LOO Expectancy: {loo_candidate['expectancy_r'].min():.6f}R")

        print(f"  Worst LOO PF: {loo_candidate['profit_factor'].min():.4f}")

    loo_results = pd.DataFrame(loo_rows)

    loo_results.to_csv(
        LOO_FILE,
        index=False,
    )

    # ========================================================
    # YEAR-BY-YEAR
    # ========================================================

    print("\n" + "=" * 72)
    print("YEAR-BY-YEAR STABILITY")
    print("=" * 72)

    trades["year"] = trades["timestamp"].dt.year

    year_rows = []

    for candidate in CANDIDATES:
        data = trades[trades["candidate_id"] == candidate].copy()

        years = sorted(data["year"].dropna().unique())

        print(f"\n{candidate}: {len(years)} years")

        for year in years:
            year_data = data[data["year"] == year]

            metrics = calculate_metrics(year_data)

            year_rows.append(
                {
                    "candidate_id": candidate,
                    "year": int(year),
                    **metrics,
                }
            )

            print(
                f"  {year}: "
                f"N={metrics['n']:,} "
                f"Net={metrics['net_r']:.2f}R "
                f"Exp={metrics['expectancy_r']:.6f} "
                f"PF={metrics['profit_factor']:.4f}"
            )

    year_results = pd.DataFrame(year_rows)

    year_results.to_csv(
        YEAR_FILE,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("TEMPORAL ROBUSTNESS SUMMARY")
    print("=" * 72)

    for candidate in CANDIDATES:
        loo = loo_results[loo_results["candidate_id"] == candidate]

        years = year_results[year_results["candidate_id"] == candidate]

        positive_loo = int((loo["net_r"] > 0).sum())

        positive_years = int((years["net_r"] > 0).sum())

        print(f"\n{candidate}")

        print(f"  LOO positive: {positive_loo}/{len(loo)}")

        print(f"  LOO worst Net: {loo['net_r'].min():.2f}R")

        print(f"  LOO worst Exp: {loo['expectancy_r'].min():.6f}R")

        print(f"  LOO worst PF: {loo['profit_factor'].min():.4f}")

        print(f"  Positive years: {positive_years}/{len(years)}")

        print(f"  Worst year Net: {years['net_r'].min():.2f}R")

        print(f"  Worst year Exp: {years['expectancy_r'].min():.6f}R")

        print(f"  Worst year PF: {years['profit_factor'].min():.4f}")

    print("\n" + "=" * 72)
    print("08AB COMPLETE")
    print("=" * 72)

    print(f"LOO results:  {LOO_FILE}")
    print(f"Year results: {YEAR_FILE}")


if __name__ == "__main__":
    main()
