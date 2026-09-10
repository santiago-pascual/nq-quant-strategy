"""
08AF — COST + SLIPPAGE VALIDATION

Purpose
-------
Validate whether the frozen NEW mean-reversion strategies retain positive
expectancy after realistic MNQ trading costs and configurable slippage.

IMPORTANT
---------
- NEW branch only.
- No parameter optimization.
- No re-selection of events.
- Uses the exact frozen 08AA trade population.
- Gross strategy results are taken directly from the frozen 08AA results.
- Costs and slippage are applied on top of the frozen gross R results.

Frozen candidates
-----------------
MRS2_NEW:
    SHORT
    HMM2
    VOL80-100
    Z2.0
    TP 27.5
    SL 25.0
    H 30
    RR 1.10

MRL1_NEW:
    LONG
    HMM1
    VOL20-40
    Z2.5
    TP 25.0
    SL 37.5
    H 8
    RR 0.667

Execution assumptions
---------------------
Instrument:
    MNQ

Contracts:
    1 micro contract

Tick:
    0.25 points

Tick value:
    $0.50

Round-trip commission:
    $1.22 / contract

Slippage:
    0, 1, or 2 ticks per side

Therefore:
    total round-trip slippage =
        2 * slippage_ticks * $0.50

The analysis is performed in R units as well as USD.

Outputs
-------
results/research_08af_cost_slippage_summary.csv
results/research_08af_cost_slippage_trades.csv
results/research_08af_cost_slippage_windows.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"

INPUT_08AA = RESULTS_DIR / "research_08aa_full_robustness_suite.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "research_08af_cost_slippage_summary.csv"

OUTPUT_TRADES = RESULTS_DIR / "research_08af_cost_slippage_trades.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08af_cost_slippage_windows.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

FROZEN = {
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


# =============================================================================
# MNQ EXECUTION PARAMETERS
# =============================================================================

CONTRACTS = 1

TICK_SIZE = 0.25
TICK_VALUE_USD = 0.50

ROUND_TRIP_COMMISSION_USD = 1.22

# Adverse slippage per side.
#
# Example:
#   1 tick entry slippage
#   1 tick exit slippage
#
# => 2 ticks round-trip.
SLIPPAGE_TICKS_PER_SIDE = [
    0,
    1,
    2,
]


# =============================================================================
# GENERAL CONSTANTS
# =============================================================================

TOLERANCE = 1e-12


# =============================================================================
# DISPLAY
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def status(label: str, passed: bool) -> None:
    print(f"{label:<60}: {'PASS' if passed else 'FAIL'}")


# =============================================================================
# METRICS
# =============================================================================


def calculate_profit_factor(
    r_values: np.ndarray,
) -> float:

    r_values = np.asarray(
        r_values,
        dtype=float,
    )

    gross_profit = r_values[r_values > 0].sum()

    gross_loss = abs(r_values[r_values < 0].sum())

    if gross_loss <= TOLERANCE:
        if gross_profit > TOLERANCE:
            return np.inf

        return np.nan

    return float(gross_profit / gross_loss)


def calculate_max_drawdown(
    r_values: np.ndarray,
) -> float:

    if len(r_values) == 0:
        return 0.0

    equity = np.cumsum(
        np.asarray(
            r_values,
            dtype=float,
        )
    )

    running_max = np.maximum.accumulate(
        np.concatenate(
            (
                [0.0],
                equity,
            )
        )
    )[1:]

    drawdown = equity - running_max

    return float(drawdown.min())


def calculate_metrics(
    r_values: np.ndarray,
) -> Dict[str, float]:

    r_values = np.asarray(
        r_values,
        dtype=float,
    )

    n = len(r_values)

    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "total_R": 0.0,
            "expectancy_R": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": 0.0,
        }

    wins = int((r_values > 0).sum())

    losses = int((r_values < 0).sum())

    resolved = wins + losses

    if resolved > 0:
        win_rate = wins / resolved
    else:
        win_rate = np.nan

    total_R = float(r_values.sum())

    expectancy_R = float(r_values.mean())

    pf = calculate_profit_factor(r_values)

    dd = calculate_max_drawdown(r_values)

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_R": total_R,
        "expectancy_R": expectancy_R,
        "profit_factor": pf,
        "max_drawdown_R": dd,
    }


# =============================================================================
# COST CALCULATIONS
# =============================================================================


def commission_R(
    sl_points: float,
) -> float:
    """
    Convert the official round-trip commission assumption into R.

    Risk per trade:
        SL points * $0.50 per point

    Since MNQ:
        1 point = 4 ticks
        1 tick = $0.50
        therefore:
        1 point = $2.00
    """

    point_value_usd = TICK_VALUE_USD / TICK_SIZE

    risk_usd = sl_points * point_value_usd * CONTRACTS

    total_commission = ROUND_TRIP_COMMISSION_USD * CONTRACTS

    return float(total_commission / risk_usd)


def slippage_R(
    sl_points: float,
    ticks_per_side: int,
) -> float:
    """
    Convert round-trip adverse slippage into R.
    """

    total_slippage_ticks = 2 * ticks_per_side

    slippage_usd = total_slippage_ticks * TICK_VALUE_USD * CONTRACTS

    point_value_usd = TICK_VALUE_USD / TICK_SIZE

    risk_usd = sl_points * point_value_usd * CONTRACTS

    return float(slippage_usd / risk_usd)


# =============================================================================
# LOAD 08AA
# =============================================================================


def load_08aa() -> pd.DataFrame:

    banner("LOADING FROZEN 08AA")

    if not INPUT_08AA.exists():
        raise FileNotFoundError(f"Missing 08AA file:\n{INPUT_08AA}")

    df = pd.read_csv(INPUT_08AA)

    print(f"08AA trades: {len(df):,}")

    required = {
        "candidate_id",
        "event_id",
        "timestamp",
        "window",
        "side",
        "tp",
        "sl",
        "horizon",
        "result",
        "r",
        "bars",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"08AA missing columns: {sorted(missing)}")

    return df


# =============================================================================
# VALIDATE FROZEN CANDIDATES
# =============================================================================


def validate_frozen(
    df: pd.DataFrame,
) -> None:

    banner("VALIDATING FROZEN CANDIDATES")

    for candidate_id, frozen in FROZEN.items():
        candidate = df[df["candidate_id"] == candidate_id].copy()

        if candidate.empty:
            raise ValueError(f"No trades found for {candidate_id}.")

        side_ok = (candidate["side"] == frozen["side"]).all()

        tp_ok = np.isclose(
            candidate["tp"].astype(float),
            frozen["tp"],
            atol=TOLERANCE,
            rtol=0.0,
        ).all()

        sl_ok = np.isclose(
            candidate["sl"].astype(float),
            frozen["sl"],
            atol=TOLERANCE,
            rtol=0.0,
        ).all()

        horizon_ok = (candidate["horizon"].astype(int) == frozen["horizon"]).all()

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

        if not all(
            [
                side_ok,
                tp_ok,
                sl_ok,
                horizon_ok,
            ]
        ):
            raise RuntimeError(f"{candidate_id} frozen configuration mismatch.")


# =============================================================================
# APPLY COSTS
# =============================================================================


def apply_costs(
    candidate_id: str,
    trades: pd.DataFrame,
    ticks_per_side: int,
) -> pd.DataFrame:

    frozen = FROZEN[candidate_id]

    result = trades.copy()

    sl_points = float(frozen["sl"])

    commission_cost_r = commission_R(sl_points)

    slippage_cost_r = slippage_R(
        sl_points,
        ticks_per_side,
    )

    total_cost_r = commission_cost_r + slippage_cost_r

    result["gross_r"] = result["r"].astype(float)

    result["commission_R"] = commission_cost_r

    result["slippage_R"] = slippage_cost_r

    result["total_cost_R"] = total_cost_r

    result["net_r"] = result["gross_r"] - total_cost_r

    result["commission_USD"] = ROUND_TRIP_COMMISSION_USD * CONTRACTS

    result["slippage_USD"] = 2 * ticks_per_side * TICK_VALUE_USD * CONTRACTS

    result["total_cost_USD"] = result["commission_USD"] + result["slippage_USD"]

    result["slippage_ticks_per_side"] = ticks_per_side

    result["scenario"] = f"{candidate_id}_{ticks_per_side}tick"

    return result


# =============================================================================
# WINDOW METRICS
# =============================================================================


def calculate_window_metrics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[dict] = []

    for window, group in trades.groupby(
        "window",
        sort=True,
    ):
        r_values = group["net_r"].to_numpy(dtype=float)

        metrics = calculate_metrics(r_values)

        rows.append(
            {
                "scenario": group["scenario"].iloc[0],
                "candidate_id": group["candidate_id"].iloc[0],
                "slippage_ticks_per_side": int(
                    group["slippage_ticks_per_side"].iloc[0]
                ),
                "window": int(window),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("08AF — COST + SLIPPAGE VALIDATION")

    print()
    print("NEW BRANCH ONLY")

    print("No parameter optimization.")

    print("Exact frozen 08AA trade population.")

    print()

    print(f"MNQ tick size       : {TICK_SIZE}")

    print(f"MNQ tick value      : ${TICK_VALUE_USD:.2f}")

    print(f"Round-trip commission: ${ROUND_TRIP_COMMISSION_USD:.2f}")

    print(f"Contracts            : {CONTRACTS}")

    print(f"Slippage scenarios   : {SLIPPAGE_TICKS_PER_SIDE} ticks/side")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df = load_08aa()

    validate_frozen(df)

    # -------------------------------------------------------------------------
    # COST TABLE
    # -------------------------------------------------------------------------

    banner("EXECUTION COST ASSUMPTIONS")

    cost_rows = []

    for candidate_id, frozen in FROZEN.items():
        sl_points = frozen["sl"]

        commission = commission_R(sl_points)

        print()
        print(candidate_id)

        print(f"  SL                  : {sl_points:.2f} points")

        print(
            f"  Risk per contract   : ${sl_points * (TICK_VALUE_USD / TICK_SIZE):.2f}"
        )

        print(f"  Commission          : {commission:.6f}R")

        for ticks in SLIPPAGE_TICKS_PER_SIDE:
            slip = slippage_R(
                sl_points,
                ticks,
            )

            total = commission + slip

            print(f"  {ticks} tick/side slippage           : {slip:.6f}R")

            print(f"  Total cost          : {total:.6f}R")

            cost_rows.append(
                {
                    "candidate_id": candidate_id,
                    "sl_points": sl_points,
                    "slippage_ticks_per_side": ticks,
                    "commission_USD": ROUND_TRIP_COMMISSION_USD,
                    "slippage_USD": (2 * ticks * TICK_VALUE_USD),
                    "total_cost_USD": (
                        ROUND_TRIP_COMMISSION_USD + (2 * ticks * TICK_VALUE_USD)
                    ),
                    "commission_R": commission,
                    "slippage_R": slip,
                    "total_cost_R": total,
                }
            )

    # -------------------------------------------------------------------------
    # SCENARIOS
    # -------------------------------------------------------------------------

    banner("RUNNING COST + SLIPPAGE SCENARIOS")

    all_summary = []
    all_trades = []
    all_windows = []

    for candidate_id in FROZEN:
        candidate = df[df["candidate_id"] == candidate_id].copy()

        print()
        print(f"{candidate_id}: {len(candidate):,} trades")

        for ticks in SLIPPAGE_TICKS_PER_SIDE:
            scenario_trades = apply_costs(
                candidate_id=candidate_id,
                trades=candidate,
                ticks_per_side=ticks,
            )

            net_r = scenario_trades["net_r"].to_numpy(dtype=float)

            metrics = calculate_metrics(net_r)

            commission_r = commission_R(FROZEN[candidate_id]["sl"])

            slip_r = slippage_R(
                FROZEN[candidate_id]["sl"],
                ticks,
            )

            total_cost_r = commission_r + slip_r

            windows = calculate_window_metrics(scenario_trades)

            positive_windows = int((windows["total_R"] > 0).sum())

            active_windows = len(windows)

            if active_windows > 0:
                positive_window_rate = positive_windows / active_windows

            else:
                positive_window_rate = np.nan

            summary_row = {
                "scenario": scenario_trades["scenario"].iloc[0],
                "candidate_id": candidate_id,
                "tp": FROZEN[candidate_id]["tp"],
                "sl": FROZEN[candidate_id]["sl"],
                "rr": (FROZEN[candidate_id]["tp"] / FROZEN[candidate_id]["sl"]),
                "horizon": FROZEN[candidate_id]["horizon"],
                "slippage_ticks_per_side": ticks,
                "commission_USD": ROUND_TRIP_COMMISSION_USD,
                "slippage_USD": (2 * ticks * TICK_VALUE_USD),
                "total_cost_USD": (
                    ROUND_TRIP_COMMISSION_USD + (2 * ticks * TICK_VALUE_USD)
                ),
                "commission_R": commission_r,
                "slippage_R": slip_r,
                "total_cost_R": total_cost_r,
                "gross_total_R": float(candidate["r"].sum()),
                "net_total_R": metrics["total_R"],
                "trades": metrics["trades"],
                "wins": metrics["wins"],
                "losses": metrics["losses"],
                "win_rate": metrics["win_rate"],
                "gross_expectancy_R": float(candidate["r"].mean()),
                "net_expectancy_R": metrics["expectancy_R"],
                "net_profit_factor": metrics["profit_factor"],
                "net_max_drawdown_R": metrics["max_drawdown_R"],
                "positive_windows": positive_windows,
                "active_windows": active_windows,
                "positive_window_rate": positive_window_rate,
            }

            all_summary.append(summary_row)

            all_trades.append(scenario_trades)

            all_windows.append(windows)

            print(
                f"  {ticks} tick/side | "
                f"cost = "
                f"{total_cost_r:.5f}R | "
                f"net Exp = "
                f"{metrics['expectancy_R']:.6f}R | "
                f"net PF = "
                f"{metrics['profit_factor']:.4f} | "
                f"net R = "
                f"{metrics['total_R']:.2f}R | "
                f"DD = "
                f"{metrics['max_drawdown_R']:.2f}R"
            )

    summary = pd.DataFrame(all_summary)

    trade_results = pd.concat(
        all_trades,
        ignore_index=True,
    )

    window_results = pd.concat(
        all_windows,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # INTEGRITY
    # -------------------------------------------------------------------------

    banner("INTEGRITY CHECKS")

    expected_scenarios = len(FROZEN) * len(SLIPPAGE_TICKS_PER_SIDE)

    status(
        "Expected scenario count",
        len(summary) == expected_scenarios,
    )

    expected_trade_rows = len(df) * len(SLIPPAGE_TICKS_PER_SIDE)

    status(
        "Expected trade rows",
        len(trade_results) == expected_trade_rows,
    )

    gross_by_candidate = df.groupby("candidate_id")["r"].sum()

    gross_check = True

    for candidate_id in FROZEN:
        scenario = summary[summary["candidate_id"] == candidate_id]

        expected_gross = float(gross_by_candidate.loc[candidate_id])

        if not np.allclose(
            scenario["gross_total_R"].to_numpy(),
            expected_gross,
            atol=TOLERANCE,
            rtol=0.0,
        ):
            gross_check = False

    status(
        "Gross 08AA R preserved",
        gross_check,
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    banner("COST + SLIPPAGE RESULTS")

    display_columns = [
        "scenario",
        "candidate_id",
        "slippage_ticks_per_side",
        "total_cost_R",
        "gross_total_R",
        "net_total_R",
        "gross_expectancy_R",
        "net_expectancy_R",
        "net_profit_factor",
        "net_max_drawdown_R",
        "win_rate",
        "positive_windows",
        "active_windows",
        "positive_window_rate",
    ]

    print(summary[display_columns].to_string(index=False))

    # -------------------------------------------------------------------------
    # ROBUSTNESS GATE
    # -------------------------------------------------------------------------

    banner("COST + SLIPPAGE ROBUSTNESS GATE")

    for candidate_id in FROZEN:
        candidate = summary[summary["candidate_id"] == candidate_id].copy()

        positive_exp = int((candidate["net_expectancy_R"] > 0).sum())

        pf_above_one = int((candidate["net_profit_factor"] > 1).sum())

        positive_total = int((candidate["net_total_R"] > 0).sum())

        scenarios = len(candidate)

        print()
        print(candidate_id)

        print(f"  Positive net expectancy : {positive_exp}/{scenarios}")

        print(f"  Net PF > 1             : {pf_above_one}/{scenarios}")

        print(f"  Positive net total R   : {positive_total}/{scenarios}")

        status(
            "All scenarios positive expectancy",
            positive_exp == scenarios,
        )

        status(
            "All scenarios PF > 1",
            pf_above_one == scenarios,
        )

        status(
            "All scenarios positive total R",
            positive_total == scenarios,
        )

    # -------------------------------------------------------------------------
    # SAVE
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

    trade_results.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    window_results.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    print(f"Summary saved : {OUTPUT_SUMMARY}")

    print(f"Trades saved  : {OUTPUT_TRADES}")

    print(f"Windows saved : {OUTPUT_WINDOWS}")

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    banner("08AF COMPLETE")

    print(f"Candidates          : {len(FROZEN)}")

    print(f"Scenarios           : {len(summary)}")

    print(f"Trade evaluations   : {len(trade_results):,}")

    print()
    print("No parameter optimization.")

    print("No event re-selection.")

    print("Exact 08AA frozen population.")

    print("08AF finished successfully.")


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
