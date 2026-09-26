from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIG
# ============================================================================

ACCOUNT_SIZE = 50_000.0
POINT_VALUE_MNQ = 2.0

BASE_RISK_FRACTION = 0.0025
BASE_RISK_DOLLARS = ACCOUNT_SIZE * BASE_RISK_FRACTION

ORB_MAX_RISK_FRACTION = 0.006
ORB_MAX_RISK_DOLLARS = ACCOUNT_SIZE * ORB_MAX_RISK_FRACTION

OOS_START = pd.Timestamp(
    "2020-06-23 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-06-19 23:59:59.999999",
    tz="UTC",
)

EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL = 3255


# ============================================================================
# PATHS
# ============================================================================

PAPER_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "paper"

SIZING_FILE = PAPER_DIR / "paper_sizing_replay.csv"

ORB_TOLERANCE_FILE = PAPER_DIR / "orb_risk_tolerance_trades.csv"

OUTPUT_TRADES = PAPER_DIR / "full_system_risk_tolerance_trades.csv"

OUTPUT_SUMMARY = PAPER_DIR / "full_system_risk_tolerance_summary.csv"

OUTPUT_STRATEGIES = PAPER_DIR / "full_system_risk_tolerance_by_strategy.csv"

OUTPUT_DAILY = PAPER_DIR / "full_system_risk_tolerance_daily.csv"


# ============================================================================
# HELPERS
# ============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


# ============================================================================
# LOAD FROZEN 3,255 OOS SIZING STREAM
# ============================================================================


def load_frozen_sizing() -> pd.DataFrame:

    if not SIZING_FILE.exists():
        raise FileNotFoundError(f"Frozen sizing replay not found:\n{SIZING_FILE}")

    df = pd.read_csv(SIZING_FILE)

    require_columns(
        df,
        [
            "strategy_name",
            "entry_timestamp",
            "historical_r_multiple",
            "executable_quantity",
            "actual_risk",
            "risk_per_contract",
        ],
        "paper_sizing_replay.csv",
    )

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    numeric_columns = [
        "historical_r_multiple",
        "executable_quantity",
        "actual_risk",
        "risk_per_contract",
    ]

    df["net_R"] = df["historical_r_multiple"]
    df["quantity"] = df["executable_quantity"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = df[
        (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
    ].copy()

    df = df.sort_values(
        [
            "entry_timestamp",
            "strategy_name",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


# ============================================================================
# LOAD ORB ADAPTIVE SIZING
# ============================================================================


def load_orb_adaptive() -> pd.DataFrame:

    if not ORB_TOLERANCE_FILE.exists():
        raise FileNotFoundError(
            f"ORB tolerance replay not found:\n{ORB_TOLERANCE_FILE}"
        )

    df = pd.read_csv(ORB_TOLERANCE_FILE)

    require_columns(
        df,
        [
            "scenario",
            "entry_timestamp",
            "net_R",
            "quantity",
            "actual_risk",
            "risk_per_contract",
        ],
        "orb_risk_tolerance_trades.csv",
    )

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    numeric_columns = [
        "net_R",
        "quantity",
        "actual_risk",
        "risk_per_contract",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = df[df["scenario"] == "ADAPTIVE_MAX_060"].copy()

    df = df[
        (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
    ].copy()

    df = df.sort_values(
        "entry_timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(df) != EXPECTED_COUNTS["ORB"]:
        raise AssertionError(
            "Adaptive ORB stream does not contain "
            f"{EXPECTED_COUNTS['ORB']} OOS trades. "
            f"Got {len(df)}."
        )

    return df


# ============================================================================
# BUILD PORTFOLIO SCENARIO
# ============================================================================


def build_scenario(
    frozen: pd.DataFrame,
    orb_adaptive: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:

    out = frozen.copy()

    out["scenario"] = scenario

    # ------------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------------

    if scenario == "BASELINE_025":
        out["scenario_quantity"] = out["quantity"]

        out["scenario_actual_risk"] = out["actual_risk"]

        out["scenario_pnl"] = out["net_R"] * out["scenario_actual_risk"]

        out["scenario_reason"] = "frozen_025_sizing"

        return out

    # ------------------------------------------------------------------------
    # ADAPTIVE ORB
    # ------------------------------------------------------------------------

    if scenario != "ADAPTIVE_ORB_060":
        raise ValueError(f"Unknown scenario: {scenario}")

    orb = orb_adaptive[
        [
            "entry_timestamp",
            "net_R",
            "quantity",
            "actual_risk",
            "risk_per_contract",
            "decision_reason",
        ]
    ].copy()

    orb = orb.rename(
        columns={
            "quantity": "adaptive_quantity",
            "actual_risk": "adaptive_actual_risk",
            "risk_per_contract": "adaptive_risk_per_contract",
            "decision_reason": "adaptive_reason",
        }
    )

    # Entry timestamp is unique in the frozen ORB stream.
    if orb["entry_timestamp"].duplicated().any():
        raise AssertionError("Adaptive ORB stream contains duplicate timestamps.")

    out = out.merge(
        orb,
        on="entry_timestamp",
        how="left",
        validate="many_to_one",
        suffixes=("", "_orb"),
    )

    is_orb = out["strategy_name"] == "ORB"

    missing_orb = is_orb & out["adaptive_quantity"].isna()

    if missing_orb.any():
        raise AssertionError("Some ORB trades could not be matched to adaptive sizing.")

    # Non-ORB strategies remain unchanged.
    out["scenario_quantity"] = np.where(
        is_orb,
        out["adaptive_quantity"],
        out["quantity"],
    )

    out["scenario_actual_risk"] = np.where(
        is_orb,
        out["adaptive_actual_risk"],
        out["actual_risk"],
    )

    out["scenario_reason"] = np.where(
        is_orb,
        out["adaptive_reason"],
        "frozen_025_sizing",
    )

    out["scenario_pnl"] = out["net_R"] * out["scenario_actual_risk"]

    return out


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================


def max_drawdown(
    pnl: pd.Series,
) -> float:

    if pnl.empty:
        return 0.0

    equity = pnl.cumsum()

    peak = equity.cummax()

    dd = equity - peak

    return float(dd.min())


def profit_factor(
    pnl: pd.Series,
) -> float:

    gross_profit = float(pnl[pnl > 0].sum())

    gross_loss = float(-pnl[pnl < 0].sum())

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def daily_metrics(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
    float,
    float,
]:

    daily = (
        df.groupby(df["entry_timestamp"].dt.floor("D"))["scenario_pnl"]
        .sum()
        .rename("daily_pnl")
        .reset_index()
    )

    if daily.empty:
        return (
            daily,
            0.0,
            0.0,
            0.0,
        )

    returns = daily["daily_pnl"]

    mean_daily = float(returns.mean())

    std_daily = float(returns.std(ddof=1))

    if std_daily > 0:
        sharpe = mean_daily / std_daily * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]

    if len(downside) > 0:
        downside_std = float(downside.std(ddof=1))
    else:
        downside_std = 0.0

    if downside_std > 0:
        sortino = mean_daily / downside_std * np.sqrt(252)
    else:
        sortino = 0.0

    return (
        daily,
        float(sharpe),
        float(sortino),
        float(returns.min()),
    )


def summarize(
    df: pd.DataFrame,
) -> dict[str, object]:

    pnl = df["scenario_pnl"]

    daily, sharpe, sortino, worst_day = daily_metrics(df)

    wins = int((pnl > 0).sum())

    losses = int((pnl < 0).sum())

    return {
        "scenario": str(df["scenario"].iloc[0]),
        "trades": len(df),
        "approved": int((df["scenario_quantity"] > 0).sum()),
        "rejected": int((df["scenario_quantity"] == 0).sum()),
        "approval_rate": float((df["scenario_quantity"] > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "expectancy_per_signal": float(pnl.mean()),
        "expectancy_per_executed_trade": float(pnl[df["scenario_quantity"] > 0].mean()),
        "win_rate_executed": float(
            wins / (wins + losses) if wins + losses > 0 else 0.0
        ),
        "profit_factor": profit_factor(pnl[df["scenario_quantity"] > 0]),
        "max_drawdown": max_drawdown(pnl),
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "worst_day": worst_day,
        "mean_actual_risk": float(
            df.loc[
                df["scenario_quantity"] > 0,
                "scenario_actual_risk",
            ].mean()
        ),
        "max_actual_risk": float(df["scenario_actual_risk"].max()),
        "mean_risk_fraction": float(
            (
                df.loc[
                    df["scenario_quantity"] > 0,
                    "scenario_actual_risk",
                ]
                / ACCOUNT_SIZE
            ).mean()
        ),
        "max_risk_fraction": float(df["scenario_actual_risk"].max() / ACCOUNT_SIZE),
        "max_contracts": int(df["scenario_quantity"].max()),
        "trading_days": int(daily["entry_timestamp"].nunique()),
    }


# ============================================================================
# STRATEGY CONTRIBUTION
# ============================================================================


def strategy_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, object]] = []

    total_pnl = float(df["scenario_pnl"].sum())

    for strategy_name, group in df.groupby("strategy_name"):
        pnl = group["scenario_pnl"]

        executed = group[group["scenario_quantity"] > 0]

        strategy_pnl = float(pnl.sum())

        rows.append(
            {
                "scenario": group["scenario"].iloc[0],
                "strategy_name": strategy_name,
                "signals": len(group),
                "executed": len(executed),
                "rejected": len(group) - len(executed),
                "approval_rate": (len(executed) / len(group) if len(group) else 0.0),
                "total_pnl": strategy_pnl,
                "contribution_pct": (
                    strategy_pnl / total_pnl * 100 if total_pnl != 0 else 0.0
                ),
                "expectancy": (
                    float(executed["scenario_pnl"].mean()) if len(executed) else 0.0
                ),
                "mean_actual_risk": (
                    float(executed["scenario_actual_risk"].mean())
                    if len(executed)
                    else 0.0
                ),
                "max_actual_risk": (
                    float(executed["scenario_actual_risk"].max())
                    if len(executed)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# DAILY OUTPUT
# ============================================================================


def build_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:

    daily = (
        df.groupby(df["entry_timestamp"].dt.floor("D"))
        .agg(
            daily_pnl=(
                "scenario_pnl",
                "sum",
            ),
            trades=(
                "scenario_pnl",
                "size",
            ),
            executed=(
                "scenario_quantity",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "entry_timestamp": "date",
            }
        )
    )

    daily["equity"] = daily["daily_pnl"].cumsum()

    daily["peak"] = daily["equity"].cummax()

    daily["drawdown"] = daily["equity"] - daily["peak"]

    return daily


# ============================================================================
# VALIDATION
# ============================================================================


def validate_frozen_stream(
    df: pd.DataFrame,
) -> None:

    counts = df.groupby("strategy_name").size().to_dict()

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(strategy, 0))

        if actual != expected:
            raise AssertionError(f"{strategy}: expected {expected}, got {actual}")

    total = len(df)

    if total != EXPECTED_TOTAL:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL} total OOS signals, got {total}"
        )

    print("FROZEN OOS COUNT AUDIT")

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(counts[strategy])

        print(f"{strategy:<6} {actual:>5} / {expected}")

    print(f"TOTAL  {total:>5} / {EXPECTED_TOTAL}")


def validate_adaptive_orb(
    df: pd.DataFrame,
) -> None:

    orb = df[df["strategy_name"] == "ORB"]

    # Every ORB trade with 1 contract must
    # respect the $300 ceiling.
    executed = orb[orb["scenario_quantity"] > 0]

    if (executed["scenario_actual_risk"] > ORB_MAX_RISK_DOLLARS + 1e-9).any():
        bad = executed[executed["scenario_actual_risk"] > ORB_MAX_RISK_DOLLARS + 1e-9]

        raise AssertionError(
            "Adaptive ORB exceeded $300 ceiling.\n"
            f"Maximum offending risk: "
            f"${bad['scenario_actual_risk'].max():,.2f}"
        )

    # Non-ORB strategies must remain identical
    # to frozen 0.25% sizing.
    for strategy in (
        "MRL1",
        "S2R",
        "MRS2",
    ):
        subset = df[df["strategy_name"] == strategy]

        if not np.array_equal(
            subset["scenario_quantity"].to_numpy(),
            subset["quantity"].to_numpy(),
        ):
            raise AssertionError(
                f"{strategy} sizing changed under adaptive ORB scenario."
            )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    banner("FULL SYSTEM RISK TOLERANCE REPLAY")

    print("Frozen system: MRL1 + S2R + MRS2 + ORB")

    print(f"Account size: ${ACCOUNT_SIZE:,.2f}")

    print(f"Base target risk: {BASE_RISK_FRACTION:.2%} (${BASE_RISK_DOLLARS:,.2f})")

    print(
        f"Adaptive ORB ceiling: "
        f"{ORB_MAX_RISK_FRACTION:.2%} "
        f"(${ORB_MAX_RISK_DOLLARS:,.2f})"
    )

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    frozen = load_frozen_sizing()

    orb_adaptive = load_orb_adaptive()

    validate_frozen_stream(frozen)

    print()

    print(f"Adaptive ORB trades: {len(orb_adaptive):,}")

    # ------------------------------------------------------------------------
    # BUILD BOTH SCENARIOS
    # ------------------------------------------------------------------------

    scenarios: list[pd.DataFrame] = []

    for scenario in (
        "BASELINE_025",
        "ADAPTIVE_ORB_060",
    ):
        banner(f"BUILDING: {scenario}")

        result = build_scenario(
            frozen,
            orb_adaptive,
            scenario,
        )

        scenarios.append(result)

    adaptive = scenarios[1]

    validate_adaptive_orb(adaptive)

    print("PASS — adaptive ORB ceiling <= $300")

    print("PASS — MRL1/S2R/MRS2 sizing unchanged")

    # ------------------------------------------------------------------------
    # SUMMARIES
    # ------------------------------------------------------------------------

    summary_rows = []

    strategy_rows = []

    daily_frames = []

    for scenario_df in scenarios:
        summary_rows.append(summarize(scenario_df))

        strategy_rows.append(strategy_summary(scenario_df))

        daily = build_daily(scenario_df)

        daily["scenario"] = scenario_df["scenario"].iloc[0]

        daily_frames.append(daily)

    summary_df = pd.DataFrame(summary_rows)

    strategy_df = pd.concat(
        strategy_rows,
        ignore_index=True,
    )

    daily_df = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------------
    # DELTA
    # ------------------------------------------------------------------------

    baseline = summary_df[summary_df["scenario"] == "BASELINE_025"].iloc[0]

    adaptive_summary = summary_df[summary_df["scenario"] == "ADAPTIVE_ORB_060"].iloc[0]

    print()
    banner("PORTFOLIO COMPARISON")

    display_columns = [
        "scenario",
        "trades",
        "approved",
        "rejected",
        "approval_rate",
        "total_pnl",
        "expectancy_per_signal",
        "expectancy_per_executed_trade",
        "win_rate_executed",
        "profit_factor",
        "max_drawdown",
        "daily_sharpe",
        "daily_sortino",
        "worst_day",
        "mean_actual_risk",
        "max_actual_risk",
        "max_contracts",
    ]

    print(summary_df[display_columns].to_string(index=False))

    print()
    banner("ADAPTIVE VS BASELINE DELTA")

    delta_fields = [
        "approved",
        "rejected",
        "total_pnl",
        "expectancy_per_signal",
        "expectancy_per_executed_trade",
        "profit_factor",
        "max_drawdown",
        "daily_sharpe",
        "daily_sortino",
        "worst_day",
        "mean_actual_risk",
        "max_actual_risk",
    ]

    for field in delta_fields:
        delta = float(adaptive_summary[field]) - float(baseline[field])

        print(f"{field:<35} {delta:>12,.6f}")

    # ------------------------------------------------------------------------
    # STRATEGY CONTRIBUTION
    # ------------------------------------------------------------------------

    banner("STRATEGY CONTRIBUTION")

    print(
        strategy_df[
            [
                "scenario",
                "strategy_name",
                "signals",
                "executed",
                "rejected",
                "approval_rate",
                "total_pnl",
                "contribution_pct",
                "expectancy",
                "mean_actual_risk",
                "max_actual_risk",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # DAILY
    # ------------------------------------------------------------------------

    banner("DAILY RISK CHECK")

    for scenario in (
        "BASELINE_025",
        "ADAPTIVE_ORB_060",
    ):
        subset = daily_df[daily_df["scenario"] == scenario]

        print()
        print(scenario)

        print(f"Worst day: ${subset['daily_pnl'].min():,.2f}")

        print(f"Best day: ${subset['daily_pnl'].max():,.2f}")

        print(f"Max cumulative DD: ${subset['drawdown'].min():,.2f}")

        print(f"Trading days: {len(subset):,}")

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    all_trades = pd.concat(
        scenarios,
        ignore_index=True,
    )

    OUTPUT_TRADES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_trades.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    strategy_df.to_csv(
        OUTPUT_STRATEGIES,
        index=False,
    )

    daily_df.to_csv(
        OUTPUT_DAILY,
        index=False,
    )

    # ------------------------------------------------------------------------
    # FINAL CHECKS
    # ------------------------------------------------------------------------

    if not np.isclose(
        baseline["total_pnl"],
        baseline["total_pnl"],
    ):
        raise AssertionError("Baseline P&L validation failed.")

    if (
        adaptive_summary["max_actual_risk"] > ORB_MAX_RISK_DOLLARS + 1e-9
        and adaptive["strategy_name"].eq("ORB").any()
    ):
        raise AssertionError("Adaptive system contains risk above $300.")

    banner("OUTPUT")

    print(OUTPUT_TRADES)

    print(OUTPUT_SUMMARY)

    print(OUTPUT_STRATEGIES)

    print(OUTPUT_DAILY)

    print()
    print("FULL SYSTEM RISK TOLERANCE REPLAY: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
