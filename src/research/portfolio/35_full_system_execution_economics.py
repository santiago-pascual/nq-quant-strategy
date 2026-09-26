from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = RESEARCH_ROOT / "results" / "portfolio" / "funded"

STRICT_INPUT = RESULTS_DIR / "adaptive_orb_xfa_strict_corrected.csv"
ADAPTIVE_INPUT = RESULTS_DIR / "adaptive_orb_xfa_adaptive_corrected.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "full_system_execution_economics.csv"
OUTPUT_TRADES = RESULTS_DIR / "full_system_execution_economics_trades.csv"


# =============================================================================
# FROZEN MODEL PARAMETERS
# =============================================================================

BASE_XFA_RISK = 125.0
MNQ_TICK_VALUE = 0.50
MNQ_ROUND_TRIP_FEE = 1.22

SLIPPAGE_TICKS = [0, 1, 2, 4, 6]
MISSED_FILL_RATES = [0.00, 0.01, 0.02, 0.05, 0.10]
RANDOM_SEED_BASE = 20260925

EXPECTED_STRICT_TRADES = 2_075
EXPECTED_ADAPTIVE_TRADES = 2_959


# =============================================================================
# UTILITIES
# =============================================================================


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise AssertionError(
            "\nCould not find required column.\n"
            f"Candidates: {candidates}\n"
            "Available columns:\n" + "\n".join(f"  - {c}" for c in df.columns)
        )

    return None


def max_drawdown(daily_pnl: pd.Series) -> float:
    cumulative = daily_pnl.cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    return float(dd.min())


def sharpe(daily_pnl: pd.Series) -> float:
    x = daily_pnl.to_numpy(dtype=float)

    if len(x) < 2:
        return np.nan

    std = x.std(ddof=1)

    if std == 0:
        return np.nan

    return float(np.sqrt(252.0) * x.mean() / std)


def sortino(daily_pnl: pd.Series) -> float:
    x = daily_pnl.to_numpy(dtype=float)

    if len(x) < 2:
        return np.nan

    downside = x[x < 0]

    if len(downside) == 0:
        return np.inf

    downside_rms = np.sqrt(np.mean(downside**2))

    if downside_rms == 0:
        return np.nan

    return float(np.sqrt(252.0) * x.mean() / downside_rms)


# =============================================================================
# LOAD FROZEN XFA STREAM
# =============================================================================


def load_stream(
    path: Path,
    expected_count: int,
    scenario: str,
) -> pd.DataFrame:
    print()
    print("-" * 80)
    print(f"Loading {scenario}")
    print(path)
    print("-" * 80)

    if not path.exists():
        raise FileNotFoundError(f"Missing input file:\n{path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Raw rows: {len(df):,}")
    print("Columns:")
    print(", ".join(df.columns))

    # -------------------------------------------------------------------------
    # Required economic field
    # -------------------------------------------------------------------------

    xfa_r_col = find_column(
        df,
        [
            "xfa_r_multiple",
            "xfa_R_multiple",
            "xfa_r",
        ],
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    timestamp_col = find_column(
        df,
        [
            "entry_timestamp",
            "timestamp",
            "entry_time",
        ],
    )

    # -------------------------------------------------------------------------
    # Strategy
    # -------------------------------------------------------------------------

    strategy_col = find_column(
        df,
        [
            "strategy_name",
            "strategy",
            "name",
        ],
        required=False,
    )

    # -------------------------------------------------------------------------
    # IMPORTANT QUANTITY LOGIC
    #
    # The corrected adaptive stream contains both:
    #
    #   executable_quantity -> original strict 0.25% executable quantity
    #   adaptive_quantity    -> quantity actually used by adaptive policy
    #
    # The 884 recovered ORB trades have executable_quantity == 0 but
    # adaptive_quantity == 1.
    #
    # Therefore quantity selection MUST be scenario-specific.
    # -------------------------------------------------------------------------

    if scenario == "STRICT_025":
        quantity_col = find_column(
            df,
            [
                "strict_quantity",
                "quantity",
                "executable_quantity",
                "scenario_quantity",
            ],
            required=False,
        )

        actual_risk_col = find_column(
            df,
            [
                "strict_actual_risk",
                "actual_risk",
                "scenario_actual_risk",
            ],
            required=False,
        )

    elif scenario == "ADAPTIVE_ORB_060":
        quantity_col = find_column(
            df,
            [
                "adaptive_quantity",
                "quantity",
                "scenario_quantity",
                "executable_quantity",
            ],
            required=False,
        )

        actual_risk_col = find_column(
            df,
            [
                "adaptive_actual_risk",
                "actual_risk",
                "scenario_actual_risk",
            ],
            required=False,
        )

    else:
        raise AssertionError(f"Unknown execution scenario: {scenario}")

    # -------------------------------------------------------------------------
    # Risk per contract
    # -------------------------------------------------------------------------

    risk_per_contract_col = find_column(
        df,
        [
            "risk_per_contract",
            "adaptive_risk_per_contract",
        ],
        required=False,
    )

    # -------------------------------------------------------------------------
    # Normalize
    # -------------------------------------------------------------------------

    df["_entry_timestamp"] = pd.to_datetime(
        df[timestamp_col],
        utc=True,
        errors="coerce",
    )

    df["_xfa_r_multiple"] = pd.to_numeric(
        df[xfa_r_col],
        errors="coerce",
    )

    if strategy_col is not None:
        df["_strategy_name"] = df[strategy_col].astype(str).str.strip()
    else:
        df["_strategy_name"] = "UNKNOWN"

    # -------------------------------------------------------------------------
    # Quantity reconstruction
    # -------------------------------------------------------------------------

    if quantity_col is not None:
        quantity = pd.to_numeric(
            df[quantity_col],
            errors="coerce",
        )

    elif actual_risk_col is not None and risk_per_contract_col is not None:
        actual_risk = pd.to_numeric(
            df[actual_risk_col],
            errors="coerce",
        )

        risk_per_contract = pd.to_numeric(
            df[risk_per_contract_col],
            errors="coerce",
        )

        quantity = actual_risk / risk_per_contract

    else:
        raise AssertionError(
            f"{scenario}: could not determine contract quantity.\n"
            "The file needs a scenario-specific quantity column or "
            "actual_risk + risk_per_contract."
        )

    df["_contracts"] = pd.to_numeric(
        quantity,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Hard validation
    # -------------------------------------------------------------------------

    missing_timestamp = int(df["_entry_timestamp"].isna().sum())
    missing_r = int(df["_xfa_r_multiple"].isna().sum())
    missing_qty = int(df["_contracts"].isna().sum())

    if missing_timestamp:
        raise AssertionError(f"{scenario}: {missing_timestamp} invalid timestamps.")

    if missing_r:
        raise AssertionError(f"{scenario}: {missing_r} invalid xfa_r_multiple values.")

    if missing_qty:
        raise AssertionError(f"{scenario}: {missing_qty} invalid quantities.")

    non_positive = int((df["_contracts"] <= 0).sum())

    if non_positive:
        # Give diagnostic information instead of silently accepting a
        # scenario mismatch.
        if scenario == "ADAPTIVE_ORB_060" and "adaptive_quantity" in df.columns:
            recovered = int(
                (
                    (pd.to_numeric(df["adaptive_quantity"], errors="coerce") > 0)
                    & (
                        pd.to_numeric(
                            df.get("executable_quantity", 0),
                            errors="coerce",
                        )
                        <= 0
                    )
                ).sum()
            )
            raise AssertionError(
                f"{scenario}: {non_positive} non-positive contract quantities. "
                f"Scenario-specific adaptive_quantity recovery candidates: "
                f"{recovered}."
            )

        raise AssertionError(
            f"{scenario}: {non_positive} non-positive contract quantities."
        )

    # -------------------------------------------------------------------------
    # Frozen-count validation
    # -------------------------------------------------------------------------

    if len(df) != expected_count:
        raise AssertionError(
            f"{scenario}: expected {expected_count:,} trades, got {len(df):,}."
        )

    # -------------------------------------------------------------------------
    # Stable historical ordering
    # -------------------------------------------------------------------------

    df = df.sort_values(
        [
            "_entry_timestamp",
            "_strategy_name",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    df["_scenario"] = scenario

    print(f"Validated rows: {len(df):,}")
    print(
        f"Date range: {df['_entry_timestamp'].min()} -> {df['_entry_timestamp'].max()}"
    )
    print(f"Mean contracts: {df['_contracts'].mean():.4f}")
    print(f"Max contracts: {df['_contracts'].max():.0f}")

    if scenario == "ADAPTIVE_ORB_060":
        adaptive_recovered = int(
            (
                (df["_strategy_name"] == "ORB")
                & (
                    pd.to_numeric(
                        df.get("adaptive_quantity", 0),
                        errors="coerce",
                    )
                    > 0
                )
                & (
                    pd.to_numeric(
                        df.get("executable_quantity", 0),
                        errors="coerce",
                    )
                    <= 0
                )
            ).sum()
        )
        print(f"Adaptive recovered ORB contracts: {adaptive_recovered:,}")

    return df


# =============================================================================
# EXECUTION MODEL
# =============================================================================


def execute_stress(
    stream: pd.DataFrame,
    slippage_ticks: int,
    missed_fill_rate: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:

    x = stream.copy()
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Missed fills
    # -------------------------------------------------------------------------

    x["_missed_fill"] = rng.random(len(x)) < missed_fill_rate

    # -------------------------------------------------------------------------
    # Gross P&L
    #
    # xfa_r_multiple already contains the economic risk multiplier from 32.
    # Therefore:
    #
    #     gross P&L = xfa_R * $125
    #
    # Do NOT reconstruct adaptive sizing here.
    # -------------------------------------------------------------------------

    x["_gross_pnl"] = x["_xfa_r_multiple"] * BASE_XFA_RISK

    # -------------------------------------------------------------------------
    # Commission
    # -------------------------------------------------------------------------

    x["_commission"] = x["_contracts"] * MNQ_ROUND_TRIP_FEE

    # -------------------------------------------------------------------------
    # Slippage
    #
    # slippage_ticks is per side.
    # Round trip = ticks * $0.50 * 2.
    # -------------------------------------------------------------------------

    x["_slippage"] = x["_contracts"] * slippage_ticks * MNQ_TICK_VALUE * 2.0

    # -------------------------------------------------------------------------
    # Net P&L before missed fill
    # -------------------------------------------------------------------------

    x["_net_before_missed"] = x["_gross_pnl"] - x["_commission"] - x["_slippage"]

    # -------------------------------------------------------------------------
    # Missed trade
    # -------------------------------------------------------------------------

    x["_net_pnl"] = np.where(
        x["_missed_fill"],
        0.0,
        x["_net_before_missed"],
    )

    x["_executed"] = ~x["_missed_fill"]

    # -------------------------------------------------------------------------
    # Daily
    # -------------------------------------------------------------------------

    x["_date"] = x["_entry_timestamp"].dt.date

    daily = x.groupby(
        "_date",
        as_index=False,
    )["_net_pnl"].sum()

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    net = float(x["_net_pnl"].sum())
    executed = int(x["_executed"].sum())

    wins = int((x["_net_pnl"] > 0).sum())
    losses = int((x["_net_pnl"] < 0).sum())

    gross_profit = float(
        x.loc[
            x["_net_pnl"] > 0,
            "_net_pnl",
        ].sum()
    )

    gross_loss = float(
        -x.loc[
            x["_net_pnl"] < 0,
            "_net_pnl",
        ].sum()
    )

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    summary = {
        "scenario": x["_scenario"].iloc[0],
        "slippage_ticks_per_side": slippage_ticks,
        "missed_fill_rate": missed_fill_rate,
        "signals": len(x),
        "executed_trades": executed,
        "missed_trades": len(x) - executed,
        "execution_rate": executed / len(x),
        "gross_pnl": float(x["_gross_pnl"].sum()),
        "commission": float(x["_commission"].sum()),
        "slippage_cost": float(x["_slippage"].sum()),
        "net_pnl": net,
        "expectancy_per_signal": net / len(x),
        "expectancy_per_executed": (net / executed if executed else np.nan),
        "win_rate": (wins / (wins + losses) if wins + losses else np.nan),
        "profit_factor": profit_factor,
        "max_DD": max_drawdown(daily["_net_pnl"]),
        "worst_day": float(daily["_net_pnl"].min()),
        "best_day": float(daily["_net_pnl"].max()),
        "daily_Sharpe": sharpe(daily["_net_pnl"]),
        "daily_Sortino": sortino(daily["_net_pnl"]),
    }

    return summary, x


# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("FULL-SYSTEM EXECUTION ECONOMICS — CORRECTED QUANTITY ROUTING")
print("=" * 80)
print(f"Results directory: {RESULTS_DIR}")

if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Results directory does not exist:\n{RESULTS_DIR}")

strict = load_stream(
    STRICT_INPUT,
    expected_count=EXPECTED_STRICT_TRADES,
    scenario="STRICT_025",
)

adaptive = load_stream(
    ADAPTIVE_INPUT,
    expected_count=EXPECTED_ADAPTIVE_TRADES,
    scenario="ADAPTIVE_ORB_060",
)


# =============================================================================
# STREAM AUDIT
# =============================================================================

print()
print("=" * 80)
print("FROZEN STREAM AUDIT")
print("=" * 80)

print(f"STRICT trades:   {len(strict):,}")
print(f"ADAPTIVE trades: {len(adaptive):,}")

if len(strict) != EXPECTED_STRICT_TRADES:
    raise AssertionError("Strict stream count mismatch.")

if len(adaptive) != EXPECTED_ADAPTIVE_TRADES:
    raise AssertionError("Adaptive stream count mismatch.")

print("Frozen stream counts: PASS")


# =============================================================================
# QUANTITY AUDIT
# =============================================================================

print()
print("=" * 80)
print("SCENARIO QUANTITY AUDIT")
print("=" * 80)

strict_recovered = int(
    (
        (strict["_strategy_name"] == "ORB")
        & (
            pd.to_numeric(
                strict.get("adaptive_quantity", 0),
                errors="coerce",
            )
            > 0
        )
        & (
            pd.to_numeric(
                strict.get("executable_quantity", 0),
                errors="coerce",
            )
            <= 0
        )
    ).sum()
)

adaptive_recovered = int(
    (
        (adaptive["_strategy_name"] == "ORB")
        & (
            pd.to_numeric(
                adaptive.get("adaptive_quantity", 0),
                errors="coerce",
            )
            > 0
        )
        & (
            pd.to_numeric(
                adaptive.get("executable_quantity", 0),
                errors="coerce",
            )
            <= 0
        )
    ).sum()
)

print(f"Strict recovered ORB candidates:   {strict_recovered:,}")
print(f"Adaptive recovered ORB candidates: {adaptive_recovered:,}")

if adaptive_recovered != 884:
    raise AssertionError(
        f"Expected 884 adaptive recovered ORB trades, found {adaptive_recovered}."
    )

if adaptive["_contracts"].le(0).any():
    raise AssertionError(
        "Adaptive stream still contains non-positive contract quantities."
    )

print("PASS — adaptive recovered ORB trades use adaptive_quantity.")


# =============================================================================
# STRESS GRID
# =============================================================================

print()
print("=" * 80)
print("EXECUTION STRESS GRID")
print("=" * 80)

summary_rows = []
trade_rows = []

streams = [
    strict,
    adaptive,
]

for stream in streams:
    scenario = stream["_scenario"].iloc[0]

    for slippage in SLIPPAGE_TICKS:
        for missed_rate in MISSED_FILL_RATES:
            seed = RANDOM_SEED_BASE + slippage * 100 + int(missed_rate * 1000)

            summary, trades = execute_stress(
                stream,
                slippage_ticks=slippage,
                missed_fill_rate=missed_rate,
                seed=seed,
            )

            summary_rows.append(summary)

            trades["_slippage_ticks_per_side"] = slippage
            trades["_missed_fill_rate"] = missed_rate

            trade_rows.append(trades)


summary_df = pd.DataFrame(summary_rows)

trades_df = pd.concat(
    trade_rows,
    ignore_index=True,
)


# =============================================================================
# GRID AUDIT
# =============================================================================

expected_scenarios = 2 * len(SLIPPAGE_TICKS) * len(MISSED_FILL_RATES)

if len(summary_df) != expected_scenarios:
    raise AssertionError(
        f"Expected {expected_scenarios} scenarios, got {len(summary_df)}."
    )

print(f"Scenario rows: {len(summary_df)}")
print("Stress grid: PASS")


# =============================================================================
# CORE RESULTS
# =============================================================================

print()
print("=" * 80)
print("CORE RESULTS — 0% MISSED FILLS")
print("=" * 80)

core = summary_df.loc[summary_df["missed_fill_rate"] == 0.0].copy()

print(
    core[
        [
            "scenario",
            "slippage_ticks_per_side",
            "net_pnl",
            "expectancy_per_executed",
            "profit_factor",
            "max_DD",
            "daily_Sharpe",
            "daily_Sortino",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 2-TICK / 5% MISSED
# =============================================================================

print()
print("=" * 80)
print("2-TICK + 5% MISSED FILLS")
print("=" * 80)

stress = summary_df.loc[
    (summary_df["slippage_ticks_per_side"] == 2)
    & (summary_df["missed_fill_rate"] == 0.05)
]

print(
    stress[
        [
            "scenario",
            "executed_trades",
            "net_pnl",
            "expectancy_per_executed",
            "profit_factor",
            "max_DD",
            "daily_Sharpe",
            "daily_Sortino",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 2-TICK / 0% MISSED
# =============================================================================

print()
print("=" * 80)
print("2-TICK + 0% MISSED FILLS")
print("=" * 80)

two_tick = summary_df.loc[
    (summary_df["slippage_ticks_per_side"] == 2)
    & (summary_df["missed_fill_rate"] == 0.0)
]

print(
    two_tick[
        [
            "scenario",
            "executed_trades",
            "net_pnl",
            "expectancy_per_executed",
            "profit_factor",
            "max_DD",
            "daily_Sharpe",
            "daily_Sortino",
        ]
    ].to_string(index=False)
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

trades_df.to_csv(
    OUTPUT_TRADES,
    index=False,
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(OUTPUT_SUMMARY)
print(OUTPUT_TRADES)

print()
print("=" * 80)
print("FULL-SYSTEM EXECUTION ECONOMICS — CORRECTED: PASS")
print("=" * 80)
