from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

FUNDED_SCRIPT = (
    ROOT / "src" / "research" / "portfolio" / "22_full_system_funded_simulation.py"
)

PAPER_SIZING_PATH = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "paper"
    / "paper_sizing_replay.csv"
)

OUTPUT_DIR = ROOT / "src" / "research" / "results" / "portfolio" / "funded"

OUTPUT_PATH = OUTPUT_DIR / "adaptive_orb_xfa_results.csv"

ACCOUNT_SIZE = 50_000.0

BASE_RISK_FRACTION = 0.0025
BASE_RISK_DOLLARS = ACCOUNT_SIZE * BASE_RISK_FRACTION

ORB_MAX_RISK_FRACTION = 0.0060
ORB_MAX_RISK_DOLLARS = ACCOUNT_SIZE * ORB_MAX_RISK_FRACTION

N_SIMULATIONS = 50_000
RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "full_system_funded_simulation",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load funded simulation module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =============================================================================
# LOAD FROZEN SYSTEM
# =============================================================================


def load_frozen(module: ModuleType) -> pd.DataFrame:
    df = module.load_trades().copy()

    required = {
        "entry_timestamp",
        "r_multiple",
        "strategy_name",
        "date_ny",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"Frozen system missing columns: {sorted(missing)}")

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    df["strategy_name"] = df["strategy_name"].astype(str).str.upper().str.strip()

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="raise",
    )

    df = df.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    expected = {
        "MRL1": 430,
        "S2R": 520,
        "MRS2": 863,
        "ORB": 1442,
    }

    print("FROZEN OOS COUNT AUDIT")

    for strategy, expected_count in expected.items():
        actual = int((df["strategy_name"] == strategy).sum())

        print(f"{strategy:<8} {actual:>5} / {expected_count}")

        if actual != expected_count:
            raise RuntimeError(f"{strategy}: expected {expected_count}, found {actual}")

    print(f"{'TOTAL':<8} {len(df):>5} / 3255")

    if len(df) != 3255:
        raise RuntimeError(f"Expected 3255 trades, found {len(df)}")

    return df


# =============================================================================
# LOAD FROZEN PAPER SIZING
# =============================================================================


def load_sizing() -> pd.DataFrame:
    if not PAPER_SIZING_PATH.exists():
        raise FileNotFoundError(f"Missing paper sizing file:\n{PAPER_SIZING_PATH}")

    df = pd.read_csv(PAPER_SIZING_PATH)

    required = {
        "strategy_name",
        "entry_timestamp",
        "theoretical_quantity",
        "risk_per_contract",
        "executable_quantity",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"paper_sizing_replay.csv missing columns: {sorted(missing)}"
        )

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    df["strategy_name"] = df["strategy_name"].astype(str).str.upper().str.strip()

    for column in [
        "theoretical_quantity",
        "risk_per_contract",
        "executable_quantity",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


# =============================================================================
# BUILD ADAPTIVE SIZING
# =============================================================================


def build_adaptive_stream(
    frozen: pd.DataFrame,
    sizing: pd.DataFrame,
) -> pd.DataFrame:

    sizing = sizing[
        [
            "strategy_name",
            "entry_timestamp",
            "theoretical_quantity",
            "risk_per_contract",
            "executable_quantity",
        ]
    ].copy()

    out = frozen.merge(
        sizing,
        on=[
            "strategy_name",
            "entry_timestamp",
        ],
        how="left",
        validate="one_to_one",
    )

    if out["theoretical_quantity"].isna().any():
        raise RuntimeError("Sizing merge failed: missing frozen sizing rows.")

    # -------------------------------------------------------------------------
    # Base columns
    # -------------------------------------------------------------------------

    out["adaptive_quantity"] = 0
    out["adaptive_actual_risk"] = 0.0
    out["risk_multiplier"] = 0.0
    out["adaptive_executed"] = False
    out["adaptive_rule"] = ""

    # -------------------------------------------------------------------------
    # MR
    # -------------------------------------------------------------------------

    mr_mask = out["strategy_name"].isin(["MRL1", "S2R", "MRS2"])

    mr_quantity = np.floor(out.loc[mr_mask, "theoretical_quantity"]).astype(int)

    if (mr_quantity < 1).any():
        raise RuntimeError("Unexpected MR trade requiring less than one MNQ.")

    out.loc[
        mr_mask,
        "adaptive_quantity",
    ] = mr_quantity

    out.loc[
        mr_mask,
        "adaptive_actual_risk",
    ] = mr_quantity * out.loc[mr_mask, "risk_per_contract"]

    if (
        out.loc[
            mr_mask,
            "adaptive_actual_risk",
        ]
        > BASE_RISK_DOLLARS + 1e-9
    ).any():
        raise RuntimeError("MR adaptive sizing exceeded $125.")

    out.loc[
        mr_mask,
        "risk_multiplier",
    ] = (
        out.loc[
            mr_mask,
            "adaptive_actual_risk",
        ]
        / BASE_RISK_DOLLARS
    )

    out.loc[
        mr_mask,
        "adaptive_executed",
    ] = True

    out.loc[
        mr_mask,
        "adaptive_rule",
    ] = "MR_STRICT_025"

    # -------------------------------------------------------------------------
    # ORB
    # -------------------------------------------------------------------------

    orb_mask = out["strategy_name"].eq("ORB")

    theoretical = out.loc[
        orb_mask,
        "theoretical_quantity",
    ]

    risk_per_contract = out.loc[
        orb_mask,
        "risk_per_contract",
    ]

    strict_quantity = np.floor(theoretical).astype(int)

    # Strict integer sizing.
    strict_risk = strict_quantity * risk_per_contract

    # Adaptive rule:
    #
    # If strict_quantity == 0,
    # try exactly ONE MNQ.
    #
    # Accept only if:
    #
    #     one-contract risk <= $300
    #
    one_contract_risk = risk_per_contract

    adaptive_quantity = strict_quantity.copy()

    subunit_mask = strict_quantity < 1

    allow_one = subunit_mask & (one_contract_risk <= ORB_MAX_RISK_DOLLARS)

    adaptive_quantity.loc[allow_one] = 1

    adaptive_actual_risk = adaptive_quantity * risk_per_contract

    out.loc[
        orb_mask,
        "adaptive_quantity",
    ] = adaptive_quantity

    out.loc[
        orb_mask,
        "adaptive_actual_risk",
    ] = adaptive_actual_risk

    out.loc[
        orb_mask,
        "risk_multiplier",
    ] = np.where(
        adaptive_quantity > 0,
        adaptive_actual_risk / BASE_RISK_DOLLARS,
        0.0,
    )

    out.loc[
        orb_mask,
        "adaptive_executed",
    ] = adaptive_quantity > 0

    out.loc[
        orb_mask & (strict_quantity >= 1),
        "adaptive_rule",
    ] = "ORB_STRICT_INTEGER"

    out.loc[
        orb_mask & subunit_mask & (adaptive_quantity == 1),
        "adaptive_rule",
    ] = "ORB_FORCE_1_UNDER_060"

    out.loc[
        orb_mask & subunit_mask & (adaptive_quantity == 0),
        "adaptive_rule",
    ] = "ORB_REJECT_OVER_060"

    # -------------------------------------------------------------------------
    # HARD SAFETY AUDITS
    # -------------------------------------------------------------------------

    executed = out["adaptive_executed"]

    max_risk = float(
        out.loc[
            executed,
            "adaptive_actual_risk",
        ].max()
    )

    if max_risk > ORB_MAX_RISK_DOLLARS + 1e-9:
        raise RuntimeError(f"Adaptive ORB exceeded $300: ${max_risk:.2f}")

    mr_max = float(
        out.loc[
            mr_mask,
            "adaptive_actual_risk",
        ].max()
    )

    if mr_max > BASE_RISK_DOLLARS + 1e-9:
        raise RuntimeError(f"MR risk exceeded $125: ${mr_max:.2f}")

    return out


# =============================================================================
# BUILD ADAPTIVE HISTORICAL SEQUENCE
# =============================================================================


def build_adaptive_sequence(
    module: ModuleType,
    adaptive: pd.DataFrame,
):
    """
    IMPORTANT:

    We keep ALL 3255 historical signals in the sequence.

    Rejected adaptive trades receive zero return.

    This preserves:
        - same trade positions in the historical timeline
        - same day boundaries
        - same start-day indices
        - same sampled historical paths

    Therefore baseline and adaptive use identical start_days.
    """

    data = adaptive.copy()

    data["adaptive_r_multiple"] = data["r_multiple"] * data["risk_multiplier"]

    # Rejected trades become zero-P&L events.
    data.loc[
        ~data["adaptive_executed"],
        "adaptive_r_multiple",
    ] = 0.0

    data["r_multiple"] = data["adaptive_r_multiple"]

    required = [
        "entry_timestamp",
        "r_multiple",
        "strategy_name",
        "date_ny",
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise RuntimeError(f"Adaptive sequence missing: {sorted(missing)}")

    return module.HistoricalSequence(
        "ADAPTIVE_ORB_XFA",
        data[required].copy(),
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:

    banner("ADAPTIVE ORB XFA REPLAY")

    print("MRL1/S2R/MRS2: 0.25% target")

    print("ORB: 0.25% target / 0.60% maximum")

    print(f"ORB maximum risk: ${ORB_MAX_RISK_DOLLARS:,.2f}")

    print(f"Simulations: {N_SIMULATIONS:,}")

    print("IMPORTANT: baseline and adaptive use the SAME historical start days.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    if not FUNDED_SCRIPT.exists():
        raise FileNotFoundError(f"Missing:\n{FUNDED_SCRIPT}")

    module = load_module(FUNDED_SCRIPT)

    print("PASS — funded simulation module loaded")

    frozen = load_frozen(module)

    sizing = load_sizing()

    # -------------------------------------------------------------------------
    # ADAPTIVE
    # -------------------------------------------------------------------------

    adaptive = build_adaptive_stream(
        frozen,
        sizing,
    )

    executed = adaptive[adaptive["adaptive_executed"]]

    rejected = adaptive[~adaptive["adaptive_executed"]]

    print()
    print(f"Frozen stream:       {len(adaptive):,}")

    print(f"Executed adaptive:   {len(executed):,}")

    print(f"Rejected adaptive:   {len(rejected):,}")

    print(f"Adaptive ORB:        {int((executed['strategy_name'] == 'ORB').sum()):,}")

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------

    banner("ADAPTIVE SIZING AUDIT")

    for strategy in [
        "MRL1",
        "S2R",
        "MRS2",
        "ORB",
    ]:
        subset = adaptive[adaptive["strategy_name"] == strategy]

        executed_subset = subset[subset["adaptive_executed"]]

        print(
            f"{strategy:<6} "
            f"signals={len(subset):4d} "
            f"executed={len(executed_subset):4d} "
            f"rejected={len(subset) - len(executed_subset):4d} "
            f"mean_risk=${executed_subset['adaptive_actual_risk'].mean():8.2f} "
            f"max_risk=${executed_subset['adaptive_actual_risk'].max():8.2f}"
        )

    max_risk = float(executed["adaptive_actual_risk"].max())

    print()
    print(f"Maximum executed risk: ${max_risk:.2f}")

    if max_risk > ORB_MAX_RISK_DOLLARS + 1e-9:
        raise RuntimeError("FAIL — adaptive ORB risk exceeded $300.")

    print("PASS — adaptive ORB maximum risk <= $300")

    # -------------------------------------------------------------------------
    # SAME HISTORICAL SEQUENCE
    # -------------------------------------------------------------------------

    baseline_sequence = module.HistoricalSequence(
        "FULL_SYSTEM",
        frozen[
            [
                "entry_timestamp",
                "r_multiple",
                "strategy_name",
                "date_ny",
            ]
        ].copy(),
    )

    adaptive_sequence = build_adaptive_sequence(
        module,
        adaptive,
    )

    if baseline_sequence.n_trades != adaptive_sequence.n_trades:
        raise RuntimeError("Baseline/adaptive trade count differs.")

    if baseline_sequence.n_days != adaptive_sequence.n_days:
        raise RuntimeError("Baseline/adaptive day count differs.")

    print()
    print(f"Historical trades: {baseline_sequence.n_trades:,}")

    print(f"Historical days:   {baseline_sequence.n_days:,}")

    # -------------------------------------------------------------------------
    # SAME START DAYS
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    start_days = module.build_start_days(
        baseline_sequence,
        N_SIMULATIONS,
        rng,
    )

    print(f"Start days generated: {len(start_days):,}")

    # -------------------------------------------------------------------------
    # PARITY CHECK OF DAY STRUCTURE
    # -------------------------------------------------------------------------

    if not np.array_equal(
        baseline_sequence.unique_dates,
        adaptive_sequence.unique_dates,
    ):
        raise RuntimeError("Baseline/adaptive historical day structures differ.")

    if not np.array_equal(
        baseline_sequence.day_start_indices,
        adaptive_sequence.day_start_indices,
    ):
        raise RuntimeError("Baseline/adaptive day-start indices differ.")

    print("PASS — identical historical day structure")

    # -------------------------------------------------------------------------
    # BUILD SAME PATHS
    # -------------------------------------------------------------------------

    path_returns, day_ids = baseline_sequence.build_batch(
        start_days,
        module.MAX_XFA_TRADES,
    )

    adaptive_path_returns, adaptive_day_ids = adaptive_sequence.build_batch(
        start_days,
        module.MAX_XFA_TRADES,
    )

    if not np.array_equal(
        day_ids,
        adaptive_day_ids,
    ):
        raise RuntimeError("Baseline/adaptive day IDs differ.")

    # -------------------------------------------------------------------------
    # SAME XFA POLICIES
    # -------------------------------------------------------------------------

    policies = []

    for name, scenario in module.RISK_SCENARIOS.items():
        for interval in module.PAYOUT_INTERVALS:
            for amount in module.PAYOUT_AMOUNTS:
                policies.append(
                    (
                        name,
                        scenario,
                        interval,
                        float(amount),
                    )
                )

    print()
    print(f"XFA policies: {len(policies)}")

    # -------------------------------------------------------------------------
    # RUN BASELINE + ADAPTIVE THROUGH SAME XFA ENGINE
    # -------------------------------------------------------------------------

    banner("RUNNING BASELINE XFA")

    baseline_result = module.run_xfa_batch(
        path_returns,
        day_ids,
        policies,
        module.MAX_XFA_TRADES,
    )

    banner("RUNNING ADAPTIVE ORB XFA")

    adaptive_result = module.run_xfa_batch(
        adaptive_path_returns,
        adaptive_day_ids,
        policies,
        module.MAX_XFA_TRADES,
    )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    rows = []

    for (
        name,
        _scenario,
        interval,
        amount,
    ) in policies:
        key = (
            name,
            interval,
            amount,
        )

        base = baseline_result[key]
        adaptive_result_array = adaptive_result[key]

        for label, result in [
            ("BASELINE_025", base),
            ("ADAPTIVE_ORB_060", adaptive_result_array),
        ]:
            failed = result["failed"]

            survival = ~failed

            survived_max = survival & (result["trades"] >= module.MAX_XFA_TRADES)

            rows.append(
                {
                    "scenario": label,
                    "risk_policy": name,
                    "payout_interval": interval,
                    "payout_amount": amount,
                    "simulations": N_SIMULATIONS,
                    "survival_rate": float(survival.mean()),
                    "failure_rate": float(failed.mean()),
                    "survived_to_max_trades_rate": float(survived_max.mean()),
                    "median_payouts": float(np.median(result["payouts"])),
                    "median_total_withdrawn": float(
                        np.median(result["total_withdrawn"])
                    ),
                    "p95_total_withdrawn": float(
                        np.percentile(
                            result["total_withdrawn"],
                            95,
                        )
                    ),
                    "median_final_balance": float(np.median(result["final_balance"])),
                    "median_net_value": float(
                        np.median(result["final_balance"] + result["total_withdrawn"])
                    ),
                    "median_max_DD": float(np.median(result["max_drawdown"])),
                    "p95_DD": float(
                        np.percentile(
                            result["max_drawdown"],
                            5,
                        )
                    ),
                    "median_trades": float(np.median(result["trades"])),
                }
            )

    results = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # MAIN POLICY COMPARISON
    # -------------------------------------------------------------------------

    banner("XFA POLICY COMPARISON")

    for policy_name in [
        "0.25%",
        "0.50%",
        "0.75%",
        "1.00%",
        "0.50_to_1.00",
    ]:
        comparison = results[results["risk_policy"] == policy_name]

        print()
        print(f"POLICY: {policy_name}")

        print(
            comparison[
                [
                    "scenario",
                    "payout_interval",
                    "payout_amount",
                    "survival_rate",
                    "median_net_value",
                    "median_max_DD",
                    "p95_DD",
                ]
            ]
            .sort_values(
                [
                    "scenario",
                    "survival_rate",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(14)
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # DELTA FOR SAME POLICY
    # -------------------------------------------------------------------------

    banner("ADAPTIVE VS BASELINE — DELTA")

    for policy_name in [
        "0.25%",
        "0.50%",
        "0.75%",
        "1.00%",
        "0.50_to_1.00",
    ]:
        subset = results[results["risk_policy"] == policy_name].copy()

        pivot = subset.pivot_table(
            index=[
                "payout_interval",
                "payout_amount",
            ],
            columns="scenario",
            values=[
                "survival_rate",
                "median_net_value",
                "median_max_DD",
                "p95_DD",
            ],
        )

        if ("BASELINE_025" in pivot.columns.get_level_values(1)) and (
            "ADAPTIVE_ORB_060" in pivot.columns.get_level_values(1)
        ):
            delta_survival = (
                pivot[
                    (
                        "survival_rate",
                        "ADAPTIVE_ORB_060",
                    )
                ]
                - pivot[
                    (
                        "survival_rate",
                        "BASELINE_025",
                    )
                ]
            )

            delta_value = (
                pivot[
                    (
                        "median_net_value",
                        "ADAPTIVE_ORB_060",
                    )
                ]
                - pivot[
                    (
                        "median_net_value",
                        "BASELINE_025",
                    )
                ]
            )

            delta_dd = (
                pivot[
                    (
                        "median_max_DD",
                        "ADAPTIVE_ORB_060",
                    )
                ]
                - pivot[
                    (
                        "median_max_DD",
                        "BASELINE_025",
                    )
                ]
            )

            print()
            print(f"POLICY {policy_name}")

            delta = pd.DataFrame(
                {
                    "delta_survival": delta_survival,
                    "delta_median_net_value": delta_value,
                    "delta_median_DD": delta_dd,
                }
            )

            print(delta.to_string())

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    banner("OUTPUT")

    print(OUTPUT_PATH)

    banner("ADAPTIVE ORB XFA REPLAY: PASS")

    print("Same frozen 3,255-trade historical timeline.")

    print("Same 50,000 start days.")

    print("Same validated XFA engine.")

    print("Only ORB sizing changed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
