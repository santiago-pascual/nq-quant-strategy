from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FUNDED_ENGINE_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "portfolio"
    / "22_full_system_funded_simulation.py"
)

COMMON_OOS_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio_4_strategy"
    / "portfolio_4_strategy_common_oos.csv"
)

PAPER_SIZING_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "paper"
    / "paper_sizing_replay.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SIMULATIONS = 50_000
RANDOM_SEED = 42

BASE_RISK_DOLLARS = 125.0
ORB_MAX_RISK_DOLLARS = 300.0

EXPECTED_TOTAL_SIGNALS = 3_255
EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1_442,
}

EXPECTED_STRICT_TRADES = 2_075
EXPECTED_ADAPTIVE_TRADES = 2_959
EXPECTED_RECOVERED_ORB = 884
EXPECTED_REJECTED_ORB = 296


# ============================================================
# HELPERS
# ============================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def assert_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{name}: missing required columns: {missing}")


def load_funded_engine():
    """
    Load script 22 dynamically.

    The filename starts with '22_', so it cannot be imported with a
    normal Python import statement.
    """
    if not FUNDED_ENGINE_PATH.exists():
        raise FileNotFoundError(f"Funded engine not found:\n{FUNDED_ENGINE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "full_system_funded_simulation_22",
        FUNDED_ENGINE_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for:\n{FUNDED_ENGINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "HistoricalSequence",
        "RISK_SCENARIOS",
        "PAYOUT_INTERVALS",
        "PAYOUT_AMOUNTS",
        "MAX_XFA_TRADES",
        "run_xfa_batch",
    ]

    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "22_full_system_funded_simulation.py is missing: " + ", ".join(missing)
        )

    return module


def load_common_oos() -> pd.DataFrame:
    """
    Load the authoritative 21_mr_orb_portfolio_analysis output.

    Important:
    portfolio_4_strategy_common_oos.csv is NOT guaranteed to use the
    internal funded-engine column names.  In particular, the portfolio
    artifact may expose the strategy as `strategy` and R as `R`, while
    script 22 expects `strategy_name` and `r_multiple`.

    Normalize those aliases here instead of modifying the frozen CSV.
    """
    if not COMMON_OOS_PATH.exists():
        raise FileNotFoundError(f"Common OOS file not found:\n{COMMON_OOS_PATH}")

    df = pd.read_csv(COMMON_OOS_PATH)

    # Strategy aliases used by the portfolio artifacts.
    strategy_column = next(
        (
            c
            for c in [
                "strategy_name",
                "strategy",
                "Strategy",
            ]
            if c in df.columns
        ),
        None,
    )

    # R aliases used throughout the project.
    r_column = next(
        (
            c
            for c in [
                "r_multiple",
                "R",
                "r",
                "net_R",
            ]
            if c in df.columns
        ),
        None,
    )

    if "entry_timestamp" not in df.columns:
        raise AssertionError("COMMON OOS: missing required column `entry_timestamp`.")

    if strategy_column is None:
        raise AssertionError(
            "COMMON OOS: could not identify strategy column. "
            f"Available columns: {list(df.columns)}"
        )

    if r_column is None:
        raise AssertionError(
            "COMMON OOS: could not identify R column. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["strategy_name"] = df[strategy_column].astype(str).str.upper().str.strip()

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    df["r_multiple"] = pd.to_numeric(
        df[r_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "strategy_name",
            "entry_timestamp",
            "r_multiple",
        ]
    ).copy()

    # The authoritative common-OOS artifact is already frozen.  Do not
    # re-filter or regenerate it here; only normalize its schema.
    df = df.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def load_paper_sizing() -> pd.DataFrame:
    if not PAPER_SIZING_PATH.exists():
        raise FileNotFoundError(f"Paper sizing file not found:\n{PAPER_SIZING_PATH}")

    df = pd.read_csv(PAPER_SIZING_PATH)

    strategy_column = next(
        (
            c
            for c in [
                "strategy_name",
                "strategy",
                "Strategy",
            ]
            if c in df.columns
        ),
        None,
    )

    if strategy_column is None:
        raise AssertionError(
            "PAPER SIZING: could not identify strategy column. "
            f"Available columns: {list(df.columns)}"
        )

    required = [
        "entry_timestamp",
        "risk_per_contract",
        "theoretical_quantity",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise AssertionError(
            "PAPER SIZING: missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    df = df.copy()

    df["strategy_name"] = df[strategy_column].astype(str).str.upper().str.strip()

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    df["risk_per_contract"] = pd.to_numeric(
        df["risk_per_contract"],
        errors="coerce",
    )

    df["theoretical_quantity"] = pd.to_numeric(
        df["theoretical_quantity"],
        errors="coerce",
    )

    if df["entry_timestamp"].isna().any():
        raise AssertionError("PAPER SIZING contains invalid entry_timestamp values.")

    if df["risk_per_contract"].isna().any():
        raise AssertionError("PAPER SIZING contains invalid risk_per_contract values.")

    return df


def build_xfa_policies(module):
    """
    Build the exact policy tuples expected by script 22.

    run_xfa_batch expects:
        (risk_name, risk_scenario_dict, payout_interval, payout_amount)
    """
    policies = []

    for risk_name, risk_scenario in module.RISK_SCENARIOS.items():
        for payout_interval in module.PAYOUT_INTERVALS:
            for payout_amount in module.PAYOUT_AMOUNTS:
                policies.append(
                    (
                        risk_name,
                        risk_scenario,
                        payout_interval,
                        float(payout_amount),
                    )
                )

    return policies


def summarize_xfa_results(
    raw_results: dict,
    n_simulations: int,
    max_trades: int,
) -> pd.DataFrame:
    """
    Convert the raw vectorized output from script 22 into the same
    economic summary schema used by the existing funded-account reports.
    """
    rows = []

    for key, records in raw_results.items():
        risk_name, payout_interval, payout_amount = key

        combined = {field: np.asarray(values) for field, values in records.items()}

        failed = combined["failed"].astype(bool)
        survival = ~failed
        survived_max = survival & (combined["trades"] >= max_trades)

        rows.append(
            {
                "risk": risk_name,
                "payout_interval": int(payout_interval),
                "payout_amount": float(payout_amount),
                "simulations": int(n_simulations),
                "survival_rate": float(survival.mean()),
                "failure_rate": float(failed.mean()),
                "survived_to_max_trades_rate": float(survived_max.mean()),
                "median_payouts": float(np.median(combined["payouts"])),
                "median_total_withdrawn": float(np.median(combined["total_withdrawn"])),
                "p95_total_withdrawn": float(
                    np.percentile(
                        combined["total_withdrawn"],
                        95,
                    )
                ),
                "median_final_balance": float(np.median(combined["final_balance"])),
                "median_net_value": float(
                    np.median(combined["final_balance"] + combined["total_withdrawn"])
                ),
                "median_max_DD": float(np.median(combined["max_drawdown"])),
                "p95_DD": float(
                    np.percentile(
                        combined["max_drawdown"],
                        5,
                    )
                ),
                "median_trades": float(np.median(combined["trades"])),
            }
        )

    return pd.DataFrame(rows)


def sample_common_calendar_dates(
    calendar_dates: np.ndarray,
    n_simulations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    dates = np.array(sorted(pd.Series(calendar_dates).dropna().unique()))

    if len(dates) == 0:
        raise AssertionError("No historical calendar dates available.")

    return rng.choice(
        dates,
        size=n_simulations,
        replace=True,
    )


def calendar_dates_to_sequence_day_indices(
    sequence,
    sampled_dates: np.ndarray,
) -> np.ndarray:
    """
    Convert sampled real calendar dates into the integer day indices
    required by HistoricalSequence.build_batch().
    """
    sequence_dates = pd.DatetimeIndex(pd.to_datetime(sequence.unique_dates)).normalize()

    sampled = pd.DatetimeIndex(pd.to_datetime(sampled_dates)).normalize()

    date_to_index = {date: idx for idx, date in enumerate(sequence_dates)}

    missing = [date for date in sampled if date not in date_to_index]

    if missing:
        raise AssertionError(
            "Sampled historical dates are missing from an "
            f"execution sequence. Missing {len(missing)} dates."
        )

    return np.asarray(
        [date_to_index[date] for date in sampled],
        dtype=np.int32,
    )


# ============================================================
# RESUMABLE MAIN REPLAY
# ============================================================

print("=" * 110)
print("ADAPTIVE ORB XFA REPLAY — CORRECTED + RESUMABLE")
print("=" * 110)
print("Existing corrected economic streams are reused.")
print("Strict/adaptive XFA summaries are cached after each completed run.")
print("A later failure will NOT rerun a completed 50,000-path XFA block.")
print()

# -----------------------------------------------------------------
# Cache files
# -----------------------------------------------------------------
CACHE_DIR = OUTPUT_DIR / "xfa_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_DATES_CACHE = CACHE_DIR / "sampled_calendar_dates.npy"
STRICT_START_DAYS_CACHE = CACHE_DIR / "strict_start_days.npy"
ADAPTIVE_START_DAYS_CACHE = CACHE_DIR / "adaptive_start_days.npy"
STRICT_SUMMARY_CACHE = CACHE_DIR / "strict_xfa_summary_cache.csv"
ADAPTIVE_SUMMARY_CACHE = CACHE_DIR / "adaptive_xfa_summary_cache.csv"

# These are the already-created economic streams from the previous
# corrected replay.  We intentionally do NOT rebuild them here.
STRICT_CORRECTED_OUTPUT = OUTPUT_DIR / "adaptive_orb_xfa_strict_corrected.csv"
ADAPTIVE_CORRECTED_OUTPUT = OUTPUT_DIR / "adaptive_orb_xfa_adaptive_corrected.csv"

# -----------------------------------------------------------------
# 0. LOAD FUNDED ENGINE
# -----------------------------------------------------------------
banner("0. LOAD FUNDED ENGINE")
module = load_funded_engine()
HistoricalSequence = module.HistoricalSequence
MAX_XFA_TRADES = int(module.MAX_XFA_TRADES)

print(f"Funded engine: {FUNDED_ENGINE_PATH}")
print(f"MAX_XFA_TRADES: {MAX_XFA_TRADES:,}")
print(f"Risk scenarios: {len(module.RISK_SCENARIOS)}")
print(f"Payout intervals: {module.PAYOUT_INTERVALS}")
print(f"Payout amounts: {module.PAYOUT_AMOUNTS}")

# -----------------------------------------------------------------
# 1. LOAD ALREADY-CORRECTED ECONOMIC STREAMS
# -----------------------------------------------------------------
banner("1. LOAD CORRECTED ECONOMIC TRADE STREAMS")

if not STRICT_CORRECTED_OUTPUT.exists():
    raise FileNotFoundError(
        "Strict corrected stream is missing. Run the original corrected "
        "32 once to create it:\n"
        f"{STRICT_CORRECTED_OUTPUT}"
    )
if not ADAPTIVE_CORRECTED_OUTPUT.exists():
    raise FileNotFoundError(
        "Adaptive corrected stream is missing. Run the original corrected "
        "32 once to create it:\n"
        f"{ADAPTIVE_CORRECTED_OUTPUT}"
    )

strict = pd.read_csv(STRICT_CORRECTED_OUTPUT)
adaptive = pd.read_csv(ADAPTIVE_CORRECTED_OUTPUT)

for name, df in [("STRICT", strict), ("ADAPTIVE", adaptive)]:
    assert_columns(
        df,
        ["strategy_name", "entry_timestamp", "xfa_r_multiple"],
        name,
    )
    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"], utc=True, errors="coerce"
    )
    df["xfa_r_multiple"] = pd.to_numeric(df["xfa_r_multiple"], errors="coerce")
    if df["entry_timestamp"].isna().any():
        raise AssertionError(f"{name}: invalid entry_timestamp values.")
    if df["xfa_r_multiple"].isna().any():
        raise AssertionError(f"{name}: invalid xfa_r_multiple values.")

strict = strict.sort_values(
    ["entry_timestamp", "strategy_name"], kind="mergesort"
).reset_index(drop=True)
adaptive = adaptive.sort_values(
    ["entry_timestamp", "strategy_name"], kind="mergesort"
).reset_index(drop=True)

if len(strict) != EXPECTED_STRICT_TRADES:
    raise AssertionError(
        f"Strict corrected stream expected {EXPECTED_STRICT_TRADES} rows, "
        f"got {len(strict)}"
    )
if len(adaptive) != EXPECTED_ADAPTIVE_TRADES:
    raise AssertionError(
        f"Adaptive corrected stream expected {EXPECTED_ADAPTIVE_TRADES} rows, "
        f"got {len(adaptive)}"
    )

print(f"Strict economic trades:   {len(strict):,}")
print(f"Adaptive economic trades: {len(adaptive):,}")
print("PASS — corrected streams reused; steps 1-9 are not recomputed.")

# -----------------------------------------------------------------
# 2. SAME HISTORICAL CALENDAR START DAYS
# -----------------------------------------------------------------
banner("2. SAME HISTORICAL CALENDAR START DAYS")


def trading_dates_ny(df: pd.DataFrame) -> np.ndarray:
    return np.array(
        sorted(
            pd.to_datetime(df["entry_timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.normalize()
            .dt.tz_localize(None)
            .unique()
        )
    )


strict_calendar_days = trading_dates_ny(strict)
adaptive_calendar_days = trading_dates_ny(adaptive)
common_calendar_days = np.intersect1d(strict_calendar_days, adaptive_calendar_days)

if len(common_calendar_days) == 0:
    raise AssertionError("No common executable calendar days.")

print(f"Strict executable days:  {len(strict_calendar_days):,}")
print(f"Adaptive executable days:{len(adaptive_calendar_days):,}")
print(f"Common executable days: {len(common_calendar_days):,}")

if SAMPLED_DATES_CACHE.exists():
    sampled_calendar_dates = np.load(SAMPLED_DATES_CACHE, allow_pickle=False)
    if len(sampled_calendar_dates) != SIMULATIONS:
        raise AssertionError(
            "Cached sampled calendar dates have the wrong simulation count."
        )
    print(f"Loaded cached start dates: {len(sampled_calendar_dates):,}")
else:
    sampled_calendar_dates = sample_common_calendar_dates(
        common_calendar_days, SIMULATIONS, RANDOM_SEED
    )
    np.save(SAMPLED_DATES_CACHE, sampled_calendar_dates)
    print(f"Created and cached start dates: {len(sampled_calendar_dates):,}")

print("PASS — identical sampled calendar start dates reused.")

# -----------------------------------------------------------------
# 3. BUILD HISTORICAL SEQUENCES
# -----------------------------------------------------------------
banner("3. BUILD HISTORICAL SEQUENCES")


def add_date_ny_for_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_ny"] = (
        pd.to_datetime(out["entry_timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    return out


# The corrected streams already contain the final economic R stream.
strict_engine_df = add_date_ny_for_engine(
    strict[["strategy_name", "entry_timestamp", "xfa_r_multiple"]].rename(
        columns={"xfa_r_multiple": "r_multiple"}
    )
)
adaptive_engine_df = add_date_ny_for_engine(
    adaptive[["strategy_name", "entry_timestamp", "xfa_r_multiple"]].rename(
        columns={"xfa_r_multiple": "r_multiple"}
    )
)

strict_sequence = HistoricalSequence("STRICT_025", strict_engine_df)
adaptive_sequence = HistoricalSequence("ADAPTIVE_ORB_060", adaptive_engine_df)

print(f"Strict trades:   {strict_sequence.n_trades:,}")
print(f"Strict days:     {strict_sequence.n_days:,}")
print(f"Adaptive trades: {adaptive_sequence.n_trades:,}")
print(f"Adaptive days:   {adaptive_sequence.n_days:,}")

# -----------------------------------------------------------------
# 4. BUILD / REUSE START-DAY INDICES
# -----------------------------------------------------------------
banner("4. BUILD / REUSE START-DAY INDICES")

if STRICT_START_DAYS_CACHE.exists() and ADAPTIVE_START_DAYS_CACHE.exists():
    strict_start_days = np.load(STRICT_START_DAYS_CACHE, allow_pickle=False)
    adaptive_start_days = np.load(ADAPTIVE_START_DAYS_CACHE, allow_pickle=False)
    if len(strict_start_days) != SIMULATIONS or len(adaptive_start_days) != SIMULATIONS:
        raise AssertionError("Cached start-day indices have the wrong size.")
    print("Loaded cached strict/adaptive start-day indices.")
else:
    strict_start_days = calendar_dates_to_sequence_day_indices(
        strict_sequence, sampled_calendar_dates
    )
    adaptive_start_days = calendar_dates_to_sequence_day_indices(
        adaptive_sequence, sampled_calendar_dates
    )
    np.save(STRICT_START_DAYS_CACHE, strict_start_days)
    np.save(ADAPTIVE_START_DAYS_CACHE, adaptive_start_days)
    print("Created and cached strict/adaptive start-day indices.")


# Explicitly verify that the two sampled sequences correspond to the same
# real New York calendar dates, not merely the same integer positions.
def sequence_day_date(sequence, day_indices):
    dates = pd.DatetimeIndex(pd.to_datetime(sequence.unique_dates)).normalize()
    return dates[np.asarray(day_indices, dtype=np.int64)]


if not np.array_equal(
    sequence_day_date(strict_sequence, strict_start_days),
    sequence_day_date(adaptive_sequence, adaptive_start_days),
):
    raise AssertionError(
        "Strict/adaptive cached start-day indices do not represent identical dates."
    )

print("PASS — strict/adaptive start-day indices map to identical NY dates.")

# -----------------------------------------------------------------
# 5. BUILD ECONOMIC TRADE PATHS
# -----------------------------------------------------------------
banner("5. BUILD ECONOMIC TRADE PATHS")

strict_path_returns, strict_day_ids = strict_sequence.build_batch(
    strict_start_days, MAX_XFA_TRADES
)
adaptive_path_returns, adaptive_day_ids = adaptive_sequence.build_batch(
    adaptive_start_days, MAX_XFA_TRADES
)

expected_shape = (SIMULATIONS, MAX_XFA_TRADES)
if strict_path_returns.shape != expected_shape:
    raise AssertionError(f"Strict path shape mismatch: {strict_path_returns.shape}")
if adaptive_path_returns.shape != expected_shape:
    raise AssertionError(f"Adaptive path shape mismatch: {adaptive_path_returns.shape}")

print(f"Strict path shape:   {strict_path_returns.shape}")
print(f"Adaptive path shape: {adaptive_path_returns.shape}")
print("PASS — economic paths rebuilt from cached start dates.")

# -----------------------------------------------------------------
# 6. BUILD POLICIES
# -----------------------------------------------------------------
banner("6. BUILD XFA POLICIES")
policies = build_xfa_policies(module)
expected_policy_count = (
    len(module.RISK_SCENARIOS)
    * len(module.PAYOUT_INTERVALS)
    * len(module.PAYOUT_AMOUNTS)
)
if len(policies) != expected_policy_count:
    raise AssertionError(
        f"Expected {expected_policy_count} XFA policies, got {len(policies)}"
    )
print(f"XFA policies: {len(policies):,}")

# -----------------------------------------------------------------
# 7. STRICT XFA — CACHE AFTER COMPLETION
# -----------------------------------------------------------------
banner("7. STRICT XFA")

if STRICT_SUMMARY_CACHE.exists():
    strict_results = pd.read_csv(STRICT_SUMMARY_CACHE)
    if len(strict_results) != expected_policy_count:
        raise AssertionError(
            "Strict XFA cache exists but has the wrong number of policies."
        )
    print(f"Loaded cached strict XFA summary: {STRICT_SUMMARY_CACHE}")
    print("SKIP — strict 50,000-path XFA simulation already completed.")
else:
    print("Running strict XFA — this is the heavy step...")
    strict_raw = module.run_xfa_batch(
        strict_path_returns,
        strict_day_ids,
        policies,
        MAX_XFA_TRADES,
    )
    strict_results = summarize_xfa_results(strict_raw, SIMULATIONS, MAX_XFA_TRADES)
    if len(strict_results) != expected_policy_count:
        raise AssertionError("Strict XFA returned the wrong number of policies.")
    strict_results.to_csv(STRICT_SUMMARY_CACHE, index=False)
    print(f"Strict XFA cache written: {STRICT_SUMMARY_CACHE}")
    print("PASS — strict XFA complete and cached.")

# -----------------------------------------------------------------
# 8. ADAPTIVE XFA — CACHE AFTER COMPLETION
# -----------------------------------------------------------------
banner("8. ADAPTIVE XFA")

if ADAPTIVE_SUMMARY_CACHE.exists():
    adaptive_results = pd.read_csv(ADAPTIVE_SUMMARY_CACHE)
    if len(adaptive_results) != expected_policy_count:
        raise AssertionError(
            "Adaptive XFA cache exists but has the wrong number of policies."
        )
    print(f"Loaded cached adaptive XFA summary: {ADAPTIVE_SUMMARY_CACHE}")
    print("SKIP — adaptive 50,000-path XFA simulation already completed.")
else:
    print("Running adaptive XFA — this is the heavy step...")
    adaptive_raw = module.run_xfa_batch(
        adaptive_path_returns,
        adaptive_day_ids,
        policies,
        MAX_XFA_TRADES,
    )
    adaptive_results = summarize_xfa_results(adaptive_raw, SIMULATIONS, MAX_XFA_TRADES)
    if len(adaptive_results) != expected_policy_count:
        raise AssertionError("Adaptive XFA returned the wrong number of policies.")
    adaptive_results.to_csv(ADAPTIVE_SUMMARY_CACHE, index=False)
    print(f"Adaptive XFA cache written: {ADAPTIVE_SUMMARY_CACHE}")
    print("PASS — adaptive XFA complete and cached.")

# -----------------------------------------------------------------
# 9. MERGE + FINAL REPORT
# -----------------------------------------------------------------
banner("9. MERGE STRICT + ADAPTIVE POLICY RESULTS")

KEYS = ["risk", "payout_interval", "payout_amount"]

strict_results = strict_results.rename(
    columns={c: f"{c}_strict" for c in strict_results.columns if c not in KEYS}
)
adaptive_results = adaptive_results.rename(
    columns={c: f"{c}_adaptive" for c in adaptive_results.columns if c not in KEYS}
)

comparison = strict_results.merge(
    adaptive_results,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

if len(comparison) != expected_policy_count:
    raise AssertionError(
        f"Expected {expected_policy_count} merged policies, got {len(comparison)}"
    )

comparison_025 = comparison[comparison["risk"] == "0.25%"].copy()
comparison_025["delta_survival"] = (
    comparison_025["survival_rate_adaptive"] - comparison_025["survival_rate_strict"]
)
comparison_025["delta_median_net_value"] = (
    comparison_025["median_net_value_adaptive"]
    - comparison_025["median_net_value_strict"]
)
comparison_025["delta_median_DD"] = (
    comparison_025["median_max_DD_adaptive"] - comparison_025["median_max_DD_strict"]
)
comparison_025["delta_median_trades"] = (
    comparison_025["median_trades_adaptive"] - comparison_025["median_trades_strict"]
)

print("0.25% policy comparison:")
display_columns = [
    "risk",
    "payout_interval",
    "payout_amount",
    "survival_rate_strict",
    "survival_rate_adaptive",
    "delta_survival",
    "median_net_value_strict",
    "median_net_value_adaptive",
    "delta_median_net_value",
    "median_max_DD_strict",
    "median_max_DD_adaptive",
    "delta_median_DD",
    "median_trades_strict",
    "median_trades_adaptive",
]
print(comparison_025[display_columns].to_string(index=False))

# -----------------------------------------------------------------
# 10. EXPORT FINAL OUTPUTS
# -----------------------------------------------------------------
banner("10. EXPORT XFA RESULTS")

COMPARISON_OUTPUT = OUTPUT_DIR / "adaptive_orb_xfa_results_corrected.csv"
STRICT_SUMMARY_OUTPUT = OUTPUT_DIR / "adaptive_orb_xfa_strict_corrected_summary.csv"
ADAPTIVE_SUMMARY_OUTPUT = OUTPUT_DIR / "adaptive_orb_xfa_adaptive_corrected_summary.csv"

comparison_025.to_csv(COMPARISON_OUTPUT, index=False)
strict_results.to_csv(STRICT_SUMMARY_OUTPUT, index=False)
adaptive_results.to_csv(ADAPTIVE_SUMMARY_OUTPUT, index=False)

print(f"Comparison: {COMPARISON_OUTPUT}")
print(f"Strict summary: {STRICT_SUMMARY_OUTPUT}")
print(f"Adaptive summary: {ADAPTIVE_SUMMARY_OUTPUT}")

# -----------------------------------------------------------------
# 11. FINAL AUDIT
# -----------------------------------------------------------------
banner("11. FINAL AUDIT")

print(f"Strict economic trades:    {len(strict):,}")
print(f"Adaptive economic trades:  {len(adaptive):,}")
print(f"Historical start samples:  {len(sampled_calendar_dates):,}")
print(f"Strict path shape:         {strict_path_returns.shape}")
print(f"Adaptive path shape:       {adaptive_path_returns.shape}")
print(f"XFA policy count:          {len(policies):,}")
print(f"Strict summary rows:       {len(strict_results):,}")
print(f"Adaptive summary rows:     {len(adaptive_results):,}")

assert len(strict) == EXPECTED_STRICT_TRADES
assert len(adaptive) == EXPECTED_ADAPTIVE_TRADES
assert len(sampled_calendar_dates) == SIMULATIONS
assert strict_path_returns.shape == expected_shape
assert adaptive_path_returns.shape == expected_shape
assert len(policies) == expected_policy_count
assert len(strict_results) == expected_policy_count
assert len(adaptive_results) == expected_policy_count
assert "xfa_r_multiple" in strict.columns
assert "xfa_r_multiple" in adaptive.columns

print()
print("Corrected economic streams were reused — no rebuilding of steps 1-9.")
print("Strict XFA is cached independently of adaptive XFA.")
print("Adaptive XFA is cached independently of strict XFA.")
print("If a later reporting step fails, rerunning this script will reuse both caches.")
print("Historical trade order is preserved.")
print("Individual trades are NOT shuffled.")
print("Strict and adaptive use identical sampled NY calendar dates.")
print("net_value = final_balance + total_withdrawn.")
print()
print("=" * 110)
print("ADAPTIVE ORB XFA REPLAY — CORRECTED + RESUMABLE: PASS")
print("=" * 110)
