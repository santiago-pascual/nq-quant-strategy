from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_project_root() -> Path:
    """
    Find repository root robustly.
    """
    for candidate in [SCRIPT_PATH.parent] + list(SCRIPT_PATH.parents):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate

    raise RuntimeError(
        "Could not identify project root. "
        "Expected a directory containing both 'src' and 'data'."
    )


PROJECT_ROOT = find_project_root()

MR_TRADES_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv"
)

S2R_TRADES_FILE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio_3_strategy"


EXPECTED_COUNTS = {
    "MRL1": 483,
    "S2R": 537,
    "MRS2": 1052,
}


# =============================================================================
# HELPERS
# =============================================================================


def banner(text: str) -> None:
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def fmt(x, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"

    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"

    return str(x)


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:

    mapping = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in mapping:
            return mapping[key]

    if required:
        raise ValueError(
            "Could not find required column.\n"
            f"Tried: {list(candidates)}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def parse_timestamp_utc(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Robust timestamp parser for mixed timezone representations.

    Each value is parsed independently so pandas never attempts to
    infer one timezone for the entire column.

    Timezone-aware values:
        -> converted directly to UTC

    Naive values:
        -> interpreted as America/New_York
        -> converted to UTC
    """

    def parse_one(value):
        if pd.isna(value):
            return pd.NaT

        # Parse one value at a time.
        ts = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(ts):
            return pd.NaT

        # Python/pandas Timestamp timezone handling.
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC")

        # Naive timestamp -> project market timezone.
        return ts.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="shift_forward",
        ).tz_convert("UTC")

    parsed = series.map(parse_one)

    bad = int(parsed.isna().sum())

    if bad:
        raise ValueError(
            f"Column '{column_name}' contains {bad} invalid/unparseable timestamps."
        )

    return pd.to_datetime(
        parsed,
        utc=True,
    )


def normalize_timestamp(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:

    out = df.copy()

    out[target_col] = parse_timestamp_utc(
        out[source_col],
        source_col,
    )

    return out


# =============================================================================
# LOAD MR
# =============================================================================


def load_mr_trades() -> pd.DataFrame:

    banner("LOADING MR TRADES")

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MR file:      {MR_TRADES_FILE}")

    if not MR_TRADES_FILE.exists():
        raise FileNotFoundError(f"MR reproduction file not found:\n{MR_TRADES_FILE}")

    df = pd.read_csv(MR_TRADES_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    strategy_col = find_column(
        df,
        [
            "strategy",
            "strategy_id",
            "candidate",
            "strategy_name",
            "model",
        ],
    )

    print(f"Strategy column: {strategy_col}")

    df["strategy"] = df[strategy_col].astype(str).str.strip().str.upper()

    strategy_map = {
        "MRL1": "MRL1",
        "MRS2": "MRS2",
        "MRL1_LONG": "MRL1",
        "MRS2_SHORT": "MRS2",
    }

    df["strategy"] = df["strategy"].map(lambda x: strategy_map.get(x, x))

    print("\nStrategy counts before filtering:")
    print(df["strategy"].value_counts(dropna=False).to_string())

    unknown = sorted(set(df["strategy"].dropna().unique()) - {"MRL1", "MRS2"})

    if unknown:
        raise ValueError(f"Unexpected strategy labels found:\n{unknown}")

    df = df[df["strategy"].isin(["MRL1", "MRS2"])].copy()

    # Entry timestamp.
    entry_col = find_column(
        df,
        [
            "entry_timestamp",
            "entry_time",
            "timestamp",
            "datetime",
        ],
    )

    df = normalize_timestamp(
        df,
        entry_col,
        "entry_timestamp",
    )

    # Exit timestamp if available.
    exit_col = find_column(
        df,
        [
            "exit_timestamp",
            "exit_time",
        ],
        required=False,
    )

    if exit_col:
        df = normalize_timestamp(
            df,
            exit_col,
            "exit_timestamp",
        )
    else:
        df["exit_timestamp"] = pd.NaT

    # R.
    r_col = find_column(
        df,
        [
            "net_R",
            "net_r",
            "r_multiple",
            "R",
            "return_R",
            "r",
        ],
    )

    df["net_R"] = pd.to_numeric(
        df[r_col],
        errors="coerce",
    )

    if df["net_R"].isna().any():
        raise ValueError("Invalid net_R values found in MR trades.")

    # Frozen counts.
    counts = df["strategy"].value_counts().to_dict()

    for strategy in ["MRL1", "MRS2"]:
        expected = EXPECTED_COUNTS[strategy]
        actual = int(counts.get(strategy, 0))

        if actual != expected:
            raise ValueError(f"{strategy}: expected {expected}, got {actual}.")

    print("\nFrozen MR counts:")
    print(f"  MRL1: {counts.get('MRL1', 0):,}")
    print(f"  MRS2: {counts.get('MRS2', 0):,}")

    return df


# =============================================================================
# LOAD S2R
# =============================================================================


def load_s2r_trades() -> pd.DataFrame:

    banner("LOADING S2R TRADES")

    print(f"S2R file: {S2R_TRADES_FILE}")

    if not S2R_TRADES_FILE.exists():
        raise FileNotFoundError(
            f"S2R authoritative reproduction file not found:\n{S2R_TRADES_FILE}"
        )

    df = pd.read_csv(S2R_TRADES_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    df["strategy"] = "S2R"

    # -------------------------------------------------------------------------
    # Entry timestamp
    # -------------------------------------------------------------------------

    entry_col = find_column(
        df,
        [
            "entry_timestamp",
            "entry_time",
            "timestamp",
            "datetime",
        ],
    )

    print(f"S2R entry timestamp column: {entry_col}")

    df = normalize_timestamp(
        df,
        entry_col,
        "entry_timestamp",
    )

    # -------------------------------------------------------------------------
    # Exit timestamp
    # -------------------------------------------------------------------------

    exit_col = find_column(
        df,
        [
            "exit_timestamp",
            "exit_time",
        ],
        required=False,
    )

    if exit_col:
        df = normalize_timestamp(
            df,
            exit_col,
            "exit_timestamp",
        )
    else:
        df["exit_timestamp"] = pd.NaT

    # -------------------------------------------------------------------------
    # R
    # -------------------------------------------------------------------------

    r_col = find_column(
        df,
        [
            "net_R",
            "net_r",
            "r_multiple",
            "R",
            "return_R",
        ],
    )

    df["net_R"] = pd.to_numeric(
        df[r_col],
        errors="coerce",
    )

    if df["net_R"].isna().any():
        raise ValueError("Invalid net_R values found in S2R trades.")

    # -------------------------------------------------------------------------
    # Frozen count
    # -------------------------------------------------------------------------

    actual = len(df)

    if actual != EXPECTED_COUNTS["S2R"]:
        raise ValueError(
            f"S2R: expected {EXPECTED_COUNTS['S2R']} trades, got {actual}."
        )

    print(f"Frozen S2R count: {actual:,}")

    return df


# =============================================================================
# COMBINE
# =============================================================================


def combine_trades() -> pd.DataFrame:

    banner("COMBINING ALL 3 STRATEGIES")

    mr = load_mr_trades()
    s2r = load_s2r_trades()

    portfolio = pd.concat(
        [mr, s2r],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    # Everything should now be UTC.
    portfolio["entry_timestamp"] = pd.to_datetime(
        portfolio["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    portfolio["exit_timestamp"] = pd.to_datetime(
        portfolio["exit_timestamp"],
        utc=True,
        errors="coerce",
    )

    portfolio["net_R"] = pd.to_numeric(
        portfolio["net_R"],
        errors="raise",
    )

    # Chronological ordering.
    portfolio = portfolio.sort_values(
        [
            "entry_timestamp",
            "strategy",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    portfolio["trade_id"] = np.arange(len(portfolio)) + 1

    # Calendar fields.
    portfolio["date"] = (
        portfolio["entry_timestamp"].dt.tz_convert("America/New_York").dt.date
    )

    portfolio["year"] = (
        portfolio["entry_timestamp"].dt.tz_convert("America/New_York").dt.year
    )

    portfolio["month"] = (
        portfolio["entry_timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.to_period("M")
        .astype(str)
    )

    portfolio["weekday"] = (
        portfolio["entry_timestamp"].dt.tz_convert("America/New_York").dt.day_name()
    )

    portfolio["hour"] = (
        portfolio["entry_timestamp"].dt.tz_convert("America/New_York").dt.hour
    )

    # Chronological equity.
    portfolio["equity_R"] = portfolio["net_R"].cumsum()

    portfolio["equity_peak_R"] = portfolio["equity_R"].cummax()

    portfolio["drawdown_R"] = portfolio["equity_R"] - portfolio["equity_peak_R"]

    portfolio["is_win"] = portfolio["net_R"] > 0

    portfolio["is_loss"] = portfolio["net_R"] < 0

    portfolio["is_flat"] = portfolio["net_R"] == 0

    # -------------------------------------------------------------------------
    # Duplicate audit
    # -------------------------------------------------------------------------

    duplicate_identity = portfolio.duplicated(
        subset=[
            "entry_timestamp",
            "strategy",
        ],
        keep=False,
    )

    duplicate_timestamp = portfolio.duplicated(
        subset=["entry_timestamp"],
        keep=False,
    )

    n_duplicate_identity = int(duplicate_identity.sum())

    n_duplicate_timestamp = int(duplicate_timestamp.sum())

    print(f"Total trades: {len(portfolio):,}")

    print(f"Duplicate (entry_timestamp, strategy) rows: {n_duplicate_identity:,}")

    print(
        "Rows participating in simultaneous "
        "entry timestamps: "
        f"{n_duplicate_timestamp:,}"
    )

    if n_duplicate_identity:
        raise ValueError("Duplicate trade identity detected.")

    return portfolio


# =============================================================================
# METRICS
# =============================================================================


def profit_factor(values: pd.Series) -> float:

    gross_profit = values[values > 0].sum()

    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        return np.inf

    return float(gross_profit / gross_loss)


def longest_streak(
    values: pd.Series,
    positive: bool,
) -> int:

    best = 0
    current = 0

    for value in values:
        condition = value > 0 if positive else value < 0

        if condition:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return best


def calculate_trade_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    values = df["net_R"].astype(float)

    wins = values[values > 0]

    losses = values[values < 0]

    metrics = {
        "trades": len(values),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "median_R": float(values.median()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": float(df["drawdown_R"].min()),
        "avg_win_R": (float(wins.mean()) if len(wins) else np.nan),
        "avg_loss_R": (float(losses.mean()) if len(losses) else np.nan),
        "payoff_ratio": (
            float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses)
            else np.nan
        ),
        "longest_win_streak": longest_streak(values, True),
        "longest_loss_streak": longest_streak(values, False),
    }

    return pd.DataFrame([metrics])


# =============================================================================
# DAILY
# =============================================================================


def build_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:

    daily = df.groupby("date", sort=True)["net_R"].sum().rename("daily_R").to_frame()

    daily.index = pd.to_datetime(daily.index)

    daily["equity_R"] = daily["daily_R"].cumsum()

    daily["peak_R"] = daily["equity_R"].cummax()

    daily["drawdown_R"] = daily["equity_R"] - daily["peak_R"]

    daily["year"] = daily.index.year

    daily["month"] = daily.index.to_period("M").astype(str)

    # Strategy daily contribution.
    pivot = df.pivot_table(
        index="date",
        columns="strategy",
        values="net_R",
        aggfunc="sum",
        fill_value=0.0,
    )

    pivot.index = pd.to_datetime(pivot.index)

    daily = daily.join(
        pivot.add_prefix("R_"),
        how="left",
    )

    return daily.reset_index(names="date")


def add_daily_risk_metrics(
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:

    values = daily["daily_R"].astype(float)

    if len(values) > 1:
        std = float(values.std(ddof=1))

        sharpe = float(values.mean() / std * np.sqrt(252)) if std > 0 else np.nan

        downside = values[values < 0]

        if len(downside) > 1:
            downside_std = float(downside.std(ddof=1))

            sortino = (
                float(values.mean() / downside_std * np.sqrt(252))
                if downside_std > 0
                else np.nan
            )

        else:
            sortino = np.nan

    else:
        std = np.nan
        sharpe = np.nan
        sortino = np.nan

    out = metrics.copy()

    out["daily_mean_R"] = float(values.mean())

    out["daily_std_R"] = std

    out["annualized_sharpe"] = sharpe

    out["annualized_sortino"] = sortino

    out["active_days"] = int((values != 0).sum())

    out["calendar_days"] = int(len(values))

    return out


# =============================================================================
# PERIOD METRICS
# =============================================================================


def period_metrics(
    df: pd.DataFrame,
    period_col: str,
) -> pd.DataFrame:

    rows = []

    for period, group in df.groupby(
        period_col,
        sort=True,
    ):
        values = group["net_R"].astype(float)

        equity = values.cumsum()
        peak = equity.cummax()
        dd = equity - peak

        rows.append(
            {
                period_col: period,
                "trades": len(group),
                "total_R": float(values.sum()),
                "mean_R": float(values.mean()),
                "median_R": float(values.median()),
                "win_rate": float((values > 0).mean()),
                "profit_factor": profit_factor(values),
                "max_drawdown_R": float(dd.min()),
                "avg_win_R": (
                    float(values[values > 0].mean()) if (values > 0).any() else np.nan
                ),
                "avg_loss_R": (
                    float(values[values < 0].mean()) if (values < 0).any() else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# STRATEGY CONTRIBUTION
# =============================================================================


def strategy_contribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total_R = df["net_R"].sum()

    for strategy, group in df.groupby(
        "strategy",
        sort=True,
    ):
        values = group["net_R"].astype(float)

        equity = values.cumsum()
        peak = equity.cummax()
        dd = equity - peak

        rows.append(
            {
                "strategy": strategy,
                "trades": len(group),
                "total_R": float(values.sum()),
                "mean_R": float(values.mean()),
                "median_R": float(values.median()),
                "win_rate": float((values > 0).mean()),
                "profit_factor": profit_factor(values),
                "max_drawdown_R": float(dd.min()),
                "share_of_total_R": (
                    float(values.sum() / total_R) if total_R != 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# OVERLAP
# =============================================================================


def analyze_overlap(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df.groupby("entry_timestamp")
        .agg(
            strategies=(
                "strategy",
                lambda x: ",".join(sorted(set(x))),
            ),
            n_strategies=(
                "strategy",
                "nunique",
            ),
            total_R=(
                "net_R",
                "sum",
            ),
        )
        .reset_index()
    )

    overlap = grouped[grouped["n_strategies"] >= 2].copy()

    return overlap.sort_values("entry_timestamp").reset_index(drop=True)


def strategy_daily_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:

    pivot = df.pivot_table(
        index="date",
        columns="strategy",
        values="net_R",
        aggfunc="sum",
        fill_value=0.0,
    )

    return pivot.corr()


# =============================================================================
# REPORT
# =============================================================================


def print_report(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    contribution: pd.DataFrame,
    overlap: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:

    banner("FINAL 3-STRATEGY PORTFOLIO REPORT")

    m = metrics.iloc[0]

    print("\nPORTFOLIO")
    print("-" * 110)

    fields = [
        ("Trades", "trades", 0),
        ("Total R", "total_R", 4),
        (
            "Mean / Expectancy R",
            "mean_R",
            6,
        ),
        ("Median R", "median_R", 4),
        ("Win rate", "win_rate", 4),
        (
            "Profit factor",
            "profit_factor",
            4,
        ),
        (
            "Max drawdown R",
            "max_drawdown_R",
            4,
        ),
        (
            "Average win R",
            "avg_win_R",
            4,
        ),
        (
            "Average loss R",
            "avg_loss_R",
            4,
        ),
        (
            "Payoff ratio",
            "payoff_ratio",
            4,
        ),
        (
            "Longest win streak",
            "longest_win_streak",
            0,
        ),
        (
            "Longest loss streak",
            "longest_loss_streak",
            0,
        ),
        (
            "Active days",
            "active_days",
            0,
        ),
        (
            "Annualized Sharpe",
            "annualized_sharpe",
            4,
        ),
        (
            "Annualized Sortino",
            "annualized_sortino",
            4,
        ),
    ]

    for label, key, digits in fields:
        print(f"{label:<28}: {fmt(m[key], digits)}")

    print("\nSTRATEGY CONTRIBUTION")
    print("-" * 110)

    print(
        contribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nYEARLY")
    print("-" * 110)

    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nMONTHLY")
    print("-" * 110)

    print(
        monthly.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSTRATEGY DAILY CORRELATION")
    print("-" * 110)

    print(
        correlation.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSIMULTANEOUS ENTRY OVERLAP")
    print("-" * 110)

    print(f"Overlap timestamps: {len(overlap):,}")

    if not overlap.empty:
        print(overlap.head(25).to_string(index=False))

    print("\nDATE RANGE")
    print("-" * 110)

    print(f"First trade: {df['entry_timestamp'].min()}")

    print(f"Last trade:  {df['entry_timestamp'].max()}")

    print("\nDAILY EQUITY")
    print("-" * 110)

    print(f"Active days: {len(daily):,}")

    print(f"Best day R: {daily['daily_R'].max():.4f}")

    print(f"Worst day R: {daily['daily_R'].min():.4f}")

    print(f"Daily max drawdown R: {daily['drawdown_R'].min():.4f}")


# =============================================================================
# SAVE
# =============================================================================


def save_outputs(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    contribution: pd.DataFrame,
    overlap: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "trades": OUTPUT_DIR / "portfolio_3_strategy_trades.csv",
        "daily": OUTPUT_DIR / "portfolio_3_strategy_daily.csv",
        "metrics": OUTPUT_DIR / "portfolio_3_strategy_metrics.csv",
        "yearly": OUTPUT_DIR / "portfolio_3_strategy_yearly.csv",
        "monthly": OUTPUT_DIR / "portfolio_3_strategy_monthly.csv",
        "contribution": OUTPUT_DIR / "portfolio_3_strategy_contribution.csv",
        "overlap": OUTPUT_DIR / "portfolio_3_strategy_overlap.csv",
        "correlation": OUTPUT_DIR / "portfolio_3_strategy_daily_correlation.csv",
    }

    df.to_csv(
        outputs["trades"],
        index=False,
    )

    daily.to_csv(
        outputs["daily"],
        index=False,
    )

    metrics.to_csv(
        outputs["metrics"],
        index=False,
    )

    yearly.to_csv(
        outputs["yearly"],
        index=False,
    )

    monthly.to_csv(
        outputs["monthly"],
        index=False,
    )

    contribution.to_csv(
        outputs["contribution"],
        index=False,
    )

    overlap.to_csv(
        outputs["overlap"],
        index=False,
    )

    correlation.to_csv(
        outputs["correlation"],
    )

    print()
    print("=" * 110)
    print("OUTPUT FILES")
    print("=" * 110)

    for name, path in outputs.items():
        print(f"{name:<16}: {path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    banner("3-STRATEGY PORTFOLIO ANALYSIS")

    print(f"Script:       {SCRIPT_PATH}")

    print(f"Project root: {PROJECT_ROOT}")

    portfolio = combine_trades()

    # -------------------------------------------------------------------------
    # Final count audit
    # -------------------------------------------------------------------------

    counts = portfolio["strategy"].value_counts().to_dict()

    print("\nFINAL TRADE COUNT AUDIT")
    print("-" * 110)

    expected_total = sum(EXPECTED_COUNTS.values())

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(
            counts.get(
                strategy,
                0,
            )
        )

        status = "PASS" if actual == expected else "FAIL"

        print(f"{strategy:<8} expected={expected:>5,} actual={actual:>5,} {status}")

    print(f"{'TOTAL':<8} expected={expected_total:>5,} actual={len(portfolio):>5,}")

    if len(portfolio) != expected_total:
        raise ValueError(
            f"Portfolio total mismatch: "
            f"expected {expected_total}, "
            f"got {len(portfolio)}."
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = calculate_trade_metrics(portfolio)

    daily = build_daily(portfolio)

    metrics = add_daily_risk_metrics(
        metrics,
        daily,
    )

    yearly = period_metrics(
        portfolio,
        "year",
    )

    monthly = period_metrics(
        portfolio,
        "month",
    )

    contribution = strategy_contribution(portfolio)

    overlap = analyze_overlap(portfolio)

    correlation = strategy_daily_correlation(portfolio)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    print_report(
        portfolio,
        metrics,
        daily,
        yearly,
        monthly,
        contribution,
        overlap,
        correlation,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_outputs(
        portfolio,
        daily,
        metrics,
        yearly,
        monthly,
        contribution,
        overlap,
        correlation,
    )

    banner("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
