from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt


# ======================================================================================
# CONFIGURATION — FROZEN S2R
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s27_full_strategy_trades.csv"
)

OUTPUT_DIR = BASE_DIR / "src" / "research" / "results" / "s2_extended"

OOS_WINDOWS = list(range(12, 23))

STRATEGY_R_COL = "_strategy_R"
WINDOW_COL = "_window_numeric"

# Frozen S2R model — informational only.
MAE_THRESHOLD = 0.70
RECOVERY_LEVEL = 0.20
RECOVERY_DEADLINE = 6


# ======================================================================================
# HELPERS
# ======================================================================================


def section(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_window(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return int(value)
        return np.nan

    text = str(value).strip()

    try:
        return int(float(text))
    except ValueError:
        return np.nan


def max_drawdown(r_values: pd.Series) -> float:
    values = pd.to_numeric(r_values, errors="coerce").dropna().to_numpy()

    if len(values) == 0:
        return np.nan

    equity = np.cumsum(values)
    running_max = np.maximum.accumulate(np.concatenate(([0.0], equity)))

    drawdowns = equity - running_max[1:]

    return float(drawdowns.min())


def profit_factor(r_values: pd.Series) -> float:
    values = pd.to_numeric(r_values, errors="coerce").dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = -values[values < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return 0.0

    return float(gross_profit / gross_loss)


def metrics(df: pd.DataFrame, r_col: str = STRATEGY_R_COL) -> dict:
    values = pd.to_numeric(df[r_col], errors="coerce").dropna()

    if len(values) == 0:
        return {
            "trades": 0,
            "total_R": np.nan,
            "mean_R": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_R": np.nan,
            "gross_profit_R": 0.0,
            "gross_loss_R": 0.0,
        }

    wins = values[values > 0]
    losses = values[values < 0]

    return {
        "trades": int(len(values)),
        "total_R": float(values.sum()),
        "mean_R": float(values.mean()),
        "win_rate": float((values > 0).mean()),
        "profit_factor": profit_factor(values),
        "max_drawdown_R": max_drawdown(values),
        "gross_profit_R": float(wins.sum()),
        "gross_loss_R": float(-losses.sum()),
    }


# ======================================================================================
# LOAD
# ======================================================================================


def load_strategy() -> pd.DataFrame:
    section("LOADING FROZEN S2R")

    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"S2R strategy file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    if STRATEGY_R_COL not in df.columns:
        raise RuntimeError(
            f"Missing strategy R column: {STRATEGY_R_COL}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    if WINDOW_COL not in df.columns:
        raise RuntimeError(f"Missing window column: {WINDOW_COL}")

    df = df.copy()

    df[WINDOW_COL] = df[WINDOW_COL].apply(normalize_window)
    df[STRATEGY_R_COL] = pd.to_numeric(
        df[STRATEGY_R_COL],
        errors="coerce",
    )

    if df[STRATEGY_R_COL].isna().any():
        raise RuntimeError("S2R contains non-finite / missing strategy R values.")

    if not np.isfinite(df[STRATEGY_R_COL].to_numpy()).all():
        raise RuntimeError("S2R contains non-finite strategy R values.")

    return df


# ======================================================================================
# OOS FILTER
# ======================================================================================


def build_oos(df: pd.DataFrame) -> pd.DataFrame:
    section("BUILDING OOS SAMPLE")

    oos = df[df[WINDOW_COL].isin(OOS_WINDOWS)].copy()

    print(f"OOS trades : {len(oos)}")
    print(
        f"OOS windows: {sorted(oos[WINDOW_COL].dropna().unique().astype(int).tolist())}"
    )

    if len(oos) == 0:
        raise RuntimeError("No OOS trades found.")

    return oos


# ======================================================================================
# YEAR STABILITY
# ======================================================================================


def build_year_table(oos: pd.DataFrame) -> pd.DataFrame:
    section("YEAR-BY-YEAR OOS PERFORMANCE")

    if "entry_timestamp" not in oos.columns:
        raise RuntimeError("entry_timestamp is required for year stability.")

    timestamps = pd.to_datetime(
        oos["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise RuntimeError("Unable to parse all entry timestamps.")

    work = oos.copy()
    work["_year"] = timestamps.dt.year

    rows = []

    for year, group in work.groupby("_year", sort=True):
        m = metrics(group)

        rows.append(
            {
                "year": int(year),
                **m,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No yearly groups were created.")

    result["contribution_pct"] = (
        result["total_R"] / result["total_R"].sum() * 100.0
        if result["total_R"].sum() != 0
        else np.nan
    )

    print(result.to_string(index=False))

    return result


# ======================================================================================
# WINDOW STABILITY — SAME DATA, YEAR CROSS-CHECK
# ======================================================================================


def build_window_table(oos: pd.DataFrame) -> pd.DataFrame:
    section("OOS WINDOW PERFORMANCE")

    rows = []

    for window, group in oos.groupby(WINDOW_COL, sort=True):
        m = metrics(group)

        rows.append(
            {
                "window": int(window),
                **m,
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result


# ======================================================================================
# REGIME DETECTION
# ======================================================================================


def detect_regime_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "regime",
        "market_regime",
        "entry_regime",
        "regime_label",
        "regime_type",
        "_regime",
        "relative_regime",
        "vol_regime",
        "trend_regime",
    ]

    return detect_column(df, candidates)


def build_regime_table(oos: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    section("REGIME STABILITY")

    regime_col = detect_regime_column(oos)

    if regime_col is None:
        print("No explicit regime column detected in S27 dataset.")
        print("REGIME TEST: SKIPPED — dataset does not expose a regime label.")
        return None, None

    print(f"Detected regime column: {regime_col}")

    values = oos[regime_col].astype(str)

    rows = []

    for regime, group in oos.groupby(values, sort=True):
        m = metrics(group)

        rows.append(
            {
                "regime": regime,
                **m,
            }
        )

    result = pd.DataFrame(rows)

    print(result.to_string(index=False))

    return result, regime_col


# ======================================================================================
# CONCENTRATION
# ======================================================================================


def concentration_analysis(
    year_table: pd.DataFrame,
    oos_metrics: dict,
) -> pd.DataFrame:

    section("OOS RESULT CONCENTRATION")

    total_r = oos_metrics["total_R"]

    years = year_table.copy()

    positive = years[years["total_R"] > 0].copy()

    positive = positive.sort_values(
        "total_R",
        ascending=False,
    )

    total_positive = positive["total_R"].sum()

    rows = []

    for rank, (_, row) in enumerate(
        positive.iterrows(),
        start=1,
    ):
        contribution = row["total_R"] / total_r * 100.0 if total_r != 0 else np.nan

        positive_share = (
            row["total_R"] / total_positive * 100.0 if total_positive != 0 else np.nan
        )

        rows.append(
            {
                "rank": rank,
                "year": int(row["year"]),
                "total_R": row["total_R"],
                "contribution_to_total_pct": contribution,
                "share_of_positive_year_R_pct": positive_share,
            }
        )

    concentration = pd.DataFrame(rows)

    if concentration.empty:
        print("No positive year groups.")
        return concentration

    top_1 = concentration.head(1)["total_R"].sum()
    top_2 = concentration.head(2)["total_R"].sum()
    top_3 = concentration.head(3)["total_R"].sum()

    print()
    print(f"Total OOS R                 : {total_r:.4f}")
    print(f"Top 1 positive year R       : {top_1:.4f}")
    print(f"Top 2 positive years R      : {top_2:.4f}")
    print(f"Top 3 positive years R      : {top_3:.4f}")

    if total_r != 0:
        print(f"Top 1 contribution          : {100 * top_1 / total_r:.2f}%")
        print(f"Top 2 contribution          : {100 * top_2 / total_r:.2f}%")
        print(f"Top 3 contribution          : {100 * top_3 / total_r:.2f}%")

    return concentration


# ======================================================================================
# AUDIT
# ======================================================================================


def audit(
    oos: pd.DataFrame,
    year_table: pd.DataFrame,
    window_table: pd.DataFrame,
    regime_table: pd.DataFrame | None,
) -> bool:

    section("S28.3 AUDIT")

    passed = True

    checks = {}

    checks["OOS trades present"] = len(oos) > 0

    checks["Finite S2R R"] = np.isfinite(oos[STRATEGY_R_COL].to_numpy()).all()

    checks["All OOS windows represented"] = set(
        oos[WINDOW_COL].astype(int).unique()
    ) == set(OOS_WINDOWS)

    checks["Year groups complete"] = int(year_table["trades"].sum()) == len(oos)

    checks["Window groups complete"] = int(window_table["trades"].sum()) == len(oos)

    if regime_table is not None:
        checks["Regime groups complete"] = int(regime_table["trades"].sum()) == len(oos)

    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"

        print(f"{name:<35}: {status}")

        if not ok:
            passed = False

    # Total-R reconciliation.
    oos_total = float(oos[STRATEGY_R_COL].sum())
    year_total = float(year_table["total_R"].sum())
    window_total = float(window_table["total_R"].sum())

    checks["Year R reconciliation"] = np.isclose(
        oos_total,
        year_total,
        atol=1e-10,
    )

    checks["Window R reconciliation"] = np.isclose(
        oos_total,
        window_total,
        atol=1e-10,
    )

    print(
        f"{'Year R reconciliation':<35}: "
        f"{'PASS' if checks['Year R reconciliation'] else 'FAIL'}"
    )

    print(
        f"{'Window R reconciliation':<35}: "
        f"{'PASS' if checks['Window R reconciliation'] else 'FAIL'}"
    )

    if not checks["Year R reconciliation"]:
        passed = False

    if not checks["Window R reconciliation"]:
        passed = False

    return passed


# ======================================================================================
# CHARTS
# ======================================================================================


def save_year_performance_chart(
    year_table: pd.DataFrame,
) -> None:

    path = OUTPUT_DIR / "s28_3_year_performance.png"

    plt.figure(figsize=(12, 7))

    plt.bar(
        year_table["year"].astype(str),
        year_table["total_R"],
    )

    plt.axhline(0.0, linestyle="--")

    plt.title("S28.3 — S2R OOS Performance by Year")
    plt.xlabel("Year")
    plt.ylabel("Total R")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(path)


def save_year_mean_chart(
    year_table: pd.DataFrame,
) -> None:

    path = OUTPUT_DIR / "s28_3_year_mean_R.png"

    plt.figure(figsize=(12, 7))

    plt.plot(
        year_table["year"],
        year_table["mean_R"],
        marker="o",
    )

    plt.axhline(0.0, linestyle="--")

    plt.title("S28.3 — S2R OOS Mean R by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean R")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(path)


def save_year_winrate_chart(
    year_table: pd.DataFrame,
) -> None:

    path = OUTPUT_DIR / "s28_3_year_winrate.png"

    plt.figure(figsize=(12, 7))

    plt.plot(
        year_table["year"],
        year_table["win_rate"] * 100.0,
        marker="o",
    )

    plt.axhline(50.0, linestyle="--")

    plt.title("S28.3 — S2R OOS Win Rate by Year")
    plt.xlabel("Year")
    plt.ylabel("Win Rate (%)")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(path)


def save_window_heatmap_style_chart(
    window_table: pd.DataFrame,
) -> None:

    path = OUTPUT_DIR / "s28_3_window_total_R.png"

    plt.figure(figsize=(13, 7))

    plt.plot(
        window_table["window"],
        window_table["total_R"],
        marker="o",
    )

    plt.axhline(0.0, linestyle="--")

    plt.title("S28.3 — S2R OOS Total R by Validation Window")
    plt.xlabel("OOS Window")
    plt.ylabel("Total R")

    plt.xticks(window_table["window"])

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(path)


def save_year_contribution_chart(
    concentration: pd.DataFrame,
) -> None:

    if concentration.empty:
        return

    path = OUTPUT_DIR / "s28_3_year_concentration.png"

    plt.figure(figsize=(12, 7))

    plt.bar(
        concentration["year"].astype(str),
        concentration["contribution_to_total_pct"],
    )

    plt.axhline(0.0, linestyle="--")

    plt.title("S28.3 — S2R OOS Year Contribution to Total R")
    plt.xlabel("Year")
    plt.ylabel("Contribution (%)")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(path)


# ======================================================================================
# SAVE
# ======================================================================================


def save_results(
    year_table: pd.DataFrame,
    window_table: pd.DataFrame,
    regime_table: pd.DataFrame | None,
    concentration: pd.DataFrame,
) -> None:

    section("SAVING RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_path = OUTPUT_DIR / "s28_3_year_stability.csv"
    window_path = OUTPUT_DIR / "s28_3_window_stability.csv"
    concentration_path = OUTPUT_DIR / "s28_3_year_concentration.csv"

    year_table.to_csv(
        year_path,
        index=False,
    )

    window_table.to_csv(
        window_path,
        index=False,
    )

    concentration.to_csv(
        concentration_path,
        index=False,
    )

    print(year_path)
    print(window_path)
    print(concentration_path)

    if regime_table is not None:
        regime_path = OUTPUT_DIR / "s28_3_regime_stability.csv"

        regime_table.to_csv(
            regime_path,
            index=False,
        )

        print(regime_path)


# ======================================================================================
# MAIN
# ======================================================================================


def main() -> None:

    print("=" * 110)
    print("S28.3 YEAR / REGIME STABILITY")
    print("=" * 110)

    print()
    print("Frozen S2R strategy.")
    print("NO PARAMETER OPTIMIZATION.")
    print()
    print("Frozen model:")
    print(f"  MAE >= {MAE_THRESHOLD:.2f}R")
    print(f"  Recovery >= +{RECOVERY_LEVEL:.2f}R")
    print(f"  Deadline = {RECOVERY_DEADLINE} bars")
    print()
    print(f"OOS windows: {OOS_WINDOWS}")

    df = load_strategy()

    oos = build_oos(df)

    oos_metrics = metrics(oos)

    section("FULL OOS BASELINE")

    print(f"Trades        : {oos_metrics['trades']}")
    print(f"Total R       : {oos_metrics['total_R']:.4f}")
    print(f"Mean R        : {oos_metrics['mean_R']:.6f}")
    print(f"Win rate      : {oos_metrics['win_rate'] * 100:.4f}%")
    print(f"Profit Factor : {oos_metrics['profit_factor']}")
    print(f"Max DD        : {oos_metrics['max_drawdown_R']:.4f}")

    year_table = build_year_table(oos)

    window_table = build_window_table(oos)

    regime_table, regime_col = build_regime_table(oos)

    concentration = concentration_analysis(
        year_table,
        oos_metrics,
    )

    audit_pass = audit(
        oos,
        year_table,
        window_table,
        regime_table,
    )

    section("GENERATING CHARTS")

    save_year_performance_chart(year_table)
    save_year_mean_chart(year_table)
    save_year_winrate_chart(year_table)
    save_window_heatmap_style_chart(window_table)
    save_year_contribution_chart(concentration)

    save_results(
        year_table,
        window_table,
        regime_table,
        concentration,
    )

    section("S28.3 FINAL STATUS")

    print(f"OOS Total R : {oos_metrics['total_R']:.4f}")
    print(f"OOS Trades  : {oos_metrics['trades']}")

    positive_years = int((year_table["total_R"] > 0).sum())

    negative_years = int((year_table["total_R"] < 0).sum())

    print(f"Positive years: {positive_years}")
    print(f"Negative years: {negative_years}")

    if audit_pass:
        print()
        print("PASS — S28.3 integrity audit passed.")
    else:
        print()
        print("FAIL — S28.3 integrity audit failed.")
        raise RuntimeError("S28.3 audit failed.")

    print()
    print("S28.3 YEAR / REGIME STABILITY COMPLETE")


if __name__ == "__main__":
    main()
