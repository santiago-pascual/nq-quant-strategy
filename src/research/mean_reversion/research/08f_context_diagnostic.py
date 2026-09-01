from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = ROOT / "research" / "mean_reversion" / "results"

DETAIL_PATH = RESULTS_DIR / "research_08d_context_detail.csv"

SUMMARY_PATH = RESULTS_DIR / "research_08d_context_summary.csv"

OUTPUT_ALL = RESULTS_DIR / "research_08f_context_diagnostic.csv"

OUTPUT_CANDIDATES = RESULTS_DIR / "research_08f_context_candidates.csv"

OUTPUT_CONTEXT_SUMMARY = RESULTS_DIR / "research_08f_context_summary.csv"


# =============================================================================
# SETTINGS
# =============================================================================

MIN_OBSERVATIONS = 100

# Minimum number of individual path relationships
# that must reach 50% WR.
MIN_PATHS_GE_50 = 2

# Stronger candidate threshold.
MIN_PATHS_GE_55 = 1

# We explicitly allow RR < 1.
# The objective here is to discover contexts with
# high WR and positive theoretical expectancy.
MIN_EXPECTANCY = 0.0


# =============================================================================
# HELPERS
# =============================================================================


def section(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# LOAD
# =============================================================================


def load_data():

    section("LOADING RESEARCH 08D")

    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Missing Research 08D detail:\n{DETAIL_PATH}")

    detail = pd.read_csv(DETAIL_PATH)

    detail.columns = [str(c).strip() for c in detail.columns]

    print(f"Detail rows: {len(detail):,}")

    print(f"Detail columns: {len(detail.columns):,}")

    required = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "observations",
    ]

    missing = [c for c in required if c not in detail.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return detail


# =============================================================================
# PARSE PATH COLUMNS
# =============================================================================


def parse_path_columns(
    df: pd.DataFrame,
):

    pattern = re.compile(r"^wr_(\d+(?:\.\d+)?)_before_(\d+(?:\.\d+)?)_h(\d+)$")

    rows = []

    for column in df.columns:
        match = pattern.match(str(column))

        if match is None:
            continue

        tp = float(match.group(1))

        sl = float(match.group(2))

        horizon = int(match.group(3))

        rr = tp / sl

        rows.append(
            {
                "column": column,
                "tp": tp,
                "sl": sl,
                "rr": rr,
                "horizon": horizon,
            }
        )

    if not rows:
        raise RuntimeError("No WR path columns were found.")

    paths = pd.DataFrame(rows)

    return paths


# =============================================================================
# CALCULATE EXPECTANCY
# =============================================================================


def calculate_expectancy(
    win_rate: float,
    rr: float,
) -> float:

    if not np.isfinite(win_rate):
        return np.nan

    return win_rate * rr - (1.0 - win_rate)


# =============================================================================
# EXPAND ALL CONTEXTS
# =============================================================================


def build_diagnostic_table(
    detail: pd.DataFrame,
):

    section("EXPANDING CONTEXT × PATH RELATIONSHIPS")

    paths = parse_path_columns(detail)

    print(f"Path relationships: {len(paths):,}")

    rows = []

    for _, context in detail.iterrows():
        observations = float(context["observations"])

        if not np.isfinite(observations):
            continue

        for _, path in paths.iterrows():
            column = path["column"]

            wr = pd.to_numeric(
                context[column],
                errors="coerce",
            )

            if not np.isfinite(wr):
                continue

            expectancy = calculate_expectancy(
                wr,
                path["rr"],
            )

            rows.append(
                {
                    "side": str(context["side"]).upper(),
                    "hmm_state": int(context["hmm_state"]),
                    "vol_bucket": str(context["vol_bucket"]),
                    "zscore": float(context["zscore"]),
                    "observations": int(observations),
                    "tp": float(path["tp"]),
                    "sl": float(path["sl"]),
                    "rr": float(path["rr"]),
                    "horizon": int(path["horizon"]),
                    "win_rate": float(wr),
                    "expectancy_r": float(expectancy),
                    "wr_ge_50": bool(wr >= 0.50),
                    "wr_ge_55": bool(wr >= 0.55),
                    "wr_ge_60": bool(wr >= 0.60),
                    "positive_expectancy": bool(expectancy > 0),
                }
            )

    result = pd.DataFrame(rows)

    print(f"Diagnostic rows: {len(result):,}")

    return result


# =============================================================================
# CONTEXT AGGREGATION
# =============================================================================


def aggregate_contexts(
    diagnostic: pd.DataFrame,
):

    section("AGGREGATING CONTEXT EVIDENCE")

    group_cols = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
    ]

    grouped = diagnostic.groupby(
        group_cols,
        as_index=False,
    ).agg(
        observations=(
            "observations",
            "max",
        ),
        path_relationships=(
            "win_rate",
            "count",
        ),
        paths_wr_50=(
            "wr_ge_50",
            "sum",
        ),
        paths_wr_55=(
            "wr_ge_55",
            "sum",
        ),
        paths_wr_60=(
            "wr_ge_60",
            "sum",
        ),
        paths_positive_expectancy=(
            "positive_expectancy",
            "sum",
        ),
        best_wr=(
            "win_rate",
            "max",
        ),
        median_wr=(
            "win_rate",
            "median",
        ),
        mean_wr=(
            "win_rate",
            "mean",
        ),
        best_expectancy=(
            "expectancy_r",
            "max",
        ),
        median_expectancy=(
            "expectancy_r",
            "median",
        ),
    )

    # -------------------------------------------------------------------------
    # Identify the best individual path for each context.
    # -------------------------------------------------------------------------

    diagnostic_sorted = diagnostic.sort_values(
        [
            "side",
            "hmm_state",
            "vol_bucket",
            "zscore",
            "win_rate",
            "expectancy_r",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
            False,
        ],
    )

    best_path = diagnostic_sorted.drop_duplicates(
        [
            "side",
            "hmm_state",
            "vol_bucket",
            "zscore",
        ],
        keep="first",
    )[
        [
            "side",
            "hmm_state",
            "vol_bucket",
            "zscore",
            "tp",
            "sl",
            "rr",
            "horizon",
            "win_rate",
            "expectancy_r",
        ]
    ].rename(
        columns={
            "tp": "best_tp",
            "sl": "best_sl",
            "rr": "best_rr",
            "horizon": "best_horizon",
            "win_rate": "best_path_wr",
            "expectancy_r": "best_path_expectancy",
        }
    )

    grouped = grouped.merge(
        best_path,
        on=group_cols,
        how="left",
    )

    # -------------------------------------------------------------------------
    # Context quality flags.
    # -------------------------------------------------------------------------

    grouped["passes_wr_50"] = grouped["paths_wr_50"] >= MIN_PATHS_GE_50

    grouped["passes_wr_55"] = grouped["paths_wr_55"] >= MIN_PATHS_GE_55

    grouped["passes_positive_expectancy"] = grouped["paths_positive_expectancy"] >= 1

    grouped["candidate"] = (
        (grouped["observations"] >= MIN_OBSERVATIONS)
        & (grouped["passes_wr_50"])
        & (grouped["passes_positive_expectancy"])
    )

    return grouped


# =============================================================================
# SELECT BEST CANDIDATES
# =============================================================================


def select_candidates(
    summary: pd.DataFrame,
):

    section("SELECTING CONTEXT CANDIDATES")

    candidates = summary[summary["candidate"]].copy()

    print(f"Contexts passing candidate filter: {len(candidates):,}")

    if candidates.empty:
        print("\nNo context satisfies all filters.")

        print("Relaxing ONLY the number of 50% paths for diagnostic purposes.")

        fallback = summary[
            (summary["observations"] >= MIN_OBSERVATIONS)
            & (summary["best_path_wr"] >= 0.50)
        ].copy()

        fallback["candidate_type"] = "BEST_SINGLE_PATH"

        return fallback

    candidates["candidate_type"] = "MULTI_PATH_50"

    return candidates


# =============================================================================
# PRINT ALL PATHS ≥ 50%
# =============================================================================


def print_high_wr_paths(
    diagnostic: pd.DataFrame,
):

    section("INDIVIDUAL PATHS WITH WR >= 50%")

    high = diagnostic[
        (diagnostic["win_rate"] >= 0.50)
        & (diagnostic["observations"] >= MIN_OBSERVATIONS)
    ].copy()

    if high.empty:
        print("NONE.")

        return

    high = high.sort_values(
        [
            "win_rate",
            "expectancy_r",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print(
        high[
            [
                "side",
                "hmm_state",
                "vol_bucket",
                "zscore",
                "observations",
                "tp",
                "sl",
                "rr",
                "horizon",
                "win_rate",
                "expectancy_r",
            ]
        ]
        .head(100)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# PRINT CONTEXTS
# =============================================================================


def print_contexts(
    summary: pd.DataFrame,
):

    section("CONTEXT RANKING")

    if summary.empty:
        print("No contexts.")

        return

    ranked = summary.sort_values(
        [
            "candidate",
            "paths_wr_60",
            "paths_wr_55",
            "paths_wr_50",
            "best_path_wr",
            "best_path_expectancy",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )

    columns = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "observations",
        "paths_wr_50",
        "paths_wr_55",
        "paths_wr_60",
        "paths_positive_expectancy",
        "best_path_wr",
        "best_tp",
        "best_sl",
        "best_rr",
        "best_horizon",
        "best_path_expectancy",
        "mean_wr",
        "median_wr",
        "candidate",
    ]

    print(
        ranked[columns]
        .head(50)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# PRINT BEST CONTEXT PER SIDE
# =============================================================================


def print_best_per_side(
    candidates: pd.DataFrame,
):

    section("BEST CONTEXTS BY SIDE")

    for side in [
        "LONG",
        "SHORT",
    ]:
        subset = candidates[candidates["side"] == side].copy()

        if subset.empty:
            print(f"{side}: NONE")

            continue

        subset = subset.sort_values(
            [
                "paths_wr_50",
                "paths_wr_55",
                "best_path_wr",
                "best_path_expectancy",
                "observations",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )

        row = subset.iloc[0]

        print(
            f"{side}: "
            f"HMM={int(row['hmm_state'])} | "
            f"VOL={row['vol_bucket']} | "
            f"Z={row['zscore']:.2f} | "
            f"N={int(row['observations']):,} | "
            f"best WR={row['best_path_wr']:.2%} | "
            f"best RR={row['best_rr']:.2f} | "
            f"best E={row['best_path_expectancy']:.4f}"
        )


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08F")

    print("CONTEXT DIAGNOSTIC")

    print("-" * 100)

    print("Uses Research 08D outputs only.")

    print("No HMM retraining.")

    print("No volatility recalculation.")

    print("No new parameter optimization.")

    print("RR < 1 is explicitly allowed.")

    print("Primary objective:")

    print("Find contexts where WR >= 50% and theoretical expectancy can be positive.")

    # =========================================================================
    # LOAD
    # =========================================================================

    detail = load_data()

    # =========================================================================
    # EXPAND
    # =========================================================================

    diagnostic = build_diagnostic_table(detail)

    # =========================================================================
    # AGGREGATE
    # =========================================================================

    summary = aggregate_contexts(diagnostic)

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    candidates = select_candidates(summary)

    # =========================================================================
    # PRINT
    # =========================================================================

    print_high_wr_paths(diagnostic)

    print_contexts(summary)

    print_best_per_side(candidates)

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    candidates.to_csv(
        OUTPUT_CANDIDATES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_CONTEXT_SUMMARY,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    section("RESEARCH 08F COMPLETE")

    print(f"All path diagnostics:\n{OUTPUT_ALL}")

    print(f"\nContext candidates:\n{OUTPUT_CANDIDATES}")

    print(f"\nContext summary:\n{OUTPUT_CONTEXT_SUMMARY}")

    print()
    print("NEXT STEP:")

    print(
        "Only after selecting the context do we move to localized parameter research."
    )

    print("No final strategy was constructed.")


if __name__ == "__main__":
    main()
