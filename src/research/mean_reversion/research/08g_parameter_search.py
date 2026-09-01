from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =============================================================================
# RESEARCH 08G
# =============================================================================
#
# LOCALIZED PARAMETER DISCOVERY
#
# Context fixed by Research 08D:
#   SIDE × HMM STATE × VOLATILITY BUCKET × Z-SCORE
#
# Parameter dimensions:
#   TP × SL × HORIZON
#
# Uses ONLY the path relationships already calculated by Research 07.
#
# No:
#   - HMM retraining
#   - volatility recalculation
#   - new context discovery
#   - failure testing
#   - production changes
#
# RR < 1 is allowed.
#
# Main objective:
# identify parameter combinations with:
#   1. high win rate
#   2. positive expectancy
#   3. sufficient observations
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = ROOT / "research" / "mean_reversion" / "results"

DETAIL_PATH = RESULTS_DIR / "research_08d_context_detail.csv"

OUTPUT_ALL = RESULTS_DIR / "research_08g_parameter_results.csv"

OUTPUT_CONTEXT_BEST = RESULTS_DIR / "research_08g_best_per_context.csv"

OUTPUT_CANDIDATES = RESULTS_DIR / "research_08g_strategy_candidates.csv"

OUTPUT_SIDE_BEST = RESULTS_DIR / "research_08g_best_per_side.csv"


# =============================================================================
# RESEARCH SETTINGS
# =============================================================================

MIN_OBSERVATIONS = 100

TARGET_WR = 0.50

MIN_EXPECTANCY = 0.0


# =============================================================================
# PRINT
# =============================================================================


def section(title: str) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# =============================================================================
# LOAD 08D
# =============================================================================


def load_detail() -> pd.DataFrame:

    section("LOADING RESEARCH 08D")

    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Research 08D detail file not found:\n{DETAIL_PATH}")

    df = pd.read_csv(DETAIL_PATH)

    df.columns = [str(c).strip() for c in df.columns]

    print(f"Rows: {len(df):,}")

    print(f"Columns: {len(df.columns):,}")

    required = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "observations",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return df


# =============================================================================
# DISCOVER PATH COLUMNS
# =============================================================================


def find_path_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("DISCOVERING RESEARCH 07 PATH RELATIONSHIPS")

    pattern = re.compile(r"^wr_(\d+(?:\.\d+)?)_before_(\d+(?:\.\d+)?)_h(\d+)$")

    paths = []

    for column in df.columns:
        match = pattern.match(str(column))

        if match is None:
            continue

        tp = float(match.group(1))

        sl = float(match.group(2))

        horizon = int(match.group(3))

        if sl <= 0:
            continue

        rr = tp / sl

        paths.append(
            {
                "column": column,
                "tp": tp,
                "sl": sl,
                "rr": rr,
                "horizon": horizon,
            }
        )

    if not paths:
        raise RuntimeError("No Research 07 path relationships found.")

    paths_df = pd.DataFrame(paths)

    paths_df = paths_df.sort_values(
        [
            "horizon",
            "rr",
            "tp",
            "sl",
        ]
    ).reset_index(drop=True)

    print(f"Path relationships found: {len(paths_df):,}")

    print()

    print(paths_df.to_string(index=False))

    return paths_df


# =============================================================================
# EXPECTANCY
# =============================================================================


def calculate_expectancy(
    win_rate: float,
    rr: float,
) -> float:

    if not np.isfinite(win_rate):
        return np.nan

    return win_rate * rr - (1.0 - win_rate)


# =============================================================================
# BUILD PARAMETER TABLE
# =============================================================================


def build_parameter_table(
    detail: pd.DataFrame,
    paths: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING CONTEXT × PARAMETER TABLE")

    rows = []

    for _, context in detail.iterrows():
        observations = pd.to_numeric(
            context["observations"],
            errors="coerce",
        )

        if not np.isfinite(observations):
            continue

        observations = int(observations)

        for _, path in paths.iterrows():
            column = path["column"]

            if column not in context.index:
                continue

            wr = pd.to_numeric(
                context[column],
                errors="coerce",
            )

            if not np.isfinite(wr):
                continue

            tp = float(path["tp"])

            sl = float(path["sl"])

            rr = float(path["rr"])

            horizon = int(path["horizon"])

            exp = calculate_expectancy(
                float(wr),
                rr,
            )

            breakeven_wr = 1.0 / (1.0 + rr)

            wr_edge = float(wr) - breakeven_wr

            rows.append(
                {
                    "side": str(context["side"]).upper(),
                    "hmm_state": int(context["hmm_state"]),
                    "vol_bucket": str(context["vol_bucket"]),
                    "zscore": float(context["zscore"]),
                    "observations": observations,
                    "tp": tp,
                    "sl": sl,
                    "rr": rr,
                    "horizon": horizon,
                    "win_rate": float(wr),
                    "expectancy_r": exp,
                    "breakeven_wr": breakeven_wr,
                    "wr_edge_vs_breakeven": wr_edge,
                    "wr_ge_50": (float(wr) >= TARGET_WR),
                    "positive_expectancy": (exp > MIN_EXPECTANCY),
                    "sufficient_sample": (observations >= MIN_OBSERVATIONS),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No parameter observations generated.")

    print(f"Parameter rows: {len(result):,}")

    return result


# =============================================================================
# RANK
# =============================================================================


def rank_parameters(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["candidate"] = (
        result["sufficient_sample"] & result["wr_ge_50"] & result["positive_expectancy"]
    )

    # Research ranking only.
    #
    # WR gets the highest weight because that is our current
    # research objective, but expectancy remains important.

    result["research_score"] = (
        result["win_rate"] * 0.50
        + result["expectancy_r"] * 0.35
        + result["wr_edge_vs_breakeven"] * 0.15
    )

    result = result.sort_values(
        by=[
            "candidate",
            "research_score",
            "win_rate",
            "expectancy_r",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return result


# =============================================================================
# BEST PER CONTEXT
# =============================================================================


def best_per_context(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("BEST PARAMETERS PER CONTEXT")

    context_columns = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
    ]

    ordered = df.sort_values(
        by=[
            *context_columns,
            "candidate",
            "research_score",
            "win_rate",
            "expectancy_r",
            "observations",
        ],
        ascending=[
            True,  # side
            True,  # HMM
            True,  # volatility
            True,  # z-score
            False,  # candidate
            False,  # research score
            False,  # WR
            False,  # expectancy
            False,  # observations
        ],
    )

    best = ordered.drop_duplicates(
        subset=context_columns,
        keep="first",
    ).copy()

    print(f"Contexts: {len(best):,}")

    return best


# =============================================================================
# CANDIDATES
# =============================================================================


def select_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("SELECTING STRATEGY CANDIDATES")

    candidates = df[df["candidate"]].copy()

    print(f"Candidates with WR >= 50% + positive expectancy: {len(candidates):,}")

    if not candidates.empty:
        candidates["candidate_type"] = "WR_GE_50_POSITIVE_EXPECTANCY"

        return candidates.sort_values(
            by=[
                "win_rate",
                "expectancy_r",
                "research_score",
                "observations",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )

    # -------------------------------------------------------------------------
    # FALLBACK
    # -------------------------------------------------------------------------

    fallback = df[(df["sufficient_sample"]) & (df["positive_expectancy"])].copy()

    fallback["candidate"] = False

    fallback["candidate_type"] = "POSITIVE_EXPECTANCY_BELOW_50_WR"

    fallback = fallback.sort_values(
        by=[
            "expectancy_r",
            "win_rate",
            "research_score",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    print("No configuration reached WR >= 50%.")

    print(f"Keeping positive-expectancy fallback rows: {len(fallback):,}")

    return fallback


# =============================================================================
# BEST PER SIDE
# =============================================================================


def best_per_side(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    section("BEST PARAMETER SET PER SIDE")

    if candidates.empty:
        print("No candidates.")

        return pd.DataFrame()

    ordered = candidates.sort_values(
        by=[
            "side",
            "candidate",
            "research_score",
            "win_rate",
            "expectancy_r",
            "observations",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            False,
        ],
    )

    best = ordered.drop_duplicates(
        subset=["side"],
        keep="first",
    ).copy()

    columns = [
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
        "wr_edge_vs_breakeven",
        "candidate",
    ]

    print(
        best[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return best


# =============================================================================
# TOP RESULTS
# =============================================================================


def print_top_results(
    df: pd.DataFrame,
) -> None:

    section("TOP PARAMETER COMBINATIONS")

    columns = [
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
        "wr_edge_vs_breakeven",
        "candidate",
    ]

    print(
        df[columns]
        .head(100)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# WR >= 50
# =============================================================================


def print_50_wr(
    df: pd.DataFrame,
) -> None:

    section("CONFIGURATIONS WITH WR >= 50%")

    high_wr = df[
        (df["win_rate"] >= 0.50) & (df["observations"] >= MIN_OBSERVATIONS)
    ].copy()

    if high_wr.empty:
        print("NONE.")

        return

    high_wr = high_wr.sort_values(
        by=[
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

    columns = [
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

    print(
        high_wr[columns]
        .head(100)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# BEST CONTEXTS
# =============================================================================


def print_best_contexts(
    best_context: pd.DataFrame,
) -> None:

    section("BEST PARAMETER PER CONTEXT")

    if best_context.empty:
        print("No contexts.")

        return

    columns = [
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
        "wr_edge_vs_breakeven",
        "candidate",
    ]

    display = best_context[columns].sort_values(
        by=[
            "candidate",
            "win_rate",
            "expectancy_r",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    print(
        display.head(100).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08G")

    print("LOCALIZED PARAMETER DISCOVERY")

    print("-" * 100)

    print("Context inherited from Research 08D.")

    print("No HMM retraining.")

    print("No volatility recalculation.")

    print("No failure test.")

    print("No production changes.")

    print("RR < 1 is allowed.")

    print()
    print("Primary objective:")

    print("Find TP / SL / horizon combinations with WR >= 50% and positive expectancy.")

    # =========================================================================
    # LOAD
    # =========================================================================

    detail = load_detail()

    # =========================================================================
    # PATHS
    # =========================================================================

    paths = find_path_columns(detail)

    # =========================================================================
    # PARAMETER TABLE
    # =========================================================================

    parameter_table = build_parameter_table(
        detail,
        paths,
    )

    # =========================================================================
    # RANK
    # =========================================================================

    ranked = rank_parameters(parameter_table)

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print_top_results(ranked)

    print_50_wr(ranked)

    # =========================================================================
    # BEST CONTEXT
    # =========================================================================

    best_context = best_per_context(ranked)

    print_best_contexts(best_context)

    # =========================================================================
    # CANDIDATES
    # =========================================================================

    candidates = select_candidates(ranked)

    # =========================================================================
    # BEST SIDE
    # =========================================================================

    best_side = best_per_side(candidates)

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    best_context.to_csv(
        OUTPUT_CONTEXT_BEST,
        index=False,
    )

    candidates.to_csv(
        OUTPUT_CANDIDATES,
        index=False,
    )

    best_side.to_csv(
        OUTPUT_SIDE_BEST,
        index=False,
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section("RESEARCH 08G COMPLETE")

    print("Files saved:")

    print()
    print(OUTPUT_ALL)

    print(OUTPUT_CONTEXT_BEST)

    print(OUTPUT_CANDIDATES)

    print(OUTPUT_SIDE_BEST)

    print()
    print("IMPORTANT:")

    print("These are research candidates, NOT a final strategy.")

    print("Next step: validate the strongest candidates before failure testing.")


if __name__ == "__main__":
    main()
