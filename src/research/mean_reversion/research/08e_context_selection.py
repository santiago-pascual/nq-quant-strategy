from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "research" / "mean_reversion" / "results"

SUMMARY_PATH = RESULTS_DIR / "research_08d_context_summary.csv"

DETAIL_PATH = RESULTS_DIR / "research_08d_context_detail.csv"

OUTPUT_CONTEXTS = RESULTS_DIR / "research_08e_context_candidates.csv"

OUTPUT_ALL = RESULTS_DIR / "research_08e_context_ranked.csv"


# =============================================================================
# RESEARCH SETTINGS
# =============================================================================

MIN_OBSERVATIONS = 100
MIN_WINDOWS = 10

# We want contexts with genuine WR potential.
TARGET_WR = 0.50

# Ranking weights.
WEIGHT_WR = 0.45
WEIGHT_STABILITY = 0.25
WEIGHT_MFE_MAE = 0.20
WEIGHT_SAMPLE = 0.10


# =============================================================================
# HELPERS
# =============================================================================


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    return df


# =============================================================================
# LOAD
# =============================================================================


def load_research_08d() -> tuple[pd.DataFrame, pd.DataFrame]:

    section("LOADING RESEARCH 08D RESULTS")

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{SUMMARY_PATH}")

    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{DETAIL_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)

    detail = pd.read_csv(DETAIL_PATH)

    summary = normalize_columns(summary)

    detail = normalize_columns(detail)

    print(f"Summary rows: {len(summary):,}")

    print(f"Detail rows:  {len(detail):,}")

    print("\nSummary columns:")

    print(summary.columns.tolist())

    print("\nDetail columns:")

    print(detail.columns.tolist())

    return summary, detail


# =============================================================================
# IDENTIFY COLUMNS
# =============================================================================


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:

    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise RuntimeError(f"Could not identify any of these columns: {candidates}")

    return None


# =============================================================================
# BUILD MASTER CONTEXT TABLE
# =============================================================================


def build_master_table(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:

    section("BUILDING MASTER CONTEXT TABLE")

    # -------------------------------------------------------------------------
    # Expected context dimensions.
    # -------------------------------------------------------------------------

    side_col = find_column(
        summary,
        ["side"],
    )

    hmm_col = find_column(
        summary,
        ["hmm_state"],
    )

    vol_col = find_column(
        summary,
        ["vol_bucket"],
    )

    z_col = find_column(
        summary,
        ["zscore"],
    )

    obs_col = find_column(
        summary,
        [
            "observations",
            "observation_count",
        ],
    )

    windows_col = find_column(
        summary,
        [
            "windows",
            "window_count",
        ],
        required=False,
    )

    wr_col = find_column(
        summary,
        [
            "mean_path_wr",
            "path_wr",
            "mean_wr",
        ],
    )

    median_wr_col = find_column(
        summary,
        [
            "median_path_wr",
            "median_wr",
        ],
        required=False,
    )

    mfe_col = find_column(
        summary,
        [
            "mean_mfe",
            "mfe",
        ],
        required=False,
    )

    mae_col = find_column(
        summary,
        [
            "mean_mae",
            "mae",
        ],
        required=False,
    )

    result = pd.DataFrame()

    result["side"] = summary[side_col].astype(str).str.upper()

    result["hmm_state"] = pd.to_numeric(
        summary[hmm_col],
        errors="coerce",
    )

    result["vol_bucket"] = summary[vol_col].astype(str)

    result["zscore"] = pd.to_numeric(
        summary[z_col],
        errors="coerce",
    )

    result["observations"] = pd.to_numeric(
        summary[obs_col],
        errors="coerce",
    )

    if windows_col is not None:
        result["windows"] = pd.to_numeric(
            summary[windows_col],
            errors="coerce",
        )

    else:
        result["windows"] = np.nan

    result["mean_path_wr"] = pd.to_numeric(
        summary[wr_col],
        errors="coerce",
    )

    if median_wr_col is not None:
        result["median_path_wr"] = pd.to_numeric(
            summary[median_wr_col],
            errors="coerce",
        )

    else:
        result["median_path_wr"] = np.nan

    if mfe_col is not None:
        result["mean_mfe"] = pd.to_numeric(
            summary[mfe_col],
            errors="coerce",
        )

    else:
        result["mean_mfe"] = np.nan

    if mae_col is not None:
        result["mean_mae"] = pd.to_numeric(
            summary[mae_col],
            errors="coerce",
        )

    else:
        result["mean_mae"] = np.nan

    # -------------------------------------------------------------------------
    # Remove invalid rows.
    # -------------------------------------------------------------------------

    result = result[result["hmm_state"].isin([0, 1, 2])]

    result = result[
        result["vol_bucket"].isin(
            [
                "0-20",
                "20-40",
                "40-60",
                "60-80",
                "80-100",
            ]
        )
    ]

    result = result[
        result["side"].isin(
            [
                "LONG",
                "SHORT",
            ]
        )
    ]

    result = result[np.isfinite(result["zscore"])]

    result = result[np.isfinite(result["mean_path_wr"])]

    result = result[np.isfinite(result["observations"])]

    result = result.reset_index(drop=True)

    print(f"Valid context rows: {len(result):,}")

    return result


# =============================================================================
# STABILITY SCORE
# =============================================================================


def calculate_stability(
    detail: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:

    section("CALCULATING CONTEXT STABILITY")

    # -------------------------------------------------------------------------
    # Try to identify window-level columns.
    # -------------------------------------------------------------------------

    if detail.empty:
        master["window_wr_std"] = np.nan
        master["windows_positive"] = np.nan
        master["windows_tested"] = np.nan

        return master

    detail = normalize_columns(detail)

    # Find context columns.
    side_col = find_column(
        detail,
        ["side"],
        required=False,
    )

    hmm_col = find_column(
        detail,
        ["hmm_state"],
        required=False,
    )

    vol_col = find_column(
        detail,
        ["vol_bucket"],
        required=False,
    )

    z_col = find_column(
        detail,
        ["zscore"],
        required=False,
    )

    window_col = find_column(
        detail,
        ["window"],
        required=False,
    )

    wr_col = find_column(
        detail,
        [
            "path_wr",
            "mean_path_wr",
            "wr",
        ],
        required=False,
    )

    if (
        side_col is None
        or hmm_col is None
        or vol_col is None
        or z_col is None
        or window_col is None
        or wr_col is None
    ):
        print("Window-level detail columns unavailable.")

        master["window_wr_std"] = np.nan
        master["windows_positive"] = np.nan
        master["windows_tested"] = np.nan

        return master

    detail = detail.copy()

    detail["hmm_state"] = pd.to_numeric(
        detail[hmm_col],
        errors="coerce",
    )

    detail["zscore"] = pd.to_numeric(
        detail[z_col],
        errors="coerce",
    )

    detail["path_wr"] = pd.to_numeric(
        detail[wr_col],
        errors="coerce",
    )

    detail["window"] = pd.to_numeric(
        detail[window_col],
        errors="coerce",
    )

    detail["side_norm"] = detail[side_col].astype(str).str.upper()

    detail["vol_norm"] = detail[vol_col].astype(str)

    detail = detail[np.isfinite(detail["path_wr"])]

    # -------------------------------------------------------------------------
    # Aggregate by context.
    # -------------------------------------------------------------------------

    grouped = (
        detail.groupby(
            [
                "side_norm",
                "hmm_state",
                "vol_norm",
                "zscore",
            ],
            dropna=False,
        )
        .agg(
            window_wr_std=(
                "path_wr",
                "std",
            ),
            windows_positive=(
                "path_wr",
                lambda x: int((x >= TARGET_WR).sum()),
            ),
            windows_tested=(
                "window",
                "nunique",
            ),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "side_norm": "side",
            "vol_norm": "vol_bucket",
        }
    )

    master = master.merge(
        grouped,
        on=[
            "side",
            "hmm_state",
            "vol_bucket",
            "zscore",
        ],
        how="left",
    )

    return master


# =============================================================================
# SCORE
# =============================================================================


def score_contexts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("SCORING CONTEXTS")

    df = df.copy()

    # -------------------------------------------------------------------------
    # Sample score.
    # -------------------------------------------------------------------------

    df["sample_score"] = np.clip(
        np.log1p(df["observations"]) / np.log1p(df["observations"].max()),
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # WR score.
    #
    # 50% is the reference point.
    # We reward higher WR but do not require it yet.
    # -------------------------------------------------------------------------

    df["wr_score"] = np.clip(
        (df["mean_path_wr"] - 0.25) / 0.50,
        0,
        1,
    )

    # -------------------------------------------------------------------------
    # Stability.
    # -------------------------------------------------------------------------

    if df["windows_positive"].notna().any():
        df["positive_window_ratio"] = df["windows_positive"] / df[
            "windows_tested"
        ].replace(0, np.nan)

        df["stability_score"] = df["positive_window_ratio"].clip(0, 1)

    else:
        df["positive_window_ratio"] = np.nan
        df["stability_score"] = 0.5

    # -------------------------------------------------------------------------
    # MFE / MAE efficiency.
    #
    # This is descriptive only.
    # It is NOT a chosen SL/TP.
    # -------------------------------------------------------------------------

    if "mean_mfe" in df.columns and "mean_mae" in df.columns:
        ratio = df["mean_mfe"] / df["mean_mae"].replace(0, np.nan)

        df["mfe_mae_score"] = (
            ratio.clip(
                0,
                2,
            )
            / 2
        )

    else:
        df["mfe_mae_score"] = 0.5

    # -------------------------------------------------------------------------
    # Final research ranking.
    # -------------------------------------------------------------------------

    df["research_score"] = (
        WEIGHT_WR * df["wr_score"]
        + WEIGHT_STABILITY * df["stability_score"]
        + WEIGHT_MFE_MAE * df["mfe_mae_score"]
        + WEIGHT_SAMPLE * df["sample_score"]
    )

    # -------------------------------------------------------------------------
    # Distance from desired 50% WR.
    # -------------------------------------------------------------------------

    df["distance_to_50wr"] = TARGET_WR - df["mean_path_wr"]

    return df.sort_values(
        [
            "mean_path_wr",
            "research_score",
            "observations",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================


def select_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("SELECTING CONTEXT CANDIDATES")

    # -------------------------------------------------------------------------
    # Strict candidate pool.
    # -------------------------------------------------------------------------

    strict = df[
        (df["observations"] >= MIN_OBSERVATIONS)
        & (df["windows"] >= MIN_WINDOWS)
        & (df["mean_path_wr"] >= TARGET_WR)
    ].copy()

    print(f"Contexts with WR >= 50%: {len(strict):,}")

    if not strict.empty:
        return strict

    # -------------------------------------------------------------------------
    # If nothing reaches 50%, DO NOT invent a winner.
    #
    # Return the strongest contexts approaching 50%.
    # -------------------------------------------------------------------------

    print("No context reaches 50% WR.")

    print("Returning strongest near-50% contexts for further investigation.")

    fallback = df[
        (df["observations"] >= MIN_OBSERVATIONS) & (df["windows"] >= MIN_WINDOWS)
    ].copy()

    fallback["wr_gap"] = TARGET_WR - fallback["mean_path_wr"]

    fallback = fallback.sort_values(
        [
            "wr_gap",
            "research_score",
            "observations",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    return fallback.head(20)


# =============================================================================
# PRINT
# =============================================================================


def print_results(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:

    section("TOP CONTEXTS")

    columns = [
        "side",
        "hmm_state",
        "vol_bucket",
        "zscore",
        "observations",
        "windows",
        "mean_path_wr",
        "median_path_wr",
        "window_wr_std",
        "positive_window_ratio",
        "mean_mfe",
        "mean_mae",
        "research_score",
    ]

    columns = [c for c in columns if c in candidates.columns]

    if candidates.empty:
        print("No candidates.")

        return

    print(
        candidates[columns]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    section("BEST CONTEXT PER SIDE")

    for side in [
        "LONG",
        "SHORT",
    ]:
        subset = candidates[candidates["side"] == side]

        if subset.empty:
            print(f"{side}: none")

            continue

        row = subset.iloc[0]

        print(
            f"{side}: "
            f"HMM={int(row['hmm_state'])} | "
            f"VOL={row['vol_bucket']} | "
            f"Z={row['zscore']} | "
            f"WR={row['mean_path_wr']:.2%} | "
            f"N={int(row['observations']):,}"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    section("MEAN REVERSION — RESEARCH 08E")

    print("CONTEXT SELECTION")

    print("Uses Research 08D outputs only.")

    print("No HMM retraining.")

    print("No volatility recalculation.")

    print("No SL/TP optimization.")

    print("No final strategy.")

    print(
        "Goal: identify the most promising LONG/SHORT × HMM × volatility × Z contexts."
    )

    # -------------------------------------------------------------------------
    # Load.
    # -------------------------------------------------------------------------

    summary, detail = load_research_08d()

    # -------------------------------------------------------------------------
    # Build.
    # -------------------------------------------------------------------------

    master = build_master_table(
        summary,
        detail,
    )

    # -------------------------------------------------------------------------
    # Stability.
    # -------------------------------------------------------------------------

    master = calculate_stability(
        detail,
        master,
    )

    # -------------------------------------------------------------------------
    # Score.
    # -------------------------------------------------------------------------

    ranked = score_contexts(master)

    # -------------------------------------------------------------------------
    # Candidate selection.
    # -------------------------------------------------------------------------

    candidates = select_candidates(ranked)

    # -------------------------------------------------------------------------
    # Print.
    # -------------------------------------------------------------------------

    print_results(
        ranked,
        candidates,
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranked.to_csv(
        OUTPUT_ALL,
        index=False,
    )

    candidates.to_csv(
        OUTPUT_CONTEXTS,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final.
    # -------------------------------------------------------------------------

    section("RESEARCH 08E COMPLETE")

    print(f"All contexts saved:\n{OUTPUT_ALL}")

    print(f"\nCandidate contexts saved:\n{OUTPUT_CONTEXTS}")

    print()
    print("IMPORTANT:")

    print("These are context candidates, NOT optimized strategy parameters.")

    print(
        "Do NOT optimize SL/TP until the HMM + volatility + side context is selected."
    )


if __name__ == "__main__":
    main()
