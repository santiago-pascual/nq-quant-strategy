from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.databento_loader import load_databento_mnq
from src.feature_engine import (
    add_return_features,
    add_volatility_features,
)
from src.session_engine import add_session_information


# =============================================================================
# RESEARCH 08R
# FAILURE HYPOTHESIS DISCOVERY
# =============================================================================
#
# Frozen candidates:
#
# MRS2 = SHORT | HMM 2 | VOL 80-100 | Z 2.0 | TP 5 | SL 2 | H 5
# MRL1 = LONG  | HMM 1 | VOL 20-40  | Z 2.5 | TP 5 | SL 2 | H 20
# MRL2 = LONG  | HMM 2 | VOL 60-80  | Z 3.5 | TP 5 | SL 2 | H 2
#
# PURPOSE:
#   Search for DESCRIPTIVE differences between winners and losers
#   using information available AT ENTRY TIME.
#
# IMPORTANT:
#   This script does NOT:
#       - optimize TP
#       - optimize SL
#       - optimize horizon
#       - retrain HMM
#       - select a final filter
#       - change the frozen strategies
#
#   It only produces hypotheses.
#
# A variable is interesting only if it:
#   1. existed before entry
#   2. has enough observations
#   3. separates winners/losers
#   4. shows temporal consistency
#
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

CACHE_DIR = RESULTS_DIR / "cache"

TRADES_PATH = RESULTS_DIR / "research_08p_full_confirmation_trades.csv"

METADATA_PATH = CACHE_DIR / "research_07_event_metadata.csv"

OUTPUT_FEATURES = RESULTS_DIR / "research_08r_entry_feature_analysis.csv"

OUTPUT_WINDOWS = RESULTS_DIR / "research_08r_entry_feature_windows.csv"

OUTPUT_HYPOTHESES = RESULTS_DIR / "research_08r_failure_hypotheses.csv"

OUTPUT_DISTRIBUTIONS = RESULTS_DIR / "research_08r_entry_feature_distributions.csv"


# =============================================================================
# FROZEN CANDIDATES
# =============================================================================

CANDIDATES = {
    "MRS2": {
        "candidate_id": "C01",
        "side": "SHORT",
        "hmm_state": 2,
        "vol_bucket": "80-100",
        "zscore": 2.0,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 5,
    },
    "MRL1": {
        "candidate_id": "C02",
        "side": "LONG",
        "hmm_state": 1,
        "vol_bucket": "20-40",
        "zscore": 2.5,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 20,
    },
    "MRL2": {
        "candidate_id": "C06",
        "side": "LONG",
        "hmm_state": 2,
        "vol_bucket": "60-80",
        "zscore": 3.5,
        "tp": 5.0,
        "sl": 2.0,
        "horizon": 2,
    },
}


# =============================================================================
# HELPERS
# =============================================================================


def section(title: str):

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_mean(series):

    x = numeric(series).dropna()

    return float(x.mean()) if len(x) else np.nan


def safe_median(series):

    x = numeric(series).dropna()

    return float(x.median()) if len(x) else np.nan


def safe_std(series):

    x = numeric(series).dropna()

    return float(x.std()) if len(x) > 1 else np.nan


def safe_quantile(series, q):

    x = numeric(series).dropna()

    return float(x.quantile(q)) if len(x) else np.nan


# =============================================================================
# LOAD TRADES
# =============================================================================


def load_trades():

    section("LOADING FROZEN TRADES")

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Missing:\n{TRADES_PATH}\n\nRun Research 08P first.")

    trades = pd.read_csv(TRADES_PATH)

    required = [
        "strategy_name",
        "candidate_id",
        "event_id",
        "window",
        "side",
        "result",
        "r",
    ]

    missing = [c for c in required if c not in trades.columns]

    if missing:
        raise RuntimeError(f"Missing trade columns: {missing}")

    trades["event_id"] = pd.to_numeric(
        trades["event_id"],
        errors="raise",
    ).astype(np.int64)

    trades["window"] = pd.to_numeric(
        trades["window"],
        errors="raise",
    ).astype(np.int16)

    trades["r"] = numeric(trades["r"])

    trades["timestamp"] = pd.to_datetime(
        trades["timestamp"],
        utc=True,
        errors="coerce",
    )

    print(f"Trades loaded: {len(trades):,}")

    print("\nFrozen candidates:")

    for name, config in CANDIDATES.items():
        n = int((trades["strategy_name"] == name).sum())

        print(f"{name}: {n:,}")

    return trades


# =============================================================================
# LOAD MARKET DATA
# =============================================================================


def load_market():

    section("LOADING MARKET DATA")

    print("Using project loader:")

    print("src.databento_loader.load_databento_mnq()")

    market = load_databento_mnq()

    print(f"Rows loaded: {len(market):,}")

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    print("Building return features...")

    market = add_return_features(market)

    print("Building volatility features...")

    market = add_volatility_features(market)

    print("Building session information...")

    market = add_session_information(market)

    return market


# =============================================================================
# NORMALIZE TIMESTAMP
# =============================================================================


def normalize_timestamp(
    df,
    column,
):

    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        utc=True,
        errors="coerce",
    )

    # Force nanosecond UTC representation
    #
    # This avoids pandas merge dtype problems
    # between datetime64[us, UTC] and datetime64[ns, UTC].
    df[column] = df[column].astype("datetime64[ns, UTC]")

    return df


# =============================================================================
# IDENTIFY MARKET TIMESTAMP
# =============================================================================


def find_market_timestamp(market):

    candidates = [
        "timestamp ET",
        "timestamp",
        "ts_event",
        "datetime",
    ]

    for column in candidates:
        if column in market.columns:
            return column

    raise RuntimeError("Could not identify market timestamp column.")


# =============================================================================
# BUILD ENTRY FEATURES
# =============================================================================


def build_entry_features(
    trades,
    market,
):

    section("BUILDING ENTRY-TIME FEATURES")

    market_timestamp = find_market_timestamp(market)

    print(f"Market timestamp: {market_timestamp}")

    trades = normalize_timestamp(
        trades,
        "timestamp",
    )

    market = normalize_timestamp(
        market,
        market_timestamp,
    )

    # -------------------------------------------------------------------------
    # Only keep variables that are available at the event timestamp.
    # -------------------------------------------------------------------------

    candidate_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "realized_vol_30",
        "market_period",
    ]

    optional_columns = [c for c in candidate_columns if c in market.columns]

    market_features = market[
        [
            market_timestamp,
            *optional_columns,
        ]
    ].copy()

    market_features = market_features.sort_values(market_timestamp).drop_duplicates(
        subset=[market_timestamp],
        keep="last",
    )

    trades_sorted = trades.sort_values("timestamp").copy()

    market_features = market_features.sort_values(market_timestamp)

    print("Mapping entry-time market features...")

    merged = pd.merge_asof(
        trades_sorted,
        market_features,
        left_on="timestamp",
        right_on=market_timestamp,
        direction="backward",
        allow_exact_matches=True,
    )

    if market_timestamp != "timestamp":
        merged = merged.drop(columns=[market_timestamp])

    print(f"Feature rows: {len(merged):,}")

    return merged


# =============================================================================
# ADD DERIVED ENTRY VARIABLES
# =============================================================================


def add_derived_features(trades):

    section("BUILDING DERIVED ENTRY VARIABLES")

    df = trades.copy()

    # -------------------------------------------------------------------------
    # Candle structure
    # -------------------------------------------------------------------------

    if all(
        c in df.columns
        for c in [
            "open",
            "high",
            "low",
            "close",
        ]
    ):
        df["range"] = df["high"] - df["low"]

        df["body"] = df["close"] - df["open"]

        df["abs_body"] = df["body"].abs()

        df["upper_wick"] = df["high"] - df[
            [
                "open",
                "close",
            ]
        ].max(axis=1)

        df["lower_wick"] = (
            df[
                [
                    "open",
                    "close",
                ]
            ].min(axis=1)
            - df["low"]
        )

        df["body_to_range"] = np.where(
            df["range"] != 0,
            df["abs_body"] / df["range"],
            np.nan,
        )

        df["close_location"] = np.where(
            df["range"] != 0,
            (df["close"] - df["low"]) / df["range"],
            np.nan,
        )

    # -------------------------------------------------------------------------
    # Absolute return
    # -------------------------------------------------------------------------

    if "return" in df.columns:
        df["abs_return"] = df["return"].abs()

    # -------------------------------------------------------------------------
    # Side-adjusted return
    # -------------------------------------------------------------------------

    if "return" in df.columns:
        df["side_return"] = np.where(
            df["side"] == "LONG",
            df["return"],
            -df["return"],
        )

    # -------------------------------------------------------------------------
    # Volatility-normalized return
    # -------------------------------------------------------------------------

    if all(
        c in df.columns
        for c in [
            "return",
            "realized_vol_30",
        ]
    ):
        df["return_vol_ratio"] = df["return"] / df["realized_vol_30"].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Hour / minute
    # -------------------------------------------------------------------------

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour

        df["minute"] = df["timestamp"].dt.minute

        df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    return df


# =============================================================================
# FEATURE LIST
# =============================================================================


def get_feature_columns(df):

    excluded = {
        "strategy_name",
        "candidate_id",
        "event_id",
        "window",
        "timestamp",
        "side",
        "result",
        "r",
        "tp",
        "sl",
        "rr",
        "horizon",
        "hmm_state",
        "vol_bucket",
        "zscore",
    }

    features = []

    for column in df.columns:
        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        features.append(column)

    return features


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================


def analyze_feature(
    group,
    feature,
):

    x = group[feature]

    valid = x.notna()

    data = group.loc[valid].copy()

    if len(data) < 30:
        return None

    wins = data[data["result"] == "WIN"]

    losses = data[data["result"] == "LOSS"]

    if len(wins) < 10 or len(losses) < 10:
        return None

    win_mean = safe_mean(wins[feature])

    loss_mean = safe_mean(losses[feature])

    win_median = safe_median(wins[feature])

    loss_median = safe_median(losses[feature])

    pooled_std = safe_std(data[feature])

    if (
        not np.isfinite(win_mean)
        or not np.isfinite(loss_mean)
        or not np.isfinite(pooled_std)
        or pooled_std == 0
    ):
        effect = np.nan

    else:
        effect = (win_mean - loss_mean) / pooled_std

    # -------------------------------------------------------------------------
    # Quantile separation
    # -------------------------------------------------------------------------

    q25 = safe_quantile(
        data[feature],
        0.25,
    )

    q50 = safe_quantile(
        data[feature],
        0.50,
    )

    q75 = safe_quantile(
        data[feature],
        0.75,
    )

    # -------------------------------------------------------------------------
    # Absolute difference in medians
    # -------------------------------------------------------------------------

    median_difference = win_median - loss_median

    return {
        "feature": feature,
        "observations": len(data),
        "wins": len(wins),
        "losses": len(losses),
        "win_mean": win_mean,
        "loss_mean": loss_mean,
        "win_median": win_median,
        "loss_median": loss_median,
        "mean_difference": (win_mean - loss_mean),
        "median_difference": median_difference,
        "pooled_std": pooled_std,
        "standardized_effect": effect,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "win_p25": safe_quantile(
            wins[feature],
            0.25,
        ),
        "win_p75": safe_quantile(
            wins[feature],
            0.75,
        ),
        "loss_p25": safe_quantile(
            losses[feature],
            0.25,
        ),
        "loss_p75": safe_quantile(
            losses[feature],
            0.75,
        ),
    }


# =============================================================================
# TEMPORAL STABILITY
# =============================================================================


def analyze_feature_by_window(
    group,
    feature,
):

    rows = []

    for (
        strategy_name,
        window,
    ), window_group in group.groupby(
        [
            "strategy_name",
            "window",
        ],
        dropna=False,
    ):
        data = window_group[window_group[feature].notna()]

        wins = data[data["result"] == "WIN"]

        losses = data[data["result"] == "LOSS"]

        if len(wins) < 3 or len(losses) < 3:
            continue

        win_mean = safe_mean(wins[feature])

        loss_mean = safe_mean(losses[feature])

        pooled_std = safe_std(data[feature])

        if np.isfinite(pooled_std) and pooled_std > 0:
            effect = (win_mean - loss_mean) / pooled_std

        else:
            effect = np.nan

        rows.append(
            {
                "strategy_name": strategy_name,
                "window": window,
                "feature": feature,
                "observations": len(data),
                "wins": len(wins),
                "losses": len(losses),
                "win_mean": win_mean,
                "loss_mean": loss_mean,
                "effect": effect,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# BUILD MASTER ANALYSIS
# =============================================================================


def build_analysis(
    trades,
    features,
):

    section("ANALYZING ENTRY FEATURES")

    rows = []

    window_rows = []

    for strategy_name in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy_name].copy()

        print(f"\n{strategy_name}")

        print(f"Observations: {len(group):,}")

        for feature in features:
            result = analyze_feature(
                group,
                feature,
            )

            if result is None:
                continue

            result["strategy_name"] = strategy_name

            rows.append(result)

            window_result = analyze_feature_by_window(
                group,
                feature,
            )

            if not window_result.empty:
                window_rows.append(window_result)

    analysis = pd.DataFrame(rows)

    if window_rows:
        windows = pd.concat(
            window_rows,
            ignore_index=True,
        )

    else:
        windows = pd.DataFrame()

    return analysis, windows


# =============================================================================
# HYPOTHESIS RANKING
# =============================================================================


def build_hypotheses(
    analysis,
    windows,
):

    section("BUILDING FAILURE HYPOTHESES")

    if analysis.empty:
        return pd.DataFrame()

    hypotheses = []

    for row in analysis.itertuples(index=False):
        strategy = row.strategy_name

        feature = row.feature

        window_group = windows[
            (windows["strategy_name"] == strategy) & (windows["feature"] == feature)
        ]

        effects = numeric(window_group["effect"]).dropna()

        if len(effects):
            positive_ratio = (effects > 0).mean()

            negative_ratio = (effects < 0).mean()

            median_window_effect = float(effects.median())

            mean_window_effect = float(effects.mean())

            effect_std = float(effects.std()) if len(effects) > 1 else np.nan

        else:
            positive_ratio = np.nan
            negative_ratio = np.nan
            median_window_effect = np.nan
            mean_window_effect = np.nan
            effect_std = np.nan

        # ---------------------------------------------------------------------
        # Descriptive score ONLY.
        #
        # It is NOT an optimization objective.
        # It is simply a ranking aid for manual investigation.
        # ---------------------------------------------------------------------

        if np.isfinite(row.standardized_effect):
            effect_strength = min(
                abs(row.standardized_effect),
                3.0,
            )

        else:
            effect_strength = 0.0

        if np.isfinite(positive_ratio):
            stability = abs(positive_ratio - 0.50) * 2

        else:
            stability = 0.0

        score = effect_strength * (0.5 + 0.5 * stability)

        hypotheses.append(
            {
                "strategy_name": strategy,
                "feature": feature,
                "observations": row.observations,
                "wins": row.wins,
                "losses": row.losses,
                "win_mean": row.win_mean,
                "loss_mean": row.loss_mean,
                "win_median": row.win_median,
                "loss_median": row.loss_median,
                "standardized_effect": row.standardized_effect,
                "mean_window_effect": mean_window_effect,
                "median_window_effect": median_window_effect,
                "window_effect_std": effect_std,
                "positive_window_ratio": positive_ratio,
                "negative_window_ratio": negative_ratio,
                "descriptive_score": score,
            }
        )

    result = pd.DataFrame(hypotheses)

    if not result.empty:
        result = result.sort_values(
            [
                "strategy_name",
                "descriptive_score",
            ],
            ascending=[
                True,
                False,
            ],
        )

    return result


# =============================================================================
# PRINT TOP HYPOTHESES
# =============================================================================


def print_hypotheses(hypotheses):

    section("TOP DESCRIPTIVE FAILURE HYPOTHESES")

    if hypotheses.empty:
        print("No hypotheses generated.")

        return

    for strategy in CANDIDATES:
        group = hypotheses[hypotheses["strategy_name"] == strategy].head(15)

        print("\n" + "-" * 100)

        print(strategy)

        columns = [
            "feature",
            "observations",
            "standardized_effect",
            "mean_window_effect",
            "positive_window_ratio",
            "win_mean",
            "loss_mean",
            "win_median",
            "loss_median",
        ]

        print(group[columns].to_string(index=False))


# =============================================================================
# DISTRIBUTIONS
# =============================================================================


def build_distributions(
    trades,
    features,
):

    section("BUILDING ENTRY FEATURE DISTRIBUTIONS")

    rows = []

    for strategy in CANDIDATES:
        group = trades[trades["strategy_name"] == strategy]

        for feature in features:
            if feature not in group.columns:
                continue

            for outcome in [
                "WIN",
                "LOSS",
            ]:
                subset = group[group["result"] == outcome][feature]

                subset = numeric(subset).dropna()

                if len(subset) == 0:
                    continue

                rows.append(
                    {
                        "strategy_name": strategy,
                        "feature": feature,
                        "result": outcome,
                        "observations": len(subset),
                        "mean": subset.mean(),
                        "std": subset.std(),
                        "p05": subset.quantile(0.05),
                        "p25": subset.quantile(0.25),
                        "p50": subset.quantile(0.50),
                        "p75": subset.quantile(0.75),
                        "p95": subset.quantile(0.95),
                        "min": subset.min(),
                        "max": subset.max(),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main():

    section("MEAN REVERSION — RESEARCH 08R")

    print("FAILURE HYPOTHESIS DISCOVERY")

    print("-" * 100)

    print("MRS2 / MRL1 / MRL2 remain completely frozen.")

    print("Only entry-time information is analyzed.")

    print("No filter is selected automatically.")

    trades = load_trades()

    market = load_market()

    trades = build_entry_features(
        trades,
        market,
    )

    trades = add_derived_features(trades)

    features = get_feature_columns(trades)

    print("\nEntry features available:")

    for feature in features:
        print(f"  - {feature}")

    analysis, windows = build_analysis(
        trades,
        features,
    )

    hypotheses = build_hypotheses(
        analysis,
        windows,
    )

    distributions = build_distributions(
        trades,
        features,
    )

    print_hypotheses(hypotheses)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_csv(
        OUTPUT_FEATURES,
        index=False,
    )

    windows.to_csv(
        OUTPUT_WINDOWS,
        index=False,
    )

    hypotheses.to_csv(
        OUTPUT_HYPOTHESES,
        index=False,
    )

    distributions.to_csv(
        OUTPUT_DISTRIBUTIONS,
        index=False,
    )

    section("RESEARCH 08R COMPLETE")

    print("No strategy parameters changed.")

    print("No filters were selected.")

    print("All results are descriptive.")

    print("\nFILES SAVED:")

    print(OUTPUT_FEATURES)

    print(OUTPUT_WINDOWS)

    print(OUTPUT_HYPOTHESES)

    print(OUTPUT_DISTRIBUTIONS)

    print("\nNEXT STEP:")

    print("Inspect only hypotheses that are:")

    print("1. observable before entry")

    print("2. sufficiently sampled")

    print("3. temporally stable")

    print("4. economically plausible")

    print(
        "Then formulate a SMALL number of frozen "
        "filter hypotheses for independent OOS testing."
    )


if __name__ == "__main__":
    main()
