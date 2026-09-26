from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.paper.market_context import (
    HMM_FEATURES,
    PaperMarketContextEngine,
)


# ============================================================================
# PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# ============================================================================
# REQUIRED CONTEXT
# ============================================================================

REQUIRED_CONTEXT_FIELDS = [
    "timestamp",
    "timestamp ET",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "hmm_state",
    "vol_percentile",
    "zscore",
    "realized_vol_30",
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
]


# ============================================================================
# HELPERS
# ============================================================================


def validate_features(
    features: pd.DataFrame,
) -> None:

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise RuntimeError(
            "PaperMarketContextEngine.features is not a pandas DataFrame."
        )

    if features.empty:
        raise RuntimeError(
            "PaperMarketContextEngine produced an empty feature DataFrame."
        )

    required_columns = [
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "realized_vol_30",
    ]

    missing = [column for column in required_columns if column not in features.columns]

    if missing:
        raise RuntimeError(
            "Feature dataframe is missing "
            "required columns:\n" + "\n".join(f"  - {column}" for column in missing)
        )


def validate_context(
    context: dict,
) -> None:

    missing = [key for key in REQUIRED_CONTEXT_FIELDS if key not in context]

    if missing:
        raise RuntimeError(
            "Required market-context fields "
            "are missing:\n" + "\n".join(f"  - {key}" for key in missing)
        )

    required_non_null = [
        "timestamp",
        "timestamp ET",
        "hmm_state",
        "vol_percentile",
        "zscore",
        "realized_vol_30",
        "past_return_30",
        "directional_pressure_30",
        "close_location_30",
        "normalized_momentum_30",
    ]

    null_fields = [key for key in required_non_null if pd.isna(context[key])]

    if null_fields:
        raise RuntimeError(
            "Generated market context "
            "contains null values:\n" + "\n".join(f"  - {key}" for key in null_fields)
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print("=" * 90)
    print("PAPER MARKET CONTEXT — OPTIMIZED CAUSAL TEST")
    print("=" * 90)

    # ========================================================================
    # 1. LOAD RAW DATA
    # ========================================================================

    load_started = time.perf_counter()

    print("\nLoading raw MNQ...")

    raw = load_data()

    load_elapsed = time.perf_counter() - load_started

    if raw is None:
        raise RuntimeError("load_data() returned None.")

    if not isinstance(
        raw,
        pd.DataFrame,
    ):
        raise RuntimeError("load_data() did not return a pandas DataFrame.")

    if raw.empty:
        raise RuntimeError("Raw MNQ data is empty.")

    print(f"Raw rows: {len(raw):,}")

    print(f"Load time: {load_elapsed:.2f}s")

    # ========================================================================
    # 2. BUILD MARKET CONTEXT ENGINE
    # ========================================================================

    print("\nBuilding canonical features...")

    feature_started = time.perf_counter()

    market_context_engine = PaperMarketContextEngine(
        raw,
        min_train_valid=500,
    )

    feature_elapsed = time.perf_counter() - feature_started

    features = market_context_engine.features

    if features is None:
        raise RuntimeError(
            "PaperMarketContextEngine did not produce a feature DataFrame."
        )

    validate_features(features)

    print(f"Feature rows: {len(features):,}")

    print(f"Feature build time: {feature_elapsed:.2f}s")

    # ========================================================================
    # 3. HMM FEATURE CHECK
    # ========================================================================

    print("\nHMM features:")

    missing_hmm_features = []

    for feature in HMM_FEATURES:
        if feature in features.columns:
            print(f"  OK  {feature}")
        else:
            print(f"  MISSING  {feature}")

            missing_hmm_features.append(feature)

    if missing_hmm_features:
        raise RuntimeError(
            "Required HMM features "
            "are missing:\n"
            + "\n".join(f"  - {feature}" for feature in missing_hmm_features)
        )

    # ========================================================================
    # 4. FIRST HMM READY INDEX
    # ========================================================================

    first_ready = market_context_engine.find_first_hmm_ready_index()

    if first_ready is None:
        raise RuntimeError("Dataset does not contain enough valid HMM history.")

    if first_ready < 0 or first_ready >= len(features):
        raise RuntimeError(f"Invalid first HMM-ready index: {first_ready}")

    first_ready_timestamp = features.iloc[first_ready]["timestamp ET"]

    print("\nFirst HMM-ready index:")

    print(f"  Index:     {first_ready:,}")

    print(f"  Timestamp: {first_ready_timestamp}")

    # ========================================================================
    # 5. EXPLICIT CAUSAL HMM CHECK
    # ========================================================================

    print("\nCAUSAL HMM TRAINING CHECK")

    print("-" * 90)

    training_features = features.iloc[:first_ready]

    valid_training = (
        training_features[list(HMM_FEATURES)]
        .replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )
        .notna()
        .all(axis=1)
    )

    training_rows = int(valid_training.sum())

    print(f"Prediction index: {first_ready}")

    print(f"Valid rows BEFORE prediction: {training_rows}")

    print("Required rows: 500")

    if training_rows != 500:
        raise RuntimeError(
            "Causal HMM readiness check failed.\n"
            f"Expected exactly 500 valid rows "
            f"before first prediction.\n"
            f"Found: {training_rows}\n"
            f"Prediction index: {first_ready}"
        )

    print("Causal HMM training history: PASS")

    # ========================================================================
    # 6. PROCESS VALIDATION WINDOW
    # ========================================================================

    validation_bars = 25

    validation_end = min(
        first_ready + validation_bars,
        len(features),
    )

    print(f"\nProcessing first {validation_end - first_ready} HMM-ready bars...")

    processing_started = time.perf_counter()

    first_context = None

    processed_contexts = []

    for index in range(
        first_ready,
        validation_end,
    ):
        context = market_context_engine.process_bar(index)

        if context is None:
            raise RuntimeError(f"process_bar({index}) returned None.")

        if not isinstance(
            context,
            dict,
        ):
            raise RuntimeError(
                f"process_bar({index}) "
                f"returned "
                f"{type(context).__name__}, "
                "expected dict."
            )

        if first_context is None:
            first_context = context

        processed_contexts.append(context)

    processing_elapsed = time.perf_counter() - processing_started

    if first_context is None:
        raise RuntimeError("No market context was produced.")

    # ========================================================================
    # 7. FIRST VALID CONTEXT
    # ========================================================================

    validate_context(first_context)

    print("\nFIRST VALID CONTEXT")

    print("-" * 90)

    print(f"Timestamp:         {first_context['timestamp']}")

    print(f"Timestamp ET:      {first_context['timestamp ET']}")

    print(f"Open:              {first_context['open']}")

    print(f"High:              {first_context['high']}")

    print(f"Low:               {first_context['low']}")

    print(f"Close:             {first_context['close']}")

    print(f"Volume:             {first_context['volume']}")

    print(f"HMM state:          {first_context['hmm_state']}")

    print(f"Vol percentile:     {first_context['vol_percentile']}")

    print(f"Z-score:            {first_context['zscore']}")

    print(f"Realized vol30:     {first_context['realized_vol_30']}")

    print(f"Past return 30:     {first_context['past_return_30']}")

    print(f"Directional p30:   {first_context['directional_pressure_30']}")

    print(f"Close location 30:  {first_context['close_location_30']}")

    print(f"Normalized mom30:   {first_context['normalized_momentum_30']}")

    print(f"\nProcessing time: {processing_elapsed:.4f}s")

    print(f"HMM fitted:       {market_context_engine.hmm_fitted}")

    print(f"HMM train rows:   {market_context_engine.hmm_training_rows:,}")

    # ========================================================================
    # 8. HMM STATE CHECK
    # ========================================================================

    print("\nHMM RUNTIME CHECK")

    print("-" * 90)

    if not market_context_engine.hmm_fitted:
        raise RuntimeError(
            "HMM should be fitted after processing the first HMM-ready bar."
        )

    if market_context_engine.hmm_training_rows != 500:
        raise RuntimeError(
            "HMM training row count "
            "is incorrect.\n"
            f"Expected: 500\n"
            f"Actual: "
            f"{market_context_engine.hmm_training_rows}"
        )

    print("HMM fitted: PASS")

    print(f"HMM training rows: {market_context_engine.hmm_training_rows}")

    # ========================================================================
    # 9. CONTEXT CONSISTENCY
    # ========================================================================

    print("\nCONTEXT CONSISTENCY CHECK")

    print("-" * 90)

    expected_contexts = validation_end - first_ready

    if len(processed_contexts) != expected_contexts:
        raise RuntimeError(
            "Processed context count does not match the requested validation window."
        )

    timestamps = [context["timestamp"] for context in processed_contexts]

    if len(timestamps) != len(set(timestamps)):
        raise RuntimeError("Duplicate timestamps found in processed contexts.")

    print(f"Processed contexts: {len(processed_contexts):,}")

    print("Duplicate timestamps: 0")

    # ========================================================================
    # 10. REQUIRED CONTEXT
    # ========================================================================

    print("\nREQUIRED CONTEXT CHECK")

    print("-" * 90)

    for key in REQUIRED_CONTEXT_FIELDS:
        if key not in first_context:
            raise RuntimeError(f"Missing required context field: {key}")

        print(f"  OK  {key}")

    print("Required context: PASS")

    # ========================================================================
    # 11. VOLATILITY PERCENTILE CHECK
    # ========================================================================

    print("\nVOLATILITY PERCENTILE CHECK")

    print("-" * 90)

    if market_context_engine.volatility_percentile is None:
        raise RuntimeError("Volatility percentile engine is missing.")

    percentile_count = market_context_engine.volatility_percentile.count

    if percentile_count <= 0:
        raise RuntimeError("Volatility percentile has no historical observations.")

    print(f"Historical volatility observations loaded: {percentile_count:,}")

    print(f"First vol percentile: {first_context['vol_percentile']}")

    if pd.isna(first_context["vol_percentile"]):
        raise RuntimeError("First HMM-ready bar has no volatility percentile.")

    print("Volatility percentile: PASS")

    # ========================================================================
    # 12. DIAGNOSTICS
    # ========================================================================

    print("\nDIAGNOSTICS")

    print("-" * 90)

    diagnostics = market_context_engine.diagnostics()

    if not isinstance(
        diagnostics,
        dict,
    ):
        raise RuntimeError("diagnostics() did not return a dictionary.")

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    # ========================================================================
    # 13. FINAL RESULT
    # ========================================================================

    print("\n" + "=" * 90)

    print("RESULT: PASS")

    print("=" * 90)


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
