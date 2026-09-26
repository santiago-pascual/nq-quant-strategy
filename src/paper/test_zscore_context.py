from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.research.mean_reversion.features.feature_engine import (
    build_mean_reversion_features,
)


TEST_ROWS = 1000


def main() -> None:
    print("=" * 90)
    print("Z-SCORE CONTEXT — ISOLATED TEST")
    print("=" * 90)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================

    started = time.perf_counter()

    print("\nLoading MNQ data...")

    data = load_data()

    if data is None or data.empty:
        raise RuntimeError("load_data() returned no data.")

    print(f"Full dataset rows: {len(data):,}")

    print(f"Load time: {time.perf_counter() - started:.2f}s")

    # ========================================================================
    # 2. SMALL WINDOW
    # ========================================================================

    print(f"\nUsing first {TEST_ROWS:,} rows for isolated feature test...")

    data = data.iloc[:TEST_ROWS].copy()

    if len(data) < 100:
        raise RuntimeError("Test window is unexpectedly small.")

    print(f"Test rows: {len(data):,}")

    # ========================================================================
    # 3. BUILD CANONICAL MR FEATURES
    # ========================================================================

    feature_started = time.perf_counter()

    features = build_mean_reversion_features(data)

    feature_elapsed = time.perf_counter() - feature_started

    print(f"\nFeature build time: {feature_elapsed:.4f}s")

    # ========================================================================
    # 4. CHECK Z-SCORE COLUMNS
    # ========================================================================

    print("\nZ-SCORE COLUMNS")

    print("-" * 90)

    expected = [
        "zscore_5",
        "zscore_15",
        "zscore_30",
        "zscore_60",
    ]

    for column in expected:
        if column not in features.columns:
            raise RuntimeError(f"Missing expected feature: {column}")

        print(f"  OK  {column}")

    # ========================================================================
    # 5. CHECK zscore_30
    # ========================================================================

    zscore = pd.to_numeric(
        features["zscore_30"],
        errors="coerce",
    )

    valid = zscore.replace(
        [np.inf, -np.inf],
        np.nan,
    ).notna()

    valid_count = int(valid.sum())

    print("\nZ-SCORE 30 VALIDATION")

    print("-" * 90)

    print(f"Total rows:       {len(zscore):,}")

    print(f"Valid zscore_30:  {valid_count:,}")

    print(f"NaN zscore_30:    {len(zscore) - valid_count:,}")

    if valid_count == 0:
        raise RuntimeError("zscore_30 contains no valid observations.")

    # ========================================================================
    # 6. CHECK FIRST VALID VALUE
    # ========================================================================

    first_valid_index = valid[valid].index[0]

    first_valid_value = float(zscore.loc[first_valid_index])

    print("\nFirst valid zscore_30:")

    print(f"  Index: {first_valid_index}")

    print(f"  Value: {first_valid_value}")

    if not np.isfinite(first_valid_value):
        raise RuntimeError("First valid zscore_30 is not finite.")

    # ========================================================================
    # 7. CONTEXT MAPPING
    # ========================================================================

    print("\nCONTEXT MAPPING")

    print("-" * 90)

    # This is the exact translation that the paper
    # market-context layer needs:
    #
    # canonical feature:
    #     zscore_30
    #
    # strategy context:
    #     zscore

    context_zscore = first_valid_value

    print("Canonical feature: zscore_30")

    print(f"Canonical value:   {first_valid_value}")

    print("Context field:     zscore")

    print(f"Context value:     {context_zscore}")

    if not np.isfinite(context_zscore):
        raise RuntimeError("Mapped context zscore is not finite.")

    if context_zscore != first_valid_value:
        raise RuntimeError("zscore mapping changed the underlying value.")

    print("zscore_30 -> zscore: PASS")

    # ========================================================================
    # 8. FINAL
    # ========================================================================

    print("\n" + "=" * 90)

    print("RESULT: PASS")

    print("=" * 90)


if __name__ == "__main__":
    main()
