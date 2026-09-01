from __future__ import annotations

import pandas as pd

from src.research.mean_reversion.features.returns import (
    add_return_features,
)
from src.research.mean_reversion.features.volatility import (
    add_volatility_features,
)
from src.research.mean_reversion.features.vwap import (
    add_vwap_features,
)
from src.research.mean_reversion.features.zscore import (
    add_zscore_features,
)
from src.research.mean_reversion.features.ou_process import (
    add_ou_features,
)


# ============================================================
# MEAN REVERSION — FEATURE ENGINE
# ============================================================
#
# PURPOSE
# -------
# Central orchestration layer for the Mean Reversion research
# feature pipeline.
#
# The engine does NOT create new mathematical features.
# It only executes the already-tested feature modules in a
# deterministic order.
#
# PIPELINE
# --------
#
#                 RAW OHLCV
#                     |
#                     v
#                  RETURNS
#                     |
#                     v
#                 VOLATILITY
#                     |
#                     v
#                    VWAP
#                     |
#                     v
#                  Z-SCORE
#                     |
#                     v
#                     OU
#                     |
#                     v
#              COMPLETE FEATURES
#
# IMPORTANT
# ---------
# No:
#
#   - entry logic
#   - exit logic
#   - stop loss
#   - take profit
#   - optimization
#   - parameter fitting
#   - signal generation
#
# belongs here.
#
# ============================================================


def build_mean_reversion_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete Mean Reversion feature dataframe.

    Each feature module is called exactly once and receives the
    output of the previous module.

    The original row count must be preserved.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    original_index = result.index.copy()

    # --------------------------------------------------------
    # 1. RETURNS
    # --------------------------------------------------------

    result = add_return_features(result)

    # --------------------------------------------------------
    # 2. VOLATILITY
    # --------------------------------------------------------
    #
    # Volatility depends on log returns created above.
    #

    result = add_volatility_features(result)

    # --------------------------------------------------------
    # 3. VWAP
    # --------------------------------------------------------
    #
    # VWAP requires OHLCV and session information.
    #
    # Normalized VWAP distance also requires atr_30, which is
    # provided by the volatility layer.
    #

    result = add_vwap_features(result)

    # --------------------------------------------------------
    # 4. Z-SCORE
    # --------------------------------------------------------

    result = add_zscore_features(result)

    # --------------------------------------------------------
    # 5. OU
    # --------------------------------------------------------

    result = add_ou_features(result)

    # --------------------------------------------------------
    # INTEGRITY CHECK
    # --------------------------------------------------------

    if len(result) != len(df):
        raise RuntimeError("Feature engine changed the row count.")

    if not result.index.equals(original_index):
        raise RuntimeError("Feature engine changed the dataframe index.")

    return result
