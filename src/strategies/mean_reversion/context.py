from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MeanReversionContextBuilder:
    """
    Builds the market-data context consumed by MeanReversionStrategy.

    The builder is responsible only for extracting already-computed
    strategy features from the incoming market data.

    It does NOT calculate:

        - HMM states
        - volatility percentiles
        - z-scores

    Those values must already exist in the incoming market data.
    """

    REQUIRED_FIELDS = (
        "hmm_state",
        "vol_percentile",
        "zscore",
    )

    def build(
        self,
        market_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Build a strategy context from incoming market data.

        Existing market fields are preserved. Required Mean Reversion
        fields are added with None when unavailable.
        """

        context = dict(market_data)

        for field in self.REQUIRED_FIELDS:
            context.setdefault(field, None)

        return context
