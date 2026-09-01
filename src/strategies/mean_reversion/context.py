from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MeanReversionContextBuilder:
    """
    Builds the market_data mapping consumed by MeanReversionStrategy.

    The strategy receives already-computed research features.
    This class does NOT calculate indicators, HMM states, volatility
    percentiles, or z-scores.

    Expected source fields:

        hmm_state
        vol_percentile
        zscore

    Additional fields are preserved so the context can be extended
    without changing the strategy interface.
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
        Convert incoming market data into the strategy context.

        Missing required fields are preserved as None so that the
        strategy can deterministically return FLAT rather than
        crashing on incomplete market data.
        """

        context: dict[str, Any] = dict(market_data)

        for field in self.REQUIRED_FIELDS:
            context.setdefault(field, None)

        return context
