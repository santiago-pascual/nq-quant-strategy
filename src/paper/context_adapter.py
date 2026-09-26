from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.paper.market_context import CausalMarketContext


class PaperMarketContextAdapter:
    """
    Adapts raw OHLCV market data to the causal market context required
    by the paper-trading engine.

    Design goals
    ------------
    - One bar enters the context at a time.
    - No future bars are visible.
    - The context is stateful.
    - The returned mapping contains the original market data plus
      causal features/context.
    - No generated research CSV/state files are used.
    """

    REQUIRED_COLUMNS = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def __init__(
        self,
        context: CausalMarketContext | None = None,
    ) -> None:
        self.context = context or CausalMarketContext()

    def update(
        self,
        market_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Consume exactly one OHLCV bar and return enriched market data.
        """

        self._validate_market_data(market_data)

        timestamp = pd.Timestamp(market_data["timestamp"])

        row = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "open": float(market_data["open"]),
                    "high": float(market_data["high"]),
                    "low": float(market_data["low"]),
                    "close": float(market_data["close"]),
                    "volume": float(market_data["volume"]),
                }
            ]
        )

        enriched = self.context.update(row)

        result = dict(market_data)

        if enriched is None:
            return result

        if isinstance(enriched, pd.Series):
            context_data = enriched.to_dict()
        elif isinstance(enriched, Mapping):
            context_data = dict(enriched)
        else:
            raise TypeError(
                "CausalMarketContext.update() must return a "
                "Mapping, pandas.Series, or None."
            )

        result.update(context_data)

        return result

    @classmethod
    def _validate_market_data(
        cls,
        market_data: Mapping[str, Any],
    ) -> None:
        missing = [
            column for column in cls.REQUIRED_COLUMNS if column not in market_data
        ]

        if missing:
            raise ValueError(
                "Missing required market-data columns: " + ", ".join(missing)
            )

        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
        ):
            value = market_data[column]

            if value is None:
                raise ValueError(f"Market-data field '{column}' cannot be None.")

            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Market-data field '{column}' must be numeric."
                ) from exc

            if not pd.notna(value):
                raise ValueError(f"Market-data field '{column}' must be finite.")
