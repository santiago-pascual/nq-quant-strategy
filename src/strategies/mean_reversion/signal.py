from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import StrategySignal

from .config import MeanReversionConfig


def qualifies_hmm(
    *,
    hmm_state: Any,
    config: MeanReversionConfig,
) -> bool:
    """Return True when the HMM state matches the frozen candidate."""

    if isinstance(hmm_state, bool):
        return False

    if not isinstance(hmm_state, int):
        return False

    return hmm_state == config.hmm_state


def qualifies_volatility(
    *,
    volatility_percentile: Any,
    config: MeanReversionConfig,
) -> bool:
    """
    Return True when volatility belongs to the frozen percentile bucket.

    Lower bound is inclusive.
    Upper bound is exclusive.

    Example:
        20 <= vol < 40
    """

    if volatility_percentile is None:
        return False

    try:
        volatility = float(volatility_percentile)
    except (TypeError, ValueError):
        return False

    if not volatility == volatility:
        return False

    return config.volatility_low <= volatility < config.volatility_high


def qualifies_zscore(
    *,
    zscore: Any,
    config: MeanReversionConfig,
) -> bool:
    """
    Return True when the z-score crosses the frozen threshold.

    LONG:
        zscore <= -threshold

    SHORT:
        zscore >= +threshold
    """

    if zscore is None:
        return False

    try:
        value = float(zscore)
    except (TypeError, ValueError):
        return False

    if not value == value:
        return False

    threshold = float(config.zscore_threshold)

    if config.side == "LONG":
        return value <= -threshold

    if config.side == "SHORT":
        return value >= threshold

    raise ValueError(f"Unsupported Mean Reversion side: {config.side!r}")


def qualifies(
    market_data: Mapping[str, Any],
    config: MeanReversionConfig,
) -> bool:
    """
    Return True when all frozen Mean Reversion entry conditions are satisfied.
    """

    return (
        qualifies_hmm(
            hmm_state=market_data.get("hmm_state"),
            config=config,
        )
        and qualifies_volatility(
            volatility_percentile=market_data.get("vol_percentile"),
            config=config,
        )
        and qualifies_zscore(
            zscore=market_data.get("zscore"),
            config=config,
        )
    )


def generate_signal(
    market_data: Mapping[str, Any],
    config: MeanReversionConfig,
) -> StrategySignal:
    """Generate the frozen Mean Reversion directional signal."""

    if not qualifies(market_data, config):
        return StrategySignal.FLAT

    if config.side == "LONG":
        return StrategySignal.LONG

    if config.side == "SHORT":
        return StrategySignal.SHORT

    raise ValueError(f"Unsupported Mean Reversion side: {config.side!r}")
