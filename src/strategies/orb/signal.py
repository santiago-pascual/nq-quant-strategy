from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.strategies.base import StrategySignal

from .context import ORBContext


def generate_orb_signal(
    market_data: Mapping[str, Any],
    context: ORBContext,
) -> StrategySignal:
    """
    Generate the frozen ORB breakout signal.

    LONG
        Current bar touches/exceeds OR high.

    SHORT
        Current bar touches/breaks OR low.

    Conservative ambiguity rule
        If the same candle touches both OR boundaries,
        no trade is taken.
    """

    if not context.entry_window_open:
        return StrategySignal.FLAT

    if context.or_high is None or context.or_low is None:
        return StrategySignal.FLAT

    high = float(market_data["high"])
    low = float(market_data["low"])

    touches_high = high >= context.or_high
    touches_low = low <= context.or_low

    # Conservative same-bar ambiguity rule.
    if touches_high and touches_low:
        return StrategySignal.FLAT

    if touches_high:
        return StrategySignal.LONG

    if touches_low:
        return StrategySignal.SHORT

    return StrategySignal.FLAT
