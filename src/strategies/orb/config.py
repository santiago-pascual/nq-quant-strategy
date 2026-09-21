from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ORBEntryMode(Enum):
    TOUCH = "touch"


class ORBExitReason(Enum):
    TARGET = "target"
    STOP = "stop"
    RTH_CLOSE = "rth_close"


@dataclass(frozen=True)
class ORBConfig:
    """
    Frozen Opening Range Breakout configuration.

    This configuration reproduces the validated ORB baseline:

        Opening Range: 09:30 -> 10:00 ET
        Entry window:  10:00 -> 11:00 ET
        Risk/Reward:   1 : 2
        One trade/day
        Entry:         breakout touch
        Stop:          opposite side of opening range
        Exit:          target, stop, or RTH close
    """

    name: str = "ORB"
    version: str = "1.0.0"

    # Session times in America/New_York.
    rth_start_hour: int = 9
    rth_start_minute: int = 30

    opening_range_end_hour: int = 10
    opening_range_end_minute: int = 0

    entry_cutoff_hour: int = 11
    entry_cutoff_minute: int = 0

    rth_end_hour: int = 16
    rth_end_minute: int = 0

    # Frozen execution model.
    entry_mode: ORBEntryMode = ORBEntryMode.TOUCH

    # Frozen risk/reward.
    rr: float = 2.0

    # Exactly one trade per session.
    max_trades_per_day: int = 1

    def __post_init__(self) -> None:
        if self.rr <= 0:
            raise ValueError("ORB rr must be positive.")

        if self.max_trades_per_day != 1:
            raise ValueError(
                "Frozen ORB configuration requires exactly one trade per day."
            )

        if self.entry_mode is not ORBEntryMode.TOUCH:
            raise ValueError(
                "Only TOUCH execution is currently supported by the frozen ORB."
            )
