from __future__ import annotations

from typing import Protocol

import pandas as pd


class HMMStateProvider(Protocol):
    """Stable regime-state boundary shared by Research Replay and Paper context."""

    def state_for(
        self,
        timestamp: pd.Timestamp,
        features: pd.DataFrame | None,
    ) -> int | None: ...
