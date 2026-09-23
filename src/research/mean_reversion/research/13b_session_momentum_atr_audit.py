"""
13B — SESSION MOMENTUM ATR / OPENING-CANDLE AUDIT

Purpose
-------
Audit the two main reconstruction uncertainties before robustness:

1. ATR lookback period.
2. Opening candle / entry convention.

The source specifies:
    - M5
    - NY session open
    - EMA 12
    - EMA 120
    - 8 ATR stop
    - trail after +0.5R
    - one trade per session
    - no fixed target

The source does NOT specify the ATR lookback period.

This script therefore tests a predefined ATR grid without selecting a
"winner". The objective is to identify stable parameter regions.

Opening convention is explicitly audited in America/New_York time:

    Signal candle: 09:30 -> 09:35 ET
    Entry candle:  09:35 -> 09:40 ET
    Entry price:   09:35 open

This is the actual NY cash-session opening candle. It corresponds to
10:30 Argentina time during US daylight-saving time and 11:30 during
US standard time.

NO ROBUSTNESS / MONTE CARLO / WALK-FORWARD OPTIMIZATION IS PERFORMED HERE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# IMPORT BASELINE ENGINE
# =============================================================================

from src.databento_loader import load_databento_mnq
from src.session_engine import add_session_information

# Import the already audited baseline functions.
from src.research.mean_reversion.research import (
    # This import style is intentionally not used because research filenames
    # are not guaranteed to be importable as normal modules.
)
