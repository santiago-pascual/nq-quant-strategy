from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================================
# PATHS
# ======================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
FUNDED_SIMULATION_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "portfolio"
    / "22_full_system_funded_simulation.py"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ======================================================================================
# LOAD EXACT FUNDED SIMULATION MODULE
# ======================================================================================


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


module = load_module_from_path(
    "full_system_funded_simulation_audit",
    FUNDED_SIMULATION_PATH,
)


# ======================================================================================
# HEADER
# ======================================================================================

print("=" * 110)
print("ADAPTIVE ORB XFA — TRADE SLOT AUDIT")
print("=" * 110)

print(f"Funded simulation module: {FUNDED_SIMULATION_PATH}")

MAX_XFA_TRADES = module.MAX_XFA_TRADES

print(f"MAX_XFA_TRADES = {MAX_XFA_TRADES}")


# ======================================================================================
# 1. ACTUAL ENGINE SOURCE
# ======================================================================================

print()
print("=" * 110)
print("1. HISTORICAL SEQUENCE BUILD_BATCH SOURCE")
print("=" * 110)

build_batch_source = inspect.getsource(module.HistoricalSequence.build_batch)

print(build_batch_source)


print()
print("=" * 110)
print("2. XFA BATCH ENGINE SOURCE")
print("=" * 110)

xfa_source = inspect.getsource(module.run_xfa_batch)

print(xfa_source)


# ======================================================================================
# 3. SYNTHETIC TEST
# ======================================================================================
#
# Four events on one day:
#
#   +1R  = real trade
#    0R  = rejected signal
#   +1R  = real trade
#   +1R  = real trade
#
# If MAX_XFA_TRADES = 3:
#
# Correct economic behavior:
#   3 real trades are available.
#
# Potential bug:
#   the 0R rejected event consumes one slot,
#   leaving only 2 real trades.
#
# ======================================================================================

print()
print("=" * 110)
print("3. SYNTHETIC SLOT-CONSUMPTION TEST")
print("=" * 110)

synthetic_returns = np.array(
    [
        1.0,
        0.0,
        1.0,
        1.0,
    ],
    dtype=float,
)

synthetic_timestamps = pd.date_range(
    "2024-01-02 10:00:00",
    periods=len(synthetic_returns),
    freq="min",
)

synthetic_dates = pd.Series(
    pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
        ]
    )
)

synthetic_df = pd.DataFrame(
    {
        "entry_timestamp": synthetic_timestamps,
        "r_multiple": synthetic_returns,
        "date_ny": synthetic_dates,
    }
)

print("Synthetic returns:")
print(synthetic_returns)

print()
print("Synthetic timestamps:")
print(synthetic_timestamps)

print()
print("Economic real trades:")
print(int(np.count_nonzero(synthetic_returns)))


# ======================================================================================
# 4. HISTORICAL SEQUENCE
# ======================================================================================

print()
print("=" * 110)
print("4. HISTORICAL SEQUENCE")
print("=" * 110)

try:
    seq = module.HistoricalSequence(
        strategy="SLOT_AUDIT",
        df=synthetic_df,
    )

    print(f"n_trades    = {seq.n_trades}")
    print(f"unique_days = {len(seq.unique_dates)}")
    print(f"returns     = {seq.returns}")

except Exception as exc:
    print()
    print("HISTORICAL SEQUENCE CONSTRUCTION FAILED")
    print(f"{type(exc).__name__}: {exc}")
    raise


# ======================================================================================
# 5. BUILD BATCH
# ======================================================================================

print()
print("=" * 110)
print("5. BUILD_BATCH OUTPUT")
print("=" * 110)

try:
    batch_days = np.array([0], dtype=np.int64)

    batch = seq.build_batch(
        start_days=batch_days,
        max_trades=MAX_XFA_TRADES,
    )

    print(f"type(batch) = {type(batch)}")
    print()

    if isinstance(batch, tuple):
        for i, item in enumerate(batch):
            print(f"tuple[{i}] type={type(item)}")

            if isinstance(item, np.ndarray):
                print(f"shape={item.shape}, dtype={item.dtype}")
                print(item)
            else:
                print(repr(item))

            print()

    elif isinstance(batch, dict):
        for key, value in batch.items():
            print(f"{key}:")
            print(f"  type={type(value)}")

            if isinstance(value, np.ndarray):
                print(f"  shape={value.shape}, dtype={value.dtype}")
                print(f"  values={value}")
            else:
                print(f"  value={value!r}")

            print()

    else:
        print(batch)

except Exception as exc:
    print()
    print("BUILD_BATCH TEST FAILED")
    print(f"{type(exc).__name__}: {exc}")


# ======================================================================================
# 6. SOURCE-LEVEL CHECK
# ======================================================================================

print()
print("=" * 110)
print("6. SOURCE-LEVEL SLOT LOGIC")
print("=" * 110)

combined_source = build_batch_source + "\n" + xfa_source

keywords = [
    "trade_count",
    "n_trades",
    "MAX_XFA_TRADES",
    "max_trades",
    "returns",
    "path_returns",
    "r_multiple",
    "day_ids",
]

for keyword in keywords:
    print(f"{keyword:<20} occurrences={combined_source.count(keyword)}")


# ======================================================================================
# 7. DIRECT ZERO-RETURN QUESTION
# ======================================================================================

print()
print("=" * 110)
print("7. ZERO-RETURN EVENT QUESTION")
print("=" * 110)

print(
    """
We need to establish one specific fact:

Does an r_multiple == 0.0 event consume one of the
MAX_XFA_TRADES slots?

Expected economic behavior for a rejected adaptive signal:

    rejected signal
        -> no order
        -> no position
        -> no P&L
        -> NO XFA TRADE SLOT CONSUMED

If build_batch / run_xfa_batch counts every row regardless
of r_multiple, then the previous adaptive replay is structurally
different from live execution.

If zero-return rows are excluded from the trade-count mechanism,
the previous replay's slot handling is valid.
"""
)


# ======================================================================================
# 8. AUDIT GATE
# ======================================================================================

print()
print("=" * 110)
print("8. AUDIT GATE")
print("=" * 110)

print(
    """
NO PRODUCTION CHANGES.

This script only inspects the existing XFA engine.

After this output we will determine whether:

    A) current adaptive XFA replay is slot-correct

or

    B) a corrected XFA replay is required where rejected ORB
       signals preserve calendar/day structure but do not consume
       MAX_XFA_TRADES.
"""
)

print()
print("=" * 110)
print("ADAPTIVE ORB XFA — TRADE SLOT AUDIT COMPLETE")
print("=" * 110)
