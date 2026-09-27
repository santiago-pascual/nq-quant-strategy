"""Raw-data reconstruction helpers for validating frozen research artifacts.

Benchmark CSV files are accepted only by comparison helpers.  Signal context
and HMM states are always rebuilt from the Databento bars and canonical code.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.databento_loader import load_databento_mnq
from src.feature_engine import add_return_features, add_volatility_features
from src.session_engine import add_session_information


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MR_HMM_ORACLE = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "cache"
    / "research_08b_causal_hmm_states.csv"
)


@dataclass(frozen=True)
class HMMParityReport:
    """Identity comparison for one causally reconstructed HMM window."""

    window: int
    reconstructed_rows: int
    oracle_rows: int
    common_timestamps: int
    missing_canonical_timestamps: int
    missing_reconstructed_timestamps: int
    mismatches: int
    mismatch_percentage: float
    first_mismatch: pd.Timestamp | None
    last_mismatch: pd.Timestamp | None
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return (
            self.reconstructed_rows == self.oracle_rows
            and self.common_timestamps == self.oracle_rows
            and self.missing_canonical_timestamps == 0
            and self.missing_reconstructed_timestamps == 0
            and self.mismatches == 0
        )


def load_canonical_market() -> pd.DataFrame:
    """Build the frozen feature universe from raw bars without target labels."""
    market = load_databento_mnq()
    market = add_return_features(market)
    market = add_volatility_features(market)
    return add_session_information(market)


def reconstruct_mr_hmm_window(window: int) -> pd.DataFrame:
    """Rebuild exactly one Research 08B causal HMM window from raw bars."""
    if window < 1 or window > 22:
        raise ValueError("window must be in the inclusive range 1..22")

    research_07 = import_module(
        "src.research.mean_reversion.research.07_path_dependent_long_short"
    )
    research_08b = import_module(
        "src.research.mean_reversion.research.08b_hmm_causal_clean"
    )

    market = load_canonical_market()
    rth = research_07.prepare_rth(market)
    rth = research_07.build_features(rth)
    windows = research_07.build_oos_windows(rth, research_07.N_OOS_WINDOWS)
    events = research_07.build_event_metadata(rth, windows)

    features = research_08b.build_features(market)
    selected = events.loc[events["window"].eq(window)].copy()
    if selected.empty:
        raise RuntimeError(f"Window {window} has no valid MR events.")

    oos_start = selected["timestamp"].min()
    oos_end = selected["timestamp"].max()
    train = features.loc[features["canonical_timestamp"] < oos_start].copy()
    oos = features.loc[
        features["canonical_timestamp"].between(oos_start, oos_end, inclusive="both")
    ].copy()

    if len(research_08b.calculate_valid_hmm_rows(train)) < research_08b.MIN_TRAIN_VALID:
        raise RuntimeError(f"Window {window} has insufficient causal HMM history.")

    model = research_08b.VolatilityRegimeModel(
        n_states=research_08b.N_STATES,
        random_state=research_08b.RANDOM_STATE,
        n_iter=research_08b.N_ITER,
    )
    model.fit(train)
    states = model.predict_states(oos)
    predicted = pd.DataFrame(
        {
            "timestamp": oos.loc[states.index, "canonical_timestamp"].to_numpy(),
            "window": window,
            "hmm_state": states.to_numpy(dtype=np.int8),
        }
    )
    predicted["timestamp"] = pd.to_datetime(predicted["timestamp"], utc=True)

    event_map = selected[["timestamp", "window"]].copy()
    event_map["timestamp"] = pd.to_datetime(event_map["timestamp"], utc=True)
    return event_map.merge(
        predicted,
        on=["window", "timestamp"],
        how="left",
        validate="one_to_one",
    )


def compare_hmm_window(
    reconstructed: pd.DataFrame,
    oracle_path: Path = MR_HMM_ORACLE,
) -> HMMParityReport:
    """Compare HMM states by timestamp; the oracle never feeds reconstruction."""
    started = perf_counter()
    required = {"timestamp", "window", "hmm_state"}
    missing = required.difference(reconstructed.columns)
    if missing:
        raise KeyError(f"Reconstructed states missing columns: {sorted(missing)}")

    window = int(reconstructed["window"].iloc[0])
    oracle = pd.read_csv(oracle_path)
    oracle = oracle.loc[oracle["window"].eq(window), list(required)].copy()
    for frame in (reconstructed, oracle):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    merged = reconstructed.merge(
        oracle,
        on="timestamp",
        how="inner",
        suffixes=("_reconstructed", "_oracle"),
    )
    mismatched = merged.loc[
        merged["hmm_state_reconstructed"].ne(merged["hmm_state_oracle"])
    ]
    timestamps = mismatched["timestamp"]
    reconstructed_timestamps = set(reconstructed["timestamp"])
    oracle_timestamps = set(oracle["timestamp"])
    return HMMParityReport(
        window=window,
        reconstructed_rows=len(reconstructed),
        oracle_rows=len(oracle),
        common_timestamps=len(merged),
        missing_canonical_timestamps=len(oracle_timestamps - reconstructed_timestamps),
        missing_reconstructed_timestamps=len(
            reconstructed_timestamps - oracle_timestamps
        ),
        mismatches=len(mismatched),
        mismatch_percentage=(100.0 * len(mismatched) / len(merged))
        if len(merged)
        else 0.0,
        first_mismatch=timestamps.min() if not timestamps.empty else None,
        last_mismatch=timestamps.max() if not timestamps.empty else None,
        elapsed_seconds=perf_counter() - started,
    )


def print_hmm_parity_summary(report: HMMParityReport) -> None:
    """Print a stable, machine-readable Window 1 validation summary."""
    print(f"CANONICAL_ROWS={report.oracle_rows}")
    print(f"RECONSTRUCTED_ROWS={report.reconstructed_rows}")
    print(f"COMMON_TIMESTAMPS={report.common_timestamps}")
    print(f"MISSING_CANONICAL_TIMESTAMPS={report.missing_canonical_timestamps}")
    print(f"MISSING_RECONSTRUCTED_TIMESTAMPS={report.missing_reconstructed_timestamps}")
    print(f"STATE_MISMATCHES={report.mismatches}")
    print(f"MISMATCH_PERCENTAGE={report.mismatch_percentage:.12f}")
    print(f"FIRST_MISMATCH_TIMESTAMP={report.first_mismatch}")
    print(f"LAST_MISMATCH_TIMESTAMP={report.last_mismatch}")
    print(f"PARITY={'PASS' if report.passed else 'FAIL'}")


if __name__ == "__main__":
    report = compare_hmm_window(reconstruct_mr_hmm_window(window=1))
    print_hmm_parity_summary(report)
