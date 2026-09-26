from __future__ import annotations

"""
AUTONOMOUS PAPER RUNNER
=======================

End-to-end autonomous replay over the canonical raw MNQ 1-minute dataset.

Pipeline:

    RAW DATABENTO MNQ
          ↓
    causal market context
          ↓
    MRL1 / S2R / MRS2 / ORB
          ↓
    strategy lifecycle
          ↓
    risk engine
          ↓
    portfolio conflict
          ↓
    paper execution
          ↓
    fills / positions
          ↓
    P&L / equity

IMPORTANT
---------
This runner deliberately does NOT consume:

    - historical trade CSVs
    - historical HMM-state CSVs
    - historical strategy signals
    - paper-replay outputs
    - funding simulation outputs
    - cached portfolio results

Everything is generated from raw market data during the replay.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import time

import pandas as pd

from src.databento_loader import load_databento_mnq
from src.paper.context_adapter import PaperMarketContextAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paper" / "autonomous"


@dataclass(frozen=True)
class AutonomousRunConfig:
    """
    Configuration for the autonomous raw-data replay.
    """

    output_dir: Path = RESULTS_DIR

    # Optional date restriction.
    # None = entire canonical raw dataset.
    start_timestamp: pd.Timestamp | None = None
    end_timestamp: pd.Timestamp | None = None

    # Progress reporting.
    progress_every_bars: int = 100_000

    # Raw-data validation.
    require_sorted_data: bool = True
    require_unique_timestamps: bool = True


@dataclass
class AutonomousRunStats:
    total_bars: int = 0
    processed_bars: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None

    context_ready_bars: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_bars": self.total_bars,
            "processed_bars": self.processed_bars,
            "first_timestamp": (
                self.first_timestamp.isoformat()
                if self.first_timestamp is not None
                else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat()
                if self.last_timestamp is not None
                else None
            ),
            "context_ready_bars": self.context_ready_bars,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _normalise_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def _prepare_market_data(
    dataframe: pd.DataFrame,
    config: AutonomousRunConfig,
) -> pd.DataFrame:
    """
    Validate and normalize the canonical Databento dataframe.

    No strategy state is loaded here.
    No historical result is loaded here.
    """

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Canonical MNQ dataset is missing required columns: {sorted(missing)}"
        )

    df = dataframe[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if config.require_sorted_data:
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Canonical MNQ data is not sorted by timestamp.")

    if config.require_unique_timestamps:
        duplicates = int(df["timestamp"].duplicated().sum())

        if duplicates:
            raise ValueError(
                f"Canonical MNQ data contains {duplicates} duplicate timestamps."
            )

    if config.start_timestamp is not None:
        start = _normalise_timestamp(config.start_timestamp)
        df = df.loc[df["timestamp"] >= start]

    if config.end_timestamp is not None:
        end = _normalise_timestamp(config.end_timestamp)
        df = df.loc[df["timestamp"] <= end]

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No MNQ bars remain after applying the requested date range.")

    return df


def _row_to_market_data(row: pd.Series) -> dict[str, Any]:
    """
    Convert one canonical OHLCV row into the Mapping expected by
    PaperMarketContextAdapter / PaperTradingEngine.
    """

    return {
        "timestamp": row["timestamp"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


class AutonomousPaperRunner:
    """
    Raw-data autonomous replay coordinator.

    The runner intentionally owns only:

        1. raw data loading
        2. chronological replay
        3. causal context generation
        4. forwarding bars to the paper engine

    Trading logic remains inside the strategy/paper architecture.
    """

    def __init__(
        self,
        *,
        paper_engine: Any,
        context_adapter: PaperMarketContextAdapter,
        config: AutonomousRunConfig | None = None,
    ) -> None:
        self.paper_engine = paper_engine
        self.context_adapter = context_adapter
        self.config = config or AutonomousRunConfig()

        self.stats = AutonomousRunStats()

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> AutonomousRunStats:
        """
        Replay the supplied canonical market dataframe chronologically.
        """

        df = _prepare_market_data(
            dataframe,
            self.config,
        )

        self.stats = AutonomousRunStats(
            total_bars=len(df),
            first_timestamp=df["timestamp"].iloc[0],
            last_timestamp=df["timestamp"].iloc[-1],
        )

        started = time.perf_counter()

        for index, row in df.iterrows():
            market_data = _row_to_market_data(row)

            # ---------------------------------------------------------
            # CAUSAL CONTEXT
            # ---------------------------------------------------------
            #
            # The adapter sees only the information available up to
            # this bar. No future rows are passed into it.
            #
            enriched_market_data = self.context_adapter.update(market_data)

            if self._context_is_ready(enriched_market_data):
                self.stats.context_ready_bars += 1

            # ---------------------------------------------------------
            # PAPER ENGINE
            # ---------------------------------------------------------
            #
            # The paper engine owns:
            #
            #   strategy evaluation
            #   lifecycle
            #   risk
            #   portfolio conflict
            #   execution
            #   fills
            #
            self.paper_engine.process_bar(
                enriched_market_data,
                context_index=index,
            )

            self.stats.processed_bars += 1

            if (
                self.config.progress_every_bars > 0
                and self.stats.processed_bars % self.config.progress_every_bars == 0
            ):
                self._print_progress()

        self.stats.elapsed_seconds = time.perf_counter() - started

        self._print_finished()

        return self.stats

    @staticmethod
    def _context_is_ready(
        market_data: Mapping[str, Any],
    ) -> bool:
        """
        Detect whether the causal context has become usable.

        The exact context implementation may expose different readiness
        fields, so this deliberately accepts the canonical readiness
        names used by the context builder.
        """

        if "context_ready" in market_data:
            return bool(market_data["context_ready"])

        if "ready" in market_data:
            return bool(market_data["ready"])

        required = (
            "zscore",
            "vol_percentile",
            "hmm_state",
        )

        return all(
            key in market_data and market_data[key] is not None for key in required
        )

    def _print_progress(self) -> None:
        pct = 100.0 * self.stats.processed_bars / max(self.stats.total_bars, 1)

        elapsed = max(self.stats.elapsed_seconds, 1e-9)

        # During the loop stats.elapsed_seconds is still zero, so use
        # processed bars as a lightweight progress indicator.
        print(
            f"[AUTONOMOUS PAPER] "
            f"{self.stats.processed_bars:,}/"
            f"{self.stats.total_bars:,} "
            f"({pct:6.2f}%)"
        )

    def _print_finished(self) -> None:
        print()
        print("=" * 80)
        print("AUTONOMOUS PAPER REPLAY FINISHED")
        print("=" * 80)
        print(f"Bars processed:       {self.stats.processed_bars:,}")
        print(f"First timestamp:      {self.stats.first_timestamp}")
        print(f"Last timestamp:       {self.stats.last_timestamp}")
        print(f"Context-ready bars:   {self.stats.context_ready_bars:,}")
        print(f"Elapsed seconds:      {self.stats.elapsed_seconds:.2f}")
        print("=" * 80)


def build_runner(
    *,
    paper_engine: Any,
    config: AutonomousRunConfig | None = None,
) -> AutonomousPaperRunner:
    """
    Build the autonomous runner with a completely fresh causal context.

    This is important:

        context_adapter = PaperMarketContextAdapter()

    creates new HMM/context state for this run.

    Nothing from previous executions is loaded.
    """

    context_adapter = PaperMarketContextAdapter()

    return AutonomousPaperRunner(
        paper_engine=paper_engine,
        context_adapter=context_adapter,
        config=config,
    )


def load_canonical_raw_mnq() -> pd.DataFrame:
    """
    Load ONLY the canonical raw Databento MNQ dataset.
    """

    print("=" * 80)
    print("LOADING CANONICAL RAW MNQ DATA")
    print("=" * 80)

    dataframe = load_databento_mnq()

    print(f"Raw bars loaded: {len(dataframe):,}")

    if not dataframe.empty:
        print(f"First: {dataframe['timestamp'].iloc[0]}")
        print(f"Last:  {dataframe['timestamp'].iloc[-1]}")

    print()

    return dataframe


def run_autonomous_paper(
    *,
    paper_engine: Any,
    config: AutonomousRunConfig | None = None,
) -> AutonomousRunStats:
    """
    Convenience entry point for the complete raw-data replay.
    """

    dataframe = load_canonical_raw_mnq()

    runner = build_runner(
        paper_engine=paper_engine,
        config=config,
    )

    return runner.run(dataframe)


__all__ = [
    "AutonomousPaperRunner",
    "AutonomousRunConfig",
    "AutonomousRunStats",
    "build_runner",
    "load_canonical_raw_mnq",
    "run_autonomous_paper",
]
