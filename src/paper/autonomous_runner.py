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
from src.paper.market_context import (
    PRECOMPUTED_CONTEXT_FEATURES,
    build_causal_context_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "paper" / "autonomous"


def is_standard_rth_close_bar(
    timestamp: Any,
    *,
    rth_start_minute: int = 9 * 60 + 30,
    rth_end_minute: int = 16 * 60,
) -> bool:
    """Return whether this minute bar ends at the configured standard close."""
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        raise ValueError("RTH close checks require timezone-aware timestamps.")
    local = stamp.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    return rth_start_minute <= minute < rth_end_minute and minute == rth_end_minute - 1


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

    # Optional equivalent history fast-forward for bounded pseudo-live runs.
    # The default runner behavior remains one-bar-at-a-time warmup.
    bootstrap_causal_history: bool = False

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
    feature_construction_seconds: float = 0.0
    bootstrap_seconds: float = 0.0

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
            "feature_construction_seconds": self.feature_construction_seconds,
            "bootstrap_seconds": self.bootstrap_seconds,
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
    *,
    apply_range: bool = True,
) -> pd.DataFrame:
    """
    Validate and normalize the canonical Databento dataframe.

    No strategy state is loaded here.
    No historical result is loaded here.
    """

    timestamp_column = (
        "timestamp"
        if "timestamp" in dataframe.columns
        else "timestamp ET"
        if "timestamp ET" in dataframe.columns
        else None
    )
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(dataframe.columns)

    if missing or timestamp_column is None:
        if timestamp_column is None:
            missing.add("timestamp or timestamp ET")
        raise ValueError(
            f"Canonical MNQ dataset is missing required columns: {sorted(missing)}"
        )

    df = dataframe[
        [timestamp_column, "open", "high", "low", "close", "volume"]
    ].copy()
    df = df.rename(columns={timestamp_column: "timestamp"})

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

    if apply_range and config.start_timestamp is not None:
        start = _normalise_timestamp(config.start_timestamp)
        df = df.loc[df["timestamp"] >= start]

    if apply_range and config.end_timestamp is not None:
        end = _normalise_timestamp(config.end_timestamp)
        df = df.loc[df["timestamp"] <= end]

    df = df.reset_index(drop=True)

    if df.empty and apply_range:
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


def identify_final_rth_bars(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    rth_start_minute: int = 9 * 60 + 30,
    rth_end_minute: int = 16 * 60,
) -> frozenset[pd.Timestamp]:
    """Identify standard 16:00 ET closes from each bar's own timestamp."""
    if timestamp_column not in dataframe:
        raise KeyError(f"Missing session timestamp column: {timestamp_column}")
    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        utc=True,
        errors="raise",
    )
    return frozenset(
        stamp
        for stamp in timestamps
        if is_standard_rth_close_bar(
            stamp,
            rth_start_minute=rth_start_minute,
            rth_end_minute=rth_end_minute,
        )
    )


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
        engine_context = getattr(paper_engine, "context_adapter", None)
        if engine_context is None:
            paper_engine.context_adapter = context_adapter
            engine_context = context_adapter
        if engine_context is not context_adapter:
            raise ValueError(
                "Autonomous runner and paper engine must share one context adapter."
            )
        if hasattr(paper_engine, "enable_simulated_fills"):
            paper_engine.enable_simulated_fills()
        self.context_adapter = engine_context
        self.config = config or AutonomousRunConfig()

        self.stats = AutonomousRunStats()
        self.daily_equity: list[dict[str, Any]] = []

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> AutonomousRunStats:
        """
        Replay the supplied canonical market dataframe chronologically.
        """

        full_df = _prepare_market_data(
            dataframe,
            self.config,
            apply_range=False,
        )
        if full_df.empty:
            raise ValueError("Canonical MNQ dataset is empty.")

        context = self.context_adapter.context

        start = (
            _normalise_timestamp(self.config.start_timestamp)
            if self.config.start_timestamp is not None
            else None
        )
        end = (
            _normalise_timestamp(self.config.end_timestamp)
            if self.config.end_timestamp is not None
            else None
        )
        if end is not None:
            full_df = full_df.loc[full_df["timestamp"] <= end].reset_index(drop=True)
        local_times = pd.to_datetime(full_df["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
        minute_of_day = local_times.dt.hour * 60 + local_times.dt.minute
        if not ((minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)).any():
            raise ValueError("Canonical MNQ dataset contains no regular-hours bars.")
        range_mask = pd.Series(True, index=full_df.index)
        if start is not None:
            range_mask &= full_df["timestamp"] >= start
        if end is not None:
            range_mask &= full_df["timestamp"] <= end
        df = full_df.loc[range_mask].reset_index(drop=True)
        if df.empty:
            raise ValueError("No MNQ bars remain after applying the requested date range.")

        selected_start = df["timestamp"].iloc[0]
        primed_through = context.last_timestamp if context.bars_seen else None
        if primed_through is not None and primed_through >= selected_start:
            raise ValueError(
                "Primed market context must end strictly before replay starts."
            )

        bootstrap_index = 0
        feature_construction_seconds = 0.0
        bootstrap_seconds = 0.0
        precomputed_feature_frame: pd.DataFrame | None = None
        if (
            self.config.bootstrap_causal_history
            and start is not None
            and primed_through is None
        ):
            feature_started = time.perf_counter()
            precomputed_feature_frame = build_causal_context_features(full_df)
            feature_construction_seconds = time.perf_counter() - feature_started
            directional_ready = (
                precomputed_feature_frame["log_return"].rolling(30).count() == 30
            )
            precomputed_feature_frame.loc[
                ~directional_ready, "close_location_30"
            ] = float("nan")
            bootstrap_rows = full_df.loc[full_df["timestamp"] < selected_start]
            if bootstrap_rows.empty:
                raise ValueError(
                    "Causal history bootstrap requires at least one bar before the replay start."
                )
            bootstrap_started = time.perf_counter()
            primed_through = context.bootstrap_causal_history(
                bootstrap_rows,
                precomputed_features=precomputed_feature_frame.iloc[:len(bootstrap_rows)],
            )
            bootstrap_seconds = time.perf_counter() - bootstrap_started
            if primed_through >= selected_start:
                raise RuntimeError("Causal bootstrap crossed the strategy start boundary.")
            bootstrap_index = int(
                full_df["timestamp"].searchsorted(selected_start, side="left")
            )

        self.stats = AutonomousRunStats(
            total_bars=len(df),
            first_timestamp=df["timestamp"].iloc[0],
            last_timestamp=df["timestamp"].iloc[-1],
            feature_construction_seconds=feature_construction_seconds,
            bootstrap_seconds=bootstrap_seconds,
        )

        started = time.perf_counter()
        self.daily_equity = []

        feature_rows = (
            precomputed_feature_frame.loc[:, PRECOMPUTED_CONTEXT_FEATURES]
            .iloc[bootstrap_index:]
            .itertuples(index=False, name=None)
            if precomputed_feature_frame is not None
            else None
        )
        for index, timestamp, open_, high, low, close, volume in full_df.iloc[
            bootstrap_index:
        ].itertuples(index=True, name=None):
            if end is not None and timestamp > end:
                break
            market_data = {
                "timestamp": timestamp,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
            market_data["is_final_rth_bar"] = is_standard_rth_close_bar(
                timestamp
            )

            if timestamp < selected_start:
                if primed_through is not None and timestamp <= primed_through:
                    continue
                self.context_adapter.update(market_data)
                continue

            if feature_rows is not None:
                self.context_adapter.set_next_precomputed_features(
                    dict(zip(PRECOMPUTED_CONTEXT_FEATURES, next(feature_rows)))
                )

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
            result = self.paper_engine.process_bar(
                market_data,
                context_index=index,
            )
            local_day = timestamp.tz_convert("America/New_York").date()
            if (
                not self.daily_equity
                or self.daily_equity[-1]["date"] != local_day.isoformat()
            ):
                self.daily_equity.append(
                    {
                        "date": local_day.isoformat(),
                        "equity": result.account_equity,
                        "realized_pnl": result.realized_pnl,
                        "commissions": result.commissions,
                    }
                )
            else:
                self.daily_equity[-1].update(
                    equity=result.account_equity,
                    realized_pnl=result.realized_pnl,
                    commissions=result.commissions,
                )
            enriched_market_data = getattr(
                self.paper_engine,
                "last_market_data",
                None,
            ) or market_data
            if self._context_is_ready(enriched_market_data):
                self.stats.context_ready_bars += 1

            self.stats.processed_bars += 1

            if (
                self.config.progress_every_bars > 0
                and self.stats.processed_bars % self.config.progress_every_bars == 0
            ):
                self._print_progress()

        peak_equity = (
            float(getattr(self.paper_engine, "config").initial_equity)
            if hasattr(self.paper_engine, "config")
            else 0.0
        )
        for daily in self.daily_equity:
            peak_equity = max(peak_equity, float(daily["equity"]))
            daily["peak_equity"] = peak_equity
            daily["drawdown"] = float(daily["equity"]) - peak_equity
            daily["drawdown_pct"] = (
                100.0 * daily["drawdown"] / peak_equity if peak_equity else 0.0
            )

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

        context_adapter = (
            getattr(paper_engine, "context_adapter", None)
            or PaperMarketContextAdapter()
        )

    creates new HMM/context state for this run.

    Nothing from previous executions is loaded.
    """

    context_adapter = PaperMarketContextAdapter()

    return AutonomousPaperRunner(
        paper_engine=paper_engine,
        context_adapter=context_adapter,
        config=config,
    )


def load_canonical_raw_mnq(*, include_contract_metadata: bool = False) -> pd.DataFrame:
    """
    Load raw Databento MNQ bars with a validated UTC ``timestamp`` column.
    """

    print("=" * 80)
    print("LOADING CANONICAL RAW MNQ DATA")
    print("=" * 80)

    dataframe = (
        load_databento_mnq(include_instrument_id=True)
        if include_contract_metadata
        else load_databento_mnq()
    )
    dataframe = _canonicalize_raw_mnq(
        dataframe, validate_contract_mapping=include_contract_metadata
    )

    print(f"Raw bars loaded: {len(dataframe):,}")

    if not dataframe.empty:
        print(f"First: {dataframe['timestamp'].iloc[0]}")
        print(f"Last:  {dataframe['timestamp'].iloc[-1]}")

    print()

    return dataframe


def _canonicalize_raw_mnq(
    dataframe: pd.DataFrame, *, validate_contract_mapping: bool = False
) -> pd.DataFrame:
    """Normalize the Databento loader's ET timestamp into a UTC column."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Databento MNQ loader must return a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("Databento MNQ loader returned no bars.")

    frame = dataframe.copy()
    timestamp_source = next(
        (name for name in ("timestamp", "timestamp ET") if name in frame.columns),
        None,
    )
    if timestamp_source is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            source_timestamps = pd.Series(frame.index, index=frame.index)
        elif frame.index.name in {"timestamp", "timestamp ET", "ts_event"}:
            source_timestamps = pd.Series(frame.index, index=frame.index)
        else:
            raise ValueError(
                "Databento MNQ data requires a timestamp/timestamp ET column "
                "or a datetime timestamp index."
            )
    else:
        source_timestamps = frame[timestamp_source].reset_index(drop=True)
        frame = frame.reset_index(drop=True)

    frame["timestamp"] = pd.to_datetime(
        source_timestamps.to_numpy(),
        utc=True,
        errors="raise",
    )
    if timestamp_source == "timestamp ET":
        frame = frame.drop(columns=["timestamp ET"])
    if timestamp_source is None:
        frame = frame.reset_index(drop=True)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Databento MNQ data is missing required columns: {sorted(missing)}"
        )
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        duplicates = int(frame["timestamp"].duplicated().sum())
        raise ValueError(
            f"Databento MNQ data contains {duplicates} duplicate timestamps."
        )
    if validate_contract_mapping and "instrument_id" in frame.columns:
        instrument_ids = pd.to_numeric(frame["instrument_id"], errors="raise")
        if instrument_ids.isna().any() or (instrument_ids <= 0).any():
            raise ValueError("Databento MNQ data contains invalid contract instrument IDs.")
        frame["instrument_id"] = instrument_ids.astype("int64")
    if validate_contract_mapping:
        if "symbol" not in frame:
            raise ValueError("Databento MNQ data is missing its continuous contract symbol.")
        symbols = frame["symbol"].astype(str)
        unexpected = sorted(set(symbols) - {"MNQ.v.0"})
        if unexpected:
            raise ValueError(
                "Databento MNQ data contains symbols outside the configured "
                f"continuous contract MNQ.v.0: {unexpected[:10]}"
            )
        frame["symbol"] = symbols
    if len(frame) != len(dataframe):
        raise RuntimeError("Canonicalizing Databento timestamps changed row count.")
    return frame


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
