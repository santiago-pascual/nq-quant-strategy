from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import pandas as pd

from src.strategies.mean_reversion.backtest import (
    MeanReversionBacktestAdapter,
)
from src.strategies.mean_reversion.config import (
    MRL1_CONFIG,
    MRS2_CONFIG,
    MeanReversionConfig,
)


class MeanReversionBacktestRunner:
    """
    Production-style research runner for the frozen Mean Reversion strategies.

    Responsibilities:
        - validate the required feature columns
        - convert DataFrame rows into adapter bars
        - run the frozen strategy/lifecycle
        - return a normalized trade DataFrame

    This runner intentionally does NOT:
        - calculate features
        - modify strategy parameters
        - apply commissions
        - apply slippage
        - perform position sizing
        - alter the generic BacktestEngine

    Execution costs will be introduced only after the payoff geometry
    reproduces the research results.
    """

    REQUIRED_COLUMNS = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hmm_state",
        "vol_percentile",
        "zscore",
    }

    def __init__(
        self,
        config: MeanReversionConfig,
    ) -> None:
        self.config = config
        self.adapter = MeanReversionBacktestAdapter(config=config)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the frozen Mean Reversion strategy over a feature-enriched
        historical DataFrame.
        """
        self._validate_input(data)

        ordered = data.copy().sort_values("timestamp").reset_index(drop=True)

        bars = [self._row_to_bar(row) for row in ordered.itertuples(index=False)]

        trades = self.adapter.run(bars)

        if not trades:
            return self._empty_result()

        return pd.DataFrame([asdict(trade) for trade in trades])

    def _validate_input(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MeanReversionBacktestRunner expects a pandas DataFrame.")

        missing = self.REQUIRED_COLUMNS.difference(data.columns)

        if missing:
            raise ValueError(
                "Missing required Mean Reversion columns: " + ", ".join(sorted(missing))
            )

        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        numeric_columns = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "hmm_state",
            "vol_percentile",
            "zscore",
        }

        for column in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise TypeError(f"Column '{column}' must be numeric.")

        if data["timestamp"].isna().any():
            raise ValueError("Column 'timestamp' contains missing values.")

        if not data["timestamp"].is_monotonic_increasing:
            # Sorting is allowed, so this is deliberately not an error.
            pass

    @staticmethod
    def _row_to_bar(row: Any) -> dict[str, Any]:
        """
        Convert one DataFrame row into the bar/context format expected by
        MeanReversionBacktestAdapter.
        """
        return {
            "timestamp": row.timestamp,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "hmm_state": int(row.hmm_state),
            "vol_percentile": float(row.vol_percentile),
            "zscore": float(row.zscore),
        }

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "strategy_name",
                "candidate_id",
                "side",
                "entry_price",
                "exit_price",
                "result",
                "r_multiple",
                "bars_elapsed",
                "exit_reason",
            ]
        )


def run_frozen_mean_reversion(
    data: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    """
    Convenience function for running one frozen candidate.

    Parameters
    ----------
    data:
        Feature-enriched historical bars.

    strategy:
        "MRS2" or "MRL1".
    """
    configs = {
        "MRS2": MRS2_CONFIG,
        "MRL1": MRL1_CONFIG,
    }

    key = strategy.upper()

    if key not in configs:
        raise ValueError(
            f"Unknown Mean Reversion strategy '{strategy}'. "
            f"Expected one of: {', '.join(configs)}"
        )

    runner = MeanReversionBacktestRunner(configs[key])
    return runner.run(data)


def summarize_trades(
    trades: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Calculate the core payoff metrics from normalized trade results.

    R-multiple is the canonical unit.
    """
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
        }

    r = pd.to_numeric(
        trades["r_multiple"],
        errors="coerce",
    ).fillna(0.0)

    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    unresolved = int((r == 0).sum())

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = r.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    return {
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_rate": (wins / len(r) if len(r) > 0 else 0.0),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "profit_factor": profit_factor,
        "max_drawdown_r": float(drawdown.min()),
    }
