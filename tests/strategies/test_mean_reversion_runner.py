from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.mean_reversion.config import (
    MRL1_CONFIG,
    MRS2_CONFIG,
)
from src.strategies.mean_reversion.runner import (
    MeanReversionBacktestRunner,
    run_frozen_mean_reversion,
    summarize_trades,
)


def make_bar(
    timestamp: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    hmm_state: int = 2,
    vol_percentile: float = 90.0,
    zscore: float = 2.5,
) -> dict:
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "volume": 1000.0,
        "hmm_state": hmm_state,
        "vol_percentile": vol_percentile,
        "zscore": zscore,
    }


def test_runner_requires_features():
    data = pd.DataFrame(
        [
            {
                "timestamp": 1,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        ]
    )

    runner = MeanReversionBacktestRunner(MRS2_CONFIG)

    with pytest.raises(ValueError):
        runner.run(data)


def test_mrs2_runner_enters_and_exits_target():
    rows = [
        make_bar(
            1,
            100.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        ),
        make_bar(
            2,
            100.0,
            high=100.0,
            low=72.5,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=0.0,
        ),
    ]

    result = MeanReversionBacktestRunner(MRS2_CONFIG).run(pd.DataFrame(rows))

    assert len(result) == 1
    assert result.iloc[0]["r_multiple"] == pytest.approx(MRS2_CONFIG.rr)
    assert result.iloc[0]["exit_reason"] == "target"


def test_mrl1_runner_enters_and_exits_stop():
    rows = [
        make_bar(
            1,
            100.0,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=-2.5,
        ),
        make_bar(
            2,
            100.0,
            high=100.0,
            low=62.5,
            hmm_state=1,
            vol_percentile=30.0,
            zscore=0.0,
        ),
    ]

    result = MeanReversionBacktestRunner(MRL1_CONFIG).run(pd.DataFrame(rows))

    assert len(result) == 1
    assert result.iloc[0]["r_multiple"] == pytest.approx(-1.0)
    assert result.iloc[0]["exit_reason"] == "stop"


def test_strategy_convenience_function():
    rows = [
        make_bar(
            1,
            100.0,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=2.5,
        ),
        make_bar(
            2,
            100.0,
            low=72.5,
            hmm_state=2,
            vol_percentile=90.0,
            zscore=0.0,
        ),
    ]

    result = run_frozen_mean_reversion(
        pd.DataFrame(rows),
        "MRS2",
    )

    assert len(result) == 1


def test_invalid_strategy_rejected():
    data = pd.DataFrame(
        [
            make_bar(
                1,
                100.0,
            )
        ]
    )

    with pytest.raises(ValueError):
        run_frozen_mean_reversion(
            data,
            "MRS2_OLD",
        )


def test_summary_metrics():
    trades = pd.DataFrame(
        {
            "r_multiple": [
                1.0,
                -1.0,
                1.0,
                0.0,
            ]
        }
    )

    summary = summarize_trades(trades)

    assert summary["trades"] == 4
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["unresolved"] == 1
    assert summary["net_r"] == pytest.approx(1.0)
    assert summary["expectancy_r"] == pytest.approx(0.25)
    assert summary["profit_factor"] == pytest.approx(2.0)
