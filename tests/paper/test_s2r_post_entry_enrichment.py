from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.paper.research_replay import add_paper_trade_attribution
from src.paper.s2r_enrichment import enrich_paper_s2r_trades
from src.research.s2_extended.validation.s2r_raw_reconstruction import (
    run_s26_state_machine,
)
from src.strategies.s2r.fitting import S2FittedModel
from src.strategies.s2r.signal import BASE_FEATURES, S2SignalModel


def _scenario(
    *,
    entry_timestamp: str,
    entry_price: float,
    mae_each_bar: list[float],
    close_r_each_bar: list[float],
    baseline_exit_bar: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = pd.Timestamp(entry_timestamp)
    timestamps = [timestamp + pd.Timedelta(minutes=bar) for bar in range(21)]
    assert len(mae_each_bar) == len(close_r_each_bar) == 20
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "high": [entry_price, *[entry_price + value * 25.0 for value in mae_each_bar]],
            "close": [entry_price, *[entry_price - value * 25.0 for value in close_r_each_bar]],
        }
    )
    paper_trade = pd.DataFrame(
        [
            {
                "strategy_name": "S2R",
                "entry_signal_timestamp": timestamp,
                "entry_timestamp": timestamp,
                "entry_price": entry_price,
                "exit_timestamp": timestamp + pd.Timedelta(minutes=baseline_exit_bar),
                "exit_price": entry_price + 25.0,
                "exit_reason": "stop",
                "exit_reason_detail": (
                    "S2 baseline stop (NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)"
                ),
                "r_multiple": -1.0,
            }
        ]
    )
    return paper_trade, bars


def _case_after_exit() -> tuple[pd.DataFrame, pd.DataFrame]:
    mae = [0.18, 0.45, 0.90] + [0.10] * 17
    mae[7] = 1.07  # S4 MAE_8R eligibility; baseline stop is also at bar 8.
    close_r = [-0.18, -0.44, -0.72, -0.38, -0.19, -0.44, -0.75, -0.95, -1.04]
    close_r += [-0.80] * (20 - len(close_r))
    return _scenario(
        entry_timestamp="2021-07-30T14:33:00Z",
        entry_price=14923.0,
        mae_each_bar=mae,
        close_r_each_bar=close_r,
        baseline_exit_bar=8,
    )


def _case_before_exit() -> tuple[pd.DataFrame, pd.DataFrame]:
    mae = [0.04, 0.04, 0.16, 0.25, 0.42, 0.57, 0.61, 0.75] + [0.50] * 12
    mae[15] = 1.0  # Baseline stop at bar 16, after the S26 deadline.
    close_r = [0.08, 0.22, -0.06, -0.13, -0.40, -0.42, -0.55, -0.59]
    close_r += [-0.64, -0.72, -0.71, -0.84, -0.69, -0.72]
    close_r += [-0.70] * (20 - len(close_r))
    close_r[13] = -0.72
    return _scenario(
        entry_timestamp="2021-08-02T16:07:00Z",
        entry_price=14969.0,
        mae_each_bar=mae,
        close_r_each_bar=close_r,
        baseline_exit_bar=16,
    )


def _research_s26_result(bars: pd.DataFrame, entry_price: float) -> dict[str, object]:
    cumulative_mae = np.maximum.accumulate(
        (bars["high"].to_numpy(dtype=float)[1:] - entry_price) / 25.0
    )
    close_r = (entry_price - bars["close"].to_numpy(dtype=float)[1:]) / 25.0
    row = {
        **{f"mae_{bar}R": float(cumulative_mae[bar - 1]) for bar in range(1, 21)},
        **{f"close_{bar}R": float(close_r[bar - 1]) for bar in range(1, 21)},
        "final_close_R": float(close_r[-1]),
    }
    return run_s26_state_machine(pd.Series(row))


def test_s26_failure_after_baseline_exit_is_analytical_only():
    paper, bars = _case_after_exit()
    original_exit = paper.loc[0, ["exit_timestamp", "exit_price", "exit_reason"]].copy()

    enriched_trades = enrich_paper_s2r_trades(paper, bars)
    enriched = enriched_trades.iloc[0]
    research = _research_s26_result(bars, 14923.0)

    assert enriched["exit_timestamp"] == original_exit["exit_timestamp"]
    assert enriched["exit_price"] == original_exit["exit_price"]
    assert enriched["exit_reason"] == original_exit["exit_reason"]
    assert enriched["s2r_recovery_state"] == research["state"] == "FAILED_TO_RECOVER"
    assert enriched["s2r_recovery_exit_type"] == research["exit_type"] == "FAILURE_EXIT"
    assert enriched["s2r_mae_bar"] == research["mae_bar"] == 3
    assert enriched["s2r_exit_bar"] == research["exit_bar"] == 9
    assert enriched["s2r_strategy_R"] == pytest.approx(research["strategy_R"])
    assert enriched["s2r_strategy_R"] == pytest.approx(-1.04)


def test_s26_failure_before_baseline_exit_is_analytical_only():
    paper, bars = _case_before_exit()
    original_exit = paper.loc[0, ["exit_timestamp", "exit_price", "exit_reason"]].copy()

    enriched_trades = enrich_paper_s2r_trades(paper, bars)
    enriched = enriched_trades.iloc[0]
    research = _research_s26_result(bars, 14969.0)

    assert enriched["exit_timestamp"] == original_exit["exit_timestamp"]
    assert enriched["exit_price"] == original_exit["exit_price"]
    assert enriched["exit_reason"] == original_exit["exit_reason"]
    assert enriched["s2r_recovery_state"] == research["state"] == "FAILED_TO_RECOVER"
    assert enriched["s2r_recovery_exit_type"] == research["exit_type"] == "FAILURE_EXIT"
    assert enriched["s2r_mae_bar"] == research["mae_bar"] == 8
    assert enriched["s2r_exit_bar"] == research["exit_bar"] == 14
    assert enriched["s2r_strategy_R"] == pytest.approx(research["strategy_R"])
    assert enriched["s2r_strategy_R"] == pytest.approx(-0.72)


def test_s4_ineligible_trade_keeps_original_s2_attribution():
    mae = [0.18, 0.20, 0.25, 0.30, 0.31, 0.32, 0.33, 0.33] + [0.40] * 12
    mae[10] = 0.95  # A later MAE does not repair the S4 bar-8 eligibility failure.
    close_r = [0.05] * 20
    paper, bars = _scenario(
        entry_timestamp="2021-07-28T15:47:00Z",
        entry_price=14993.25,
        mae_each_bar=mae,
        close_r_each_bar=close_r,
        baseline_exit_bar=19,
    )

    enriched_trades = enrich_paper_s2r_trades(paper, bars)
    enriched = enriched_trades.iloc[0]

    assert not enriched["s2r_enrichment_eligible"]
    assert enriched["s2r_recovery_state"] == "NO_RECOVERY_ENRICHMENT"
    assert enriched["s2r_recovery_exit_type"] == "ORIGINAL_S2"
    assert pd.isna(enriched["s2r_mae_bar"])
    assert pd.isna(enriched["s2r_exit_bar"])
    assert enriched["s2r_strategy_R"] == pytest.approx(-1.0)

    context = bars.iloc[[0]].copy()
    context["hmm_state"] = 2
    context["hmm_window"] = 1
    context["realized_vol_30"] = 2.0
    for feature in BASE_FEATURES:
        context[feature] = -1.0
    model = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 0.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=tuple(float(value) for value in range(100)),
    )
    attributed = add_paper_trade_attribution(
        enriched_trades, context, {1: model}
    ).iloc[0]
    assert attributed["research_state"] == "NO_RECOVERY_ENRICHMENT"
    assert attributed["research_exit_type"] == "ORIGINAL_S2"


def test_explicit_s26_metadata_overrides_baseline_exit_reason_attribution():
    paper, bars = _case_after_exit()
    enriched_trade = enrich_paper_s2r_trades(paper, bars)
    timestamp = paper.loc[0, "entry_signal_timestamp"]
    enriched_trade["hmm_state"] = 2
    enriched_trade["hmm_window"] = 1
    market = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "hmm_state": 2,
                "hmm_window": 1,
                "realized_vol_30": 2.0,
                **{feature: -1.0 for feature in BASE_FEATURES},
            }
        ]
    )
    model = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 0.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=tuple(float(value) for value in range(100)),
    )

    attributed = add_paper_trade_attribution(enriched_trade, market, {1: model}).iloc[0]

    assert "NO_RECOVERY_ENRICHMENT" in attributed["exit_reason_detail"]
    assert attributed["research_state"] == "FAILED_TO_RECOVER"
    assert attributed["research_exit_type"] == "FAILURE_EXIT"
    assert attributed["s2r_strategy_R"] == pytest.approx(-1.04)


def test_enrichment_matches_the_three_validated_research_cases():
    results = Path(__file__).resolve().parents[2] / "src" / "research" / "results" / "s2_extended"
    s3 = pd.read_csv(results / "s3_failure_path_enriched.csv")
    s4 = pd.read_csv(results / "s4_adverse_recovery_enriched.csv")
    s26 = pd.read_csv(results / "s26_mae_recovery_integration_trades.csv")
    s27 = pd.read_csv(results / "s27_full_strategy_trades.csv")
    cases = {
        "2021-07-28T15:47:00Z": "NO_RECOVERY_ENRICHMENT",
        "2021-07-30T14:33:00Z": "FAILED_TO_RECOVER",
        "2021-08-02T16:07:00Z": "FAILED_TO_RECOVER",
    }
    s3_entries = pd.to_datetime(s3["entry_timestamp"], utc=True)
    s4_entries = pd.to_datetime(s4["entry_timestamp"], utc=True)
    s27_entries = pd.to_datetime(s27["entry_timestamp"], utc=True)

    for entry_text, expected_state in cases.items():
        entry = pd.Timestamp(entry_text)
        path = s3.loc[s3_entries.eq(entry)].iloc[0]
        entry_price = float(path["entry_price"])
        path_length = int(path["path_length"])
        bars = pd.DataFrame(
            {
                "timestamp": [entry + pd.Timedelta(minutes=i) for i in range(path_length)],
                "high": [
                    entry_price,
                    *[
                        entry_price + float(path[f"mae_{bar}R"]) * 25.0
                        for bar in range(1, path_length)
                    ],
                ],
                "close": [
                    entry_price,
                    *[
                        entry_price - float(path[f"close_{bar}R"]) * 25.0
                        for bar in range(1, path_length)
                    ],
                ],
            }
        )
        raw_points = float(path["raw_points"])
        baseline_exit = pd.Timestamp(path["exit_timestamp"]).tz_convert("UTC")
        paper = pd.DataFrame(
            [
                {
                    "strategy_name": "S2R",
                    "entry_timestamp": entry,
                    "entry_price": entry_price,
                    "exit_timestamp": baseline_exit,
                    "exit_price": entry_price - raw_points,
                    "exit_reason": str(path["exit_reason"]),
                    "r_multiple": raw_points / float(path["stop_points"]),
                }
            ]
        )

        actual = enrich_paper_s2r_trades(paper, bars).iloc[0]
        research_trade = s27.loc[s27_entries.eq(entry)].iloc[0]
        eligible = bool(s4_entries.eq(entry).any())

        assert actual["s2r_enrichment_eligible"] is eligible
        assert actual["s2r_recovery_state"] == expected_state
        assert research_trade["_state"] == expected_state
        assert actual["exit_timestamp"] == baseline_exit
        assert actual["exit_price"] == pytest.approx(entry_price - raw_points)
        if eligible:
            s4_row_index = int(np.flatnonzero(s4_entries.eq(entry).to_numpy())[0])
            research_model = s26.loc[s26["original_index"].eq(s4_row_index)].iloc[0]
            assert actual["s2r_recovery_exit_type"] == research_model["exit_type"]
            assert actual["s2r_mae_bar"] == research_model["mae_bar"]
            assert actual["s2r_exit_bar"] == research_model["exit_bar"]
            assert actual["s2r_strategy_R"] == pytest.approx(
                research_model["strategy_R"]
            )
            assert actual["s2r_recovery_exit_type"] == research_trade["_exit_type"]
            assert actual["s2r_strategy_R"] == pytest.approx(
                research_trade["_strategy_R"]
            )
        else:
            assert actual["s2r_recovery_exit_type"] == "ORIGINAL_S2"
            assert research_trade["_exit_type"] == "ORIGINAL_S2"
