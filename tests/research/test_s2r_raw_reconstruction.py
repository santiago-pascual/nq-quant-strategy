from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.s2_extended.validation.s2r_raw_reconstruction import (
    BASE_FEATURES,
    MAE_THRESHOLD_R,
    RECOVERY_DEADLINE_BARS,
    RECOVERY_LEVEL_R,
    S4_ADVERSE_THRESHOLD_R,
    TOTAL_COST_POINTS,
    build_s4_cohort,
    build_s3_paths,
    calculate_quality,
    entry_qualifies,
    fit_signal_parameters,
    integrate_s27,
    resolve_short_trade,
    run_s26,
    run_s26_state_machine,
    transform_volatility,
    walk_signal_positions,
)


def _quality_train() -> pd.DataFrame:
    rows = []
    for state, offset in ((2, 0.0), (1, 100.0)):
        for index in range(20):
            row = {"hmm_state": state, "realized_vol_30": float(index)}
            for feature_index, feature in enumerate(BASE_FEATURES):
                row[feature] = offset + index + feature_index
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_uses_state_two_lower_tail_and_fifth_percentile_scales() -> None:
    thresholds, scales, reference = fit_signal_parameters(_quality_train())
    expected = _quality_train().query("hmm_state == 2")
    for feature in BASE_FEATURES:
        values = expected[feature].to_numpy(dtype=float)
        assert thresholds[feature] == np.quantile(values, 0.175)
        assert scales[feature] == np.quantile(values, 0.175) - np.quantile(
            values, 0.05
        )
    assert np.array_equal(reference, np.repeat(np.arange(20, dtype=float), 2))


def test_volatility_transform_uses_right_search_and_full_train_denominator() -> None:
    reference = np.array([1.0, 2.0, 2.0, 3.0])
    assert transform_volatility(2.0, reference) == 0.75
    assert transform_volatility(0.0, reference) == 0.0
    assert transform_volatility(4.0, reference) == 1.0


def test_quality_is_mean_of_four_clipped_feature_scores() -> None:
    thresholds = dict.fromkeys(BASE_FEATURES, 10.0)
    scales = dict.fromkeys(BASE_FEATURES, 10.0)
    row = pd.Series(dict(zip(BASE_FEATURES, (0.0, 5.0, 10.0, 20.0))))
    assert calculate_quality(row, thresholds, scales) == 0.375


def test_short_entry_gates_are_inclusive_at_lower_and_exclusive_at_upper_volatility() -> None:
    thresholds = dict.fromkeys(BASE_FEATURES, 0.0)
    scales = dict.fromkeys(BASE_FEATURES, 1.0)
    row = pd.Series(dict.fromkeys(BASE_FEATURES, -1.0))
    assert calculate_quality(row, thresholds, scales) == 1.0
    assert entry_qualifies(
        row,
        hmm_state=2,
        thresholds=thresholds,
        scales=scales,
        volatility_percentile=0.40,
    )
    assert not entry_qualifies(
        row,
        hmm_state=1,
        thresholds=thresholds,
        scales=scales,
        volatility_percentile=0.50,
    )
    assert not entry_qualifies(
        row,
        hmm_state=2,
        thresholds=thresholds,
        scales=scales,
        volatility_percentile=0.60,
    )
    weak = row.copy()
    weak.loc[BASE_FEATURES[0]] = 0.0
    weak.loc[BASE_FEATURES[1]] = 0.0
    assert not entry_qualifies(
        weak,
        hmm_state=2,
        thresholds=thresholds,
        scales=scales,
        volatility_percentile=0.50,
    )
    assert MAE_THRESHOLD_R == 0.70
    assert RECOVERY_LEVEL_R == 0.20
    assert RECOVERY_DEADLINE_BARS == 6


def _session(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def test_short_trade_takes_profit_only_after_entry_bar() -> None:
    session = _session(
        [200.0, 101.0, 99.0],
        [0.0, 99.0, 56.0],
        [100.0, 100.0, 60.0],
    )
    result = resolve_short_trade(session, 0)
    assert result == {"raw_points": 43.75, "reason": "target", "exit_position": 2}


def test_short_trade_stops_out() -> None:
    session = _session(
        [100.0, 125.0, 110.0],
        [100.0, 120.0, 100.0],
        [100.0, 124.0, 110.0],
    )
    result = resolve_short_trade(session, 0)
    assert result == {"raw_points": -25.0, "reason": "stop", "exit_position": 1}


def test_same_bar_target_and_stop_uses_conservative_stop() -> None:
    session = _session(
        [100.0, 130.0, 110.0],
        [100.0, 50.0, 100.0],
        [100.0, 100.0, 110.0],
    )
    result = resolve_short_trade(session, 0)
    assert result == {
        "raw_points": -25.0,
        "reason": "both_hit_conservative_stop",
        "exit_position": 1,
    }


def test_short_trade_times_out_at_horizon_close_and_cost_is_111_points() -> None:
    session = _session(
        [100.0] * 22,
        [100.0] * 22,
        [100.0, 100.0, 99.0] + [98.0] * 19,
    )
    result = resolve_short_trade(session, 0)
    assert result == {"raw_points": 2.0, "reason": "timeout", "exit_position": 20}
    assert np.isclose(TOTAL_COST_POINTS, 1.11)


def test_recovery_occurs_strictly_after_mae_and_deadline_is_inclusive() -> None:
    row = pd.Series(
        {
            "final_close_R": -1.0,
            "window": 1,
            "mae_1R": 0.70,
            "mae_2R": 0.90,
            "mae_3R": 0.90,
            "mae_4R": 0.90,
            "mae_5R": 0.90,
            "mae_6R": 0.90,
            "mae_7R": 0.90,
            "close_1R": 0.50,
            "close_2R": 0.10,
            "close_3R": 0.10,
            "close_4R": 0.10,
            "close_5R": 0.10,
            "close_6R": 0.10,
            "close_7R": 0.20,
        }
    )
    result = run_s26_state_machine(row)
    assert result == {
        "strategy_R": 0.20,
        "state": "RECOVERED",
        "mae_bar": 1,
        "recovery_bar": 7,
        "exit_bar": 7,
        "exit_type": "RECOVERY_EXIT",
    }


def test_recovery_deadline_failure_exits_at_deadline_close() -> None:
    row = pd.Series(
        {
            "final_close_R": -1.0,
            "mae_1R": 0.70,
            "mae_2R": 0.8,
            "mae_3R": 0.8,
            "mae_4R": 0.8,
            "mae_5R": 0.8,
            "mae_6R": 0.8,
            "mae_7R": 0.8,
            "close_1R": -0.8,
            "close_2R": -0.7,
            "close_3R": -0.6,
            "close_4R": -0.5,
            "close_5R": -0.4,
            "close_6R": -0.3,
            "close_7R": -0.2,
        }
    )
    result = run_s26_state_machine(row)
    assert result["state"] == "FAILED_TO_RECOVER"
    assert result["exit_bar"] == 7
    assert result["strategy_R"] == -0.2


def test_one_active_position_skips_repeated_and_exit_bar_signals() -> None:
    positions = np.array([0, 1, 2, 3, 4])
    selected = walk_signal_positions(
        positions=positions,
        session_length=30,
        qualifies=lambda _: True,
        resolve_exit_position=lambda position: position + 2,
        horizon=20,
    )
    assert selected == [0, 3]


def test_s4_transition_filters_only_mae_at_bar_eight() -> None:
    rows = []
    for value in (0.74, 0.75, 0.80):
        row = {
            "entry_timestamp": pd.Timestamp("2021-01-01", tz="UTC"),
            "exit_timestamp": pd.Timestamp("2021-01-02", tz="UTC"),
            "session_id": "2020-12-31",
            "window": 1,
            "net_R": -1.0,
        }
        for bar in range(1, 21):
            row[f"mae_{bar}R"] = value
            row[f"mfe_{bar}R"] = 0.1
            row[f"close_{bar}R"] = -0.1
        row["final_close_R"] = -0.1
        rows.append(row)
    cohort = build_s4_cohort(pd.DataFrame(rows))
    assert len(cohort) == 2
    assert (cohort["mae_8"] >= S4_ADVERSE_THRESHOLD_R).all()
    assert cohort["recovery_group"].tolist() == ["FAILURE", "FAILURE"]


def test_s26_and_s27_stage_transition_preserves_unmatched_s2_trades() -> None:
    s2 = pd.DataFrame(
        [
            {
                "entry_timestamp": pd.Timestamp("2021-01-01", tz="UTC"),
                "exit_timestamp": pd.Timestamp("2021-01-01 00:20", tz="UTC"),
                "session_id": "2020-12-31",
                "window": 1,
                "net_R": -0.5,
            },
            {
                "entry_timestamp": pd.Timestamp("2021-01-02", tz="UTC"),
                "exit_timestamp": pd.Timestamp("2021-01-02 00:20", tz="UTC"),
                "session_id": "2021-01-01",
                "window": 1,
                "net_R": 0.25,
            },
        ]
    )
    s4 = pd.DataFrame(
        [
            {
                "entry_timestamp": s2.iloc[0]["entry_timestamp"],
                "exit_timestamp": s2.iloc[0]["exit_timestamp"],
                "session_id": s2.iloc[0]["session_id"],
                "window": 1,
                "final_close_R": -1.0,
                "mae_1R": 0.8,
                "close_1R": -0.8,
                "mae_2R": 0.8,
                "close_2R": 0.2,
            }
        ]
    )
    s26 = run_s26(s4)
    s27 = integrate_s27(s2, s4, s26)
    assert len(s26) == 1
    assert len(s27) == 2
    assert s27.loc[0, "_strategy_R"] == 0.2
    assert s27.loc[0, "_state"] == "RECOVERED"
    assert s27.loc[1, "_strategy_R"] == 0.25
    assert s27.loc[1, "_state"] == "NO_RECOVERY_ENRICHMENT"


def test_s2_to_s3_to_s4_to_s26_is_deterministic_on_raw_bars() -> None:
    timestamps = pd.date_range(
        "2021-01-04 09:30",
        periods=21,
        freq="min",
        tz="America/New_York",
    )
    closes = np.full(21, 100.0)
    closes[3:20] = 95.0
    closes[20] = 90.0
    highs = np.full(21, 100.0)
    highs[2] = 120.0
    market = pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "session_date": ["2021-01-03"] * 21,
            "high": highs,
            "low": np.full(21, 99.0),
            "close": closes,
        }
    )
    s2 = pd.DataFrame(
        [
            {
                "entry_timestamp": timestamps[0],
                "exit_timestamp": timestamps[20],
                "session_id": "2021-01-03",
                "window": 1,
                "net_R": -0.5,
                "exit_reason": "timeout",
            }
        ]
    )
    s3_first = build_s3_paths(market, s2)
    s3_second = build_s3_paths(market, s2)
    pd.testing.assert_frame_equal(s3_first, s3_second)
    s4 = build_s4_cohort(s3_first)
    s26 = run_s26(s4)
    assert len(s2) == len(s3_first) == len(s4) == len(s26) == 1
    assert s4.loc[0, "mae_8"] == 0.8
    assert s26.loc[0, "state"] == "RECOVERED"
    assert s26.loc[0, "recovery_bar"] == 3
