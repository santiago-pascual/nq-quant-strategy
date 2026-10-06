from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.regime import VolatilityRegimeModel
from src.paper.research_replay import (
    DEFAULT_END,
    DEFAULT_START,
    FrozenResearchHMMProvider,
    PARITY_FIELDS,
    ResearchReplayContextAdapter,
    summarize_replay,
    _ReplayEventLogger,
    _build_paper_engine,
    _fit_s2r_models,
    _mark_final_orb_rth_bars,
    _mark_s2r_entry_eligible,
    _paper_trades_from_events,
    add_paper_trade_attribution,
    build_argument_parser,
    build_candidate_diagnostics,
    build_candidate_evaluation_diagnostics,
    build_reference_trades,
    compare_trade_ledgers,
    load_frozen_hmm_states,
    _normalize_exit_reason,
    parse_oos_range,
    research_causal_expanding_percentile,
)
from src.paper.logger import PaperEventType
from src.paper.run_autonomous import build_real_paper_engine
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.fitting import S2FittedModel
from src.strategies.s2r.signal import BASE_FEATURES, S2SignalModel
from src.strategies.s2r.strategy import S2RStrategy


def test_frozen_state_loader_validates_hash_and_preserves_timestamp_state(tmp_path):
    path = tmp_path / "states.csv"
    path.write_text(
        "timestamp,window,hmm_state\n"
        "2021-05-06 17:14:00+00:00,1,2\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    states = load_frozen_hmm_states(path, expected_sha256=digest)

    assert states.iloc[0]["timestamp"] == pd.Timestamp("2021-05-06 17:14:00Z")
    assert int(states.iloc[0]["hmm_state"]) == 2
    assert int(states.iloc[0]["research_hmm_window"]) == 1
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_hmm_states(path, expected_sha256="0" * 64)


def test_research_adapter_uses_frozen_state_without_causal_inference(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Causal HMM inference must not be called.")

    monkeypatch.setattr(VolatilityRegimeModel, "predict_states", fail_if_called)
    frozen = {
        "timestamp": pd.Timestamp("2021-05-06 17:14:00Z").to_pydatetime(),
        "hmm_state": 2,
        "hmm_window": 1,
        "vol_percentile": 42.0,
    }

    enriched = ResearchReplayContextAdapter().update(frozen)

    assert enriched == frozen
    assert enriched["hmm_state"] == 2


def test_hmm_provider_boundary_marks_frozen_provider_test_only():
    stamp = pd.Timestamp("2021-05-06 17:14:00Z")
    frozen = FrozenResearchHMMProvider(
        pd.DataFrame({"timestamp": [stamp], "hmm_state": [2]})
    )
    assert frozen.TEST_ONLY is True
    adapter = ResearchReplayContextAdapter(hmm_provider=frozen)
    enriched = adapter.update({"timestamp": stamp, "hmm_state": 0})
    assert enriched["hmm_state"] == 2


def test_s2r_uses_its_fitted_reference_volatility():
    signal_model = S2SignalModel(
        thresholds={feature: 1.0 for feature in BASE_FEATURES},
        scales={feature: 1.0 for feature in BASE_FEATURES},
    )
    fitted = S2FittedModel(
        signal_model=signal_model,
        volatility_reference=(1.0, 2.0, 3.0, 4.0),
    )
    strategy = S2RStrategy(fitted_model=fitted)
    row = {
        "hmm_state": 2,
        "realized_vol_30": 2.0,
        **{feature: 0.0 for feature in BASE_FEATURES},
    }

    assert fitted.transform_volatility(2.0) == 0.5
    assert strategy.generate_signal(row).value == "short"


def test_s2r_uses_window_specific_hmm_state_instead_of_shared_provider_state():
    fitted = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 1.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=(1.0, 2.0, 3.0, 4.0),
    )
    strategy = S2RStrategy()
    strategy.set_fitted_models_for_bar({1: fitted})
    row = {
        "hmm_state": 2,
        "s2r_hmm_states": ((1, 1),),
        "realized_vol_30": 2.0,
        **{feature: 0.0 for feature in BASE_FEATURES},
    }

    assert strategy.generate_signal(row).value == "flat"

    row["s2r_hmm_states"] = ((1, 2),)
    row["hmm_state"] = 0
    assert strategy.generate_signal(row).value == "short"


def test_replay_engine_uses_model_driven_s2r_strategy():
    engine = _build_paper_engine(
        ("S2R",),
        context_adapter=ResearchReplayContextAdapter(),
        initial_equity=50_000.0,
        commission_per_contract=0.0,
        price_offset=0.0,
        logger=_ReplayEventLogger(),
    )

    assert isinstance(engine.strategies[0], S2RStrategy)
    assert engine.risk.limits.max_total_risk is None


def test_real_paper_engine_keeps_production_aggregate_risk_cap():
    engine, _ = build_real_paper_engine(Path("."))

    assert engine.risk.limits.max_total_risk == pytest.approx(375.0)


def test_s2r_schedule_evaluates_shared_boundary_in_both_windows(monkeypatch):
    class WindowRegime:
        instances = 0

        def __init__(self, n_states, random_state):
            assert n_states == 3
            assert random_state == 42
            self.state = 2 if WindowRegime.instances == 0 else 1
            WindowRegime.instances += 1

        def fit(self, frame):
            self.training_rows = len(frame)

        def predict_states(self, frame):
            return pd.Series(self.state, index=frame.index, name="hmm_state")

    monkeypatch.setattr("src.paper.research_replay.VolatilityRegimeModel", WindowRegime)
    timestamps = pd.to_datetime(
        [
            "2020-01-01T00:00:00Z",
            "2021-12-31T23:59:00Z",
            "2022-01-01T00:00:00Z",
            "2022-01-01T00:01:00Z",
            "2022-01-01T00:02:00Z",
        ],
        utc=True,
    )
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            # These global provider labels must not train S2R's window models.
            "hmm_state": [0] * len(timestamps),
            "realized_vol_30": [0.1, 0.2, 0.3, 0.4, 0.5],
            "realized_vol_5": [0.1, 0.2, 0.3, 0.4, 0.5],
            "realized_vol_15": [0.1, 0.2, 0.3, 0.4, 0.5],
            "realized_vol_60": [0.1, 0.2, 0.3, 0.4, 0.5],
            "variance_ratio_5_30": [1.0, 2.0, 3.0, 4.0, 5.0],
            "variance_ratio_5_60": [1.5, 2.5, 3.5, 4.5, 5.5],
            **{
                feature: [float(index) for index in range(len(timestamps))]
                for feature in BASE_FEATURES
            },
        }
    )
    reference = pd.DataFrame(
        [
            {
                "window": 1,
                "train_start": "2021-12-31T00:00:00Z",
                "train_end": "2022-01-01T00:00:00Z",
                "oos_start": "2022-01-01T00:00:00Z",
                "oos_end": "2022-01-01T00:02:00Z",
            },
            {
                "window": 2,
                "train_start": "2021-12-31T00:00:00Z",
                "train_end": "2022-01-01T00:02:00Z",
                "oos_start": "2022-01-01T00:02:00Z",
                "oos_end": "2022-01-01T00:03:00Z",
            },
        ]
    )

    models, mapping = _fit_s2r_models(market, reference)

    assert sorted(models) == [1, 2]
    assert mapping["s2r_window"].tolist() == [-1, -1, 1, 1, -1]
    assert mapping["s2r_windows"].tolist() == [(), (), (1,), (1,), (1, 2)]
    assert mapping["s2r_hmm_states"].tolist() == [
        (), (), ((1, 2),), ((1, 2),), ((1, 2), (2, 1))
    ]
    assert np.isfinite(models[1].signal_model.thresholds[BASE_FEATURES[0]])
    assert np.isnan(models[2].signal_model.thresholds[BASE_FEATURES[0]])
    assert len(mapping) == len(market)
    assert not mapping["timestamp"].duplicated().any()


def test_shared_boundary_evaluates_both_models_but_submits_one_economic_order():
    from src.strategies.s2r.fitting import S2FittedModel

    model = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 0.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=tuple(float(value) for value in range(100)),
    )
    logger = _ReplayEventLogger()
    engine = _build_paper_engine(
        ["S2R"],
        context_adapter=ResearchReplayContextAdapter({1: model, 2: model}),
        initial_equity=50_000.0,
        commission_per_contract=0.0,
        price_offset=0.0,
        logger=logger,
    )
    timestamp = pd.Timestamp("2024-05-06 13:30:00", tz="UTC").to_pydatetime()
    engine.connect()
    try:
        result = engine.process_bar(
            {
                "timestamp": timestamp,
                "symbol": "MNQ",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000.0,
                "hmm_state": 2,
                "hmm_window": None,
                "s2r_windows": (1, 2),
                "realized_vol_30": 50.0,
                **{feature: -1.0 for feature in BASE_FEATURES},
            }
        )
    finally:
        engine.disconnect()

    evaluations = build_candidate_evaluation_diagnostics(logger.events)
    assert evaluations["window"].tolist() == [1, 2]
    assert evaluations["qualifies"].tolist() == [True, True]
    assert evaluations["timestamp"].nunique() == 1
    assert result.decisions["S2R"].action.value == "enter"
    assert len(result.submitted_orders) == 1
    assert len(result.fills) == 1
    assert len(result.open_positions) == 1
    economic_candidates = build_candidate_diagnostics(logger.events)
    assert len(economic_candidates) == 1


def test_candidate_evaluation_diagnostics_do_not_depend_on_reference_trades():
    logger = _ReplayEventLogger()
    timestamp = pd.Timestamp("2024-05-06 13:30:00", tz="UTC").to_pydatetime()
    logger.append(
        PaperEventType.CANDIDATE_EVALUATION,
        {
            "strategy_name": "S2R",
            "evaluation_id": "S2R|2024-05-06T13:30:00+00:00|window-1",
            "entry_timestamp": timestamp.isoformat(),
            "session_id": "2024-05-06",
            "window": 1,
            "hmm_state": 2,
            "qualifies": False,
            "reason": "base_signal_failed",
        },
        timestamp=timestamp,
    )
    diagnostics = build_candidate_evaluation_diagnostics(logger.events)
    assert diagnostics[["window", "qualifies"]].to_dict("records") == [
        {"window": 1, "qualifies": False}
    ]
    assert not any("reference" in str(value).lower() for value in diagnostics.columns)


def test_boundary_trade_attribution_uses_the_model_that_qualified():
    timestamp = pd.Timestamp("2024-05-06 13:30:00", tz="UTC")
    paper = pd.DataFrame(
        [
            {
                "strategy_name": "S2R",
                "entry_signal_timestamp": timestamp,
                "exit_reason_detail": "S2R recovery RECOVERED",
                "s2r_recovery_state": "RECOVERED",
                "s2r_recovery_exit_type": "RECOVERY_EXIT",
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "hmm_state": 2,
                "hmm_window": np.nan,
                "s2r_windows": (1, 2),
                "realized_vol_30": 50.0,
                **{feature: -1.0 for feature in BASE_FEATURES},
            }
        ]
    )
    qualifying = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 0.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=tuple(float(value) for value in range(100)),
    )
    failing = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: -0.5 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=tuple(float(value) for value in range(100)),
    )

    attributed = add_paper_trade_attribution(
        paper, market, {1: qualifying, 2: failing}
    ).iloc[0]

    assert attributed["window"] == 1
    assert attributed["quality"] == pytest.approx(1.0)
    assert attributed["volatility_percentile"] == pytest.approx(0.51)


def test_frozen_s2r_schedule_is_metadata_only_and_trade_independent():
    from src.paper.research_replay import S2R_WINDOW_SCHEDULE

    schedule = pd.read_csv(S2R_WINDOW_SCHEDULE)
    assert list(schedule.columns) == [
        "window",
        "train_start",
        "train_end",
        "oos_start",
        "oos_end",
    ]
    assert schedule["window"].tolist() == list(range(1, 23))
    parsed = schedule.assign(
        oos_start=pd.to_datetime(schedule["oos_start"], utc=True),
        oos_end=pd.to_datetime(schedule["oos_end"], utc=True),
    )
    assert (
        parsed["oos_start"].iloc[1:].reset_index(drop=True)
        >= parsed["oos_end"].iloc[:-1].reset_index(drop=True)
    ).all()


@pytest.mark.parametrize(
    ("timestamp_utc", "expected_local_date"),
    [
        ("2021-01-04 14:30:00+00:00", date(2021, 1, 4)),
        ("2021-07-06 13:30:00+00:00", date(2021, 7, 6)),
    ],
)
def test_orb_context_uses_new_york_local_session(
    timestamp_utc, expected_local_date
):
    strategy = ORBStrategy()

    strategy.evaluate(
        {
            "timestamp": pd.Timestamp(timestamp_utc).to_pydatetime(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )

    assert strategy._context is not None
    assert strategy._context.session_date == expected_local_date
    assert strategy._context.is_rth


def test_comparator_reports_exact_missing_extra_and_first_mismatch():
    base = {
        "direction": "SHORT",
        "entry_price": 100.0,
        "exit_price": 99.0,
        "exit_reason": "target",
        "r_multiple": 0.5,
    }
    reference = pd.DataFrame(
        [
            {
                "trade_key": "MRL1|2021-01-01T14:00:00+00:00",
                "strategy_name": "MRL1",
                "entry_timestamp": pd.Timestamp("2021-01-01 14:00Z"),
                "exit_timestamp": pd.Timestamp("2021-01-01 14:01Z"),
                **base,
            },
            {
                "trade_key": "MRS2|2021-01-01T14:00:00+00:00",
                "strategy_name": "MRS2",
                "entry_timestamp": pd.Timestamp("2021-01-01 14:00Z"),
                "exit_timestamp": pd.Timestamp("2021-01-01 14:01Z"),
                **base,
            },
            {
                "trade_key": "S2R|2021-01-01T14:00:00+00:00",
                "strategy_name": "S2R",
                "entry_timestamp": pd.Timestamp("2021-01-01 14:00Z"),
                "exit_timestamp": pd.Timestamp("2021-01-01 14:01Z"),
                **base,
            },
        ]
    )
    paper = pd.DataFrame(
        [
            reference.iloc[0].to_dict(),
            {
                **reference.iloc[2].to_dict(),
                "exit_price": 98.0,
                "r_multiple": 1.0,
            },
            {
                "trade_key": "ORB|2021-01-01T14:00:00+00:00",
                "strategy_name": "ORB",
                "entry_timestamp": pd.Timestamp("2021-01-01 14:00Z"),
                "exit_timestamp": pd.Timestamp("2021-01-01 14:01Z"),
                **base,
            },
        ]
    )

    parity = compare_trade_ledgers(reference, paper)
    statuses = parity.set_index("strategy_name")["status"].to_dict()

    assert statuses == {
        "MRL1": "EXACT",
        "MRS2": "MISSING",
        "S2R": "MISMATCH",
        "ORB": "EXTRA",
    }
    mismatch = parity.loc[parity["strategy_name"].eq("S2R")].iloc[0]
    assert mismatch["first_difference_field"] == "exit_price"
    assert tuple(PARITY_FIELDS[:7]) == (
        "entry_timestamp",
        "exit_timestamp",
        "direction",
        "entry_price",
        "exit_price",
        "exit_reason",
        "r_multiple",
    )


def test_causal_expanding_percentile_uses_strictly_prior_rows():
    values = pd.Series([1.0, 1.0, 3.0, float("nan"), 2.0])

    result = research_causal_expanding_percentile(values)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 1.0
    assert pd.isna(result.iloc[3])
    assert result.iloc[4] == pytest.approx(2 / 3)


def test_reference_builder_normalizes_mr_s2r_and_ny_orb_artifacts():
    timestamps = pd.date_range("2021-05-06 17:14Z", periods=3, freq="min")
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 99.0, 98.0],
            "hmm_state": [2, 2, 1],
            "research_hmm_window": [1, 1, 1],
            "vol_percentile": [50.0, 51.0, 52.0],
        }
    )
    references = {
        "mean_reversion_trades": pd.DataFrame(
            [
                {
                    "strategy_name": "MRL1",
                    "entry_timestamp": "2021-05-06 17:14:00+00:00",
                    "bars_elapsed": 1,
                    "side": "LONG",
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "exit_reason": "target",
                    "r_multiple": 0.5,
                    "candidate_id": "MRL1_NEW",
                    "entry_hmm_state": 1,
                    "window": 1,
                    "entry_zscore": -2.8,
                }
            ]
        ),
        "s2r_trades": pd.DataFrame(
            [
                {
                    "_entry_ts": "2021-05-06 17:14:00+00:00",
                    "_exit_ts": "2021-05-06 17:34:00+00:00",
                    "session_id": "2021-05-05",
                    "quality": 0.9977525830778488,
                    "vol_percentile": 0.4739338852701669,
                    "stop_points": 25.0,
                    "raw_points": -7.75,
                    "net_R": -0.3544,
                    "exit_reason": "timeout",
                    "window": 1,
                    "_state": "NO_RECOVERY_ENRICHMENT",
                    "_exit_type": "ORIGINAL_S2",
                }
            ]
        ),
        "orb_trades": pd.DataFrame(
            [
                {
                    "session_date": "2021-05-06",
                    "direction": "LONG",
                    "entry_timestamp": "2021-05-06 10:00:00-04:00",
                    "exit_timestamp": "2021-05-06 10:02:00-04:00",
                    "entry_price": 100.0,
                    "risk_points": 10.0,
                    "raw_points": 20.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            ]
        ),
    }

    trades = build_reference_trades(
        references,
        market,
        pd.Timestamp("2021-05-06 00:00:00Z"),
        pd.Timestamp("2021-05-06 23:59:59Z"),
    ).set_index("strategy_name")

    assert trades.loc["MRL1", "exit_timestamp"] == timestamps[1]
    assert trades.loc["MRL1", "volatility_percentile"] == 50.0
    assert trades.loc["S2R", "r_multiple"] == pytest.approx(-7.75 / 25.0)
    assert trades.loc["S2R", "quality"] == pytest.approx(0.9977525830778488)
    assert trades.loc["S2R", "volatility_percentile"] == pytest.approx(
        0.4739338852701669
    )
    assert trades.loc["S2R", "research_exit_type"] == "ORIGINAL_S2"
    assert trades.loc["ORB", "exit_reason"] == "target"


def test_cli_uses_official_default_range_and_deterministic_outputs():
    args = build_argument_parser().parse_args([])
    start, end = parse_oos_range(args.start, args.end)

    assert args.start == DEFAULT_START
    assert args.end == DEFAULT_END
    assert start == pd.Timestamp("2020-06-23 00:00:00Z")
    assert end == pd.Timestamp("2026-06-19 23:59:59.999999999Z")
    assert str(args.output_dir).endswith("results\\paper\\research_replay")


def test_orb_trade_flows_through_paper_engine_and_lifecycle():
    logger = _ReplayEventLogger()
    engine = _build_paper_engine(
        ("ORB",),
        context_adapter=ResearchReplayContextAdapter(),
        initial_equity=50_000.0,
        commission_per_contract=0.0,
        price_offset=0.0,
        logger=logger,
    )
    local_start = pd.Timestamp("2021-01-04 09:30", tz="America/New_York")
    bars = [
        {
            "timestamp": (local_start + pd.Timedelta(minutes=minute)).tz_convert(
                "UTC"
            ).to_pydatetime(),
            "symbol": "MNQ",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "is_final_rth_bar": False,
        }
        for minute in range(30)
    ]
    bars.extend(
        [
            {
                "timestamp": (local_start + pd.Timedelta(minutes=30)).tz_convert(
                    "UTC"
                ).to_pydatetime(),
                "symbol": "MNQ",
                "open": 100.5,
                "high": 101.25,
                "low": 100.25,
                "close": 101.0,
                "volume": 1.0,
                "is_final_rth_bar": False,
            },
            {
                "timestamp": (local_start + pd.Timedelta(minutes=31)).tz_convert(
                    "UTC"
                ).to_pydatetime(),
                "symbol": "MNQ",
                "open": 101.0,
                "high": 105.25,
                "low": 100.5,
                "close": 105.0,
                "volume": 1.0,
                "is_final_rth_bar": False,
            },
        ]
    )

    engine.connect()
    try:
        for bar in bars:
            engine.process_bar(bar)
    finally:
        engine.disconnect()

    trades = _paper_trades_from_events(
        logger.events,
        point_value=engine.config.point_value,
        commission_per_contract=engine.config.commission_per_contract,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["strategy_name"] == "ORB"
    assert trades.iloc[0]["direction"] == "LONG"
    assert trades.iloc[0]["entry_price"] == 101.0
    assert trades.iloc[0]["exit_price"] == 105.0
    assert trades.iloc[0]["exit_reason"] == "target"
    parity_paper = trades.copy()
    parity_paper["candidate_id"] = "ORB"
    parity_paper["hmm_state"] = float("nan")
    parity_paper["window"] = float("nan")
    parity_paper["quality"] = float("nan")
    parity_paper["volatility_percentile"] = float("nan")
    parity_paper["volatility_percentile_scale"] = ""
    parity_paper["zscore"] = float("nan")
    parity_paper["research_state"] = ""
    parity_paper["research_exit_type"] = ""
    parity_paper["session_date"] = "2021-01-04"
    parity = compare_trade_ledgers(parity_paper.copy(), parity_paper)
    assert parity.iloc[0]["status"] == "EXACT"


@pytest.mark.parametrize(
    ("opening_range_width", "expected_decision", "expected_quantity", "expected_rule"),
    [
        (100.0, "approved", 1, "ORB_FORCE_1_UNDER_060"),
        (150.0, "approved", 1, "ORB_FORCE_1_UNDER_060"),
        (151.0, "rejected", 0, "ORB_REJECT_OVER_060"),
    ],
)
def test_orb_adaptive_one_contract_risk_cap_routes_at_300_dollars(
    opening_range_width,
    expected_decision,
    expected_quantity,
    expected_rule,
):
    logger = _ReplayEventLogger()
    engine = _build_paper_engine(
        ("ORB",),
        context_adapter=ResearchReplayContextAdapter(),
        initial_equity=50_000.0,
        commission_per_contract=0.0,
        price_offset=0.0,
        logger=logger,
    )
    local_start = pd.Timestamp("2021-01-04 09:30", tz="America/New_York")
    engine.connect()
    try:
        for minute in range(30):
            timestamp = (local_start + pd.Timedelta(minutes=minute)).tz_convert("UTC")
            engine.process_bar(
                {
                    "timestamp": timestamp.to_pydatetime(),
                    "symbol": "MNQ",
                    "open": 1000.0 + opening_range_width / 2,
                    "high": 1000.0 + opening_range_width,
                    "low": 1000.0,
                    "close": 1000.0 + opening_range_width / 2,
                    "volume": 1.0,
                    "is_final_rth_bar": False,
                }
            )
        breakout_timestamp = (local_start + pd.Timedelta(minutes=30)).tz_convert(
            "UTC"
        )
        engine.process_bar(
            {
                "timestamp": breakout_timestamp.to_pydatetime(),
                "symbol": "MNQ",
                "open": 1000.0 + opening_range_width,
                "high": 1000.0 + opening_range_width + 0.25,
                "low": 1000.0 + opening_range_width - 0.25,
                "close": 1000.0 + opening_range_width + 0.25,
                "volume": 1.0,
                "is_final_rth_bar": False,
            }
        )
    finally:
        engine.disconnect()

    risk_decision = next(
        event
        for event in logger.events
        if event["event_type"] == "risk_decision"
    )["payload"]
    assert risk_decision["decision"] == expected_decision
    assert risk_decision["quantity"] == expected_quantity
    assert risk_decision["sizing_rule"] == expected_rule
    assert risk_decision["theoretical_quantity"] == pytest.approx(
        125.0 / (2 * opening_range_width)
    )
    assert risk_decision["executable_quantity"] == 0
    if expected_decision == "approved":
        assert risk_decision["adaptive_quantity"] == 1
        assert risk_decision["total_risk"] == pytest.approx(
            2 * opening_range_width
        )
        assert len(engine.execution.get_positions()) == 1
    else:
        assert risk_decision["adaptive_quantity"] is None
        assert risk_decision["risk_per_contract"] == pytest.approx(302.0)
        assert not engine.execution.get_positions()

    diagnostics = build_candidate_diagnostics(logger.events)
    assert len(diagnostics) == 1
    candidate = diagnostics.iloc[0]
    assert candidate["candidate_decision"] == (
        "ACCEPTED" if expected_decision == "approved" else "REJECTED"
    )
    if expected_decision == "rejected":
        assert "stop distance" in candidate["rejection_reason"]
        assert candidate["candidate_outcome"] == (
            "REJECTED(stop distance is too large for the configured risk per trade)"
        )
    else:
        assert candidate["candidate_outcome"] == "ACCEPTED"


def test_candidate_diagnostics_do_not_manufacture_candidates_from_references():
    timestamps = pd.date_range("2021-01-04 14:00Z", periods=3, freq="min")
    reference = pd.DataFrame(
        [
            {
                "trade_key": f"MRL1|{timestamp.isoformat()}",
                "strategy_name": "MRL1",
                "entry_timestamp": timestamp,
                "candidate_id": "MRL1_NEW",
            }
            for timestamp in timestamps
        ]
    )
    events = [
        {
            "event_type": "strategy_decision",
            "timestamp": timestamps[0],
            "payload": {
                "strategy_name": "MRL1",
                "action": "enter",
                "signal": "long",
                "reason": "test candidate",
            },
        },
        {
            "event_type": "risk_decision",
            "timestamp": timestamps[0],
            "payload": {
                "strategy_name": "MRL1",
                "decision": "rejected",
                "reason": "stop distance too large",
                "quantity": 0,
            },
        },
        {
            "event_type": "strategy_decision",
            "timestamp": timestamps[2],
            "payload": {
                "strategy_name": "MRL1",
                "action": "enter",
                "signal": "long",
                "reason": "test candidate",
            },
        },
        {
            "event_type": "risk_decision",
            "timestamp": timestamps[2],
            "payload": {
                "strategy_name": "MRL1",
                "decision": "approved",
                "reason": "risk checks passed",
                "quantity": 1,
            },
        },
        {
            "event_type": "error",
            "timestamp": timestamps[2],
            "payload": {
                "message": "Portfolio conflict rejected entry: opposite side.",
                "context": {"strategy_name": "MRL1"},
            },
        },
    ]

    diagnostics = build_candidate_diagnostics(events).set_index(
        "entry_timestamp"
    )

    assert diagnostics.loc[timestamps[0], "status"] == "RISK_REJECTED"
    assert timestamps[1] not in diagnostics.index
    assert diagnostics.loc[timestamps[2], "status"] == "CONFLICT_REJECTED"
    assert diagnostics.loc[timestamps[0], "candidate_decision"] == "REJECTED"
    assert diagnostics.loc[timestamps[0], "rejection_reason"] == (
        "stop distance too large"
    )
    assert diagnostics.loc[timestamps[2], "candidate_decision"] == "REJECTED"
    assert "Portfolio conflict rejected" in diagnostics.loc[
        timestamps[2], "rejection_reason"
    ]


def test_candidate_ledger_emits_one_explicit_terminal_outcome_per_enter():
    timestamps = pd.date_range("2021-01-04 14:00Z", periods=3, freq="min")
    events = []
    for timestamp, risk_decision in zip(
        timestamps,
        ("rejected", "approved", "approved"),
    ):
        events.append(
            {
                "event_type": "strategy_decision",
                "timestamp": timestamp,
                "payload": {"strategy_name": "MRL1", "action": "enter"},
            }
        )
        events.append(
            {
                "event_type": "risk_decision",
                "timestamp": timestamp,
                "payload": {
                    "strategy_name": "MRL1",
                    "decision": risk_decision,
                    "reason": "unit rejection" if risk_decision == "rejected" else "approved",
                    "quantity": 0 if risk_decision == "rejected" else 1,
                },
            }
        )
        if timestamp == timestamps[1]:
            events.extend([
                {"event_type": "order_created", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1", "broker_order_id": "o1"}},
                {"event_type": "order_submitted", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1", "broker_order_id": "o1"}},
            ])
        elif timestamp == timestamps[2]:
            events.extend([
                {"event_type": "order_created", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1", "broker_order_id": "o2"}},
                {"event_type": "order_submitted", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1", "broker_order_id": "o2"}},
                {"event_type": "fill", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1", "broker_order_id": "o2", "quantity": 1}},
                {"event_type": "position_opened", "timestamp": timestamp,
                 "payload": {"strategy_name": "MRL1"}},
            ])

    diagnostics = build_candidate_diagnostics(events).set_index("entry_timestamp")
    assert diagnostics["terminal_outcome"].tolist() == [
        "RISK_REJECTED", "ORDER_NO_FILL", "OPEN_AT_END",
    ]
    assert diagnostics.loc[timestamps[0], "rejection_reason"] == "unit rejection"
    assert diagnostics.loc[timestamps[1], "rejection_reason"]
    assert diagnostics.loc[timestamps[2], "rejection_reason"] == ""
    assert len(diagnostics) == 3


def test_candidate_accounting_counts_generated_decisions_and_reasons():
    timestamps = pd.date_range("2021-01-04 14:00Z", periods=2, freq="min")
    events = [
        event
        for timestamp, risk_decision in zip(
            timestamps,
            (
                {
                    "decision": "approved",
                    "reason": "risk checks passed",
                    "quantity": 1,
                    "executable_quantity": 0,
                    "adaptive_quantity": 1,
                },
                {
                    "decision": "rejected",
                    "reason": "stop distance is too large",
                    "quantity": 0,
                },
            ),
        )
        for event in [
            {
                "event_type": "strategy_decision",
                "timestamp": timestamp,
                "payload": {
                    "strategy_name": "ORB",
                    "action": "enter",
                    "signal": "long",
                    "reason": "candidate",
                },
            },
            {
                "event_type": "risk_decision",
                "timestamp": timestamp,
                "payload": {"strategy_name": "ORB", **risk_decision},
            },
        ]
    ]
    events.insert(
        2,
        {
            "event_type": "order_submitted",
            "timestamp": timestamps[0],
            "payload": {"strategy_name": "ORB"},
        },
    )

    diagnostics = build_candidate_diagnostics(events)
    summary = summarize_replay(
        pd.DataFrame(columns=["strategy_name", "gross_pnl", "commission", "net_pnl"]),
        pd.DataFrame(columns=["strategy_name", "status"]),
        ("ORB",),
        open_positions=0,
        candidate_diagnostics=diagnostics,
    ).iloc[0]

    assert summary["candidates_generated"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["rejection_reason_counts"] == (
        '{"stop distance is too large": 1}'
    )


def test_s2r_attribution_uses_model_context_not_reference_ledger():
    timestamp = pd.Timestamp("2021-05-06 17:14:00Z")
    paper = pd.DataFrame(
        [
            {
                "strategy_name": "S2R",
                "entry_signal_timestamp": timestamp,
                "exit_reason_detail": "S2R recovery RECOVERED",
                "s2r_recovery_state": "RECOVERED",
                "s2r_recovery_exit_type": "RECOVERY_EXIT",
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "hmm_state": 1,
                "hmm_window": 6,
                "realized_vol_30": 4.0,
                **{feature: 100.0 for feature in BASE_FEATURES},
            }
        ]
    )
    fitted = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 1.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=(1.0, 2.0, 3.0, 4.0),
    )

    attributed = add_paper_trade_attribution(paper, market, {6: fitted}).iloc[0]

    assert attributed["hmm_state"] == 1
    assert attributed["window"] == 6
    assert attributed["quality"] == pytest.approx(
        fitted.signal_model.calculate_quality(
            {feature: 100.0 for feature in BASE_FEATURES}
        )
    )
    assert attributed["volatility_percentile"] == pytest.approx(1.0)
    assert attributed["research_state"] == "RECOVERED"
    assert attributed["session_date"] == "2021-05-05"


def test_candidate_diagnostics_support_empty_reference_and_event_sets():
    diagnostics = build_candidate_diagnostics([])

    assert diagnostics.empty
    assert "status" in diagnostics.columns


def test_final_orb_rth_marker_uses_last_available_bar_per_ny_session():
    timestamps = pd.DatetimeIndex(
        [
            "2024-01-08 09:30",
            "2024-01-08 15:58",
            "2024-01-08 15:59",
            "2024-01-08 16:00",
            "2024-01-08 16:59",
            "2024-01-09 09:30",
            "2024-01-09 12:58",
            "2024-01-09 12:59",
            "2024-01-09 16:00",
            "2024-01-09 16:59",
            "2024-01-10 09:30",
            "2024-01-10 13:13",
            "2024-01-10 13:14",
            "2024-01-10 16:59",
        ]
    ).tz_localize("America/New_York")
    rth = pd.DataFrame({"timestamp ET": timestamps})

    assert _mark_final_orb_rth_bars(rth).tolist() == [
        False, False, True, False, False,
        False, False, True, False, False,
        False, False, True, False,
    ]


def test_early_orb_session_close_exits_position_before_next_session():
    from src.strategies.base import StrategyAction
    from src.strategies.orb.lifecycle import ORBTradeState

    timestamps = pd.DatetimeIndex(
        [
            "2024-07-03 12:58",
            "2024-07-03 12:59",
            "2024-07-05 09:30",
            "2024-07-05 15:59",
        ]
    ).tz_localize("America/New_York")
    bars = pd.DataFrame(
        {
            "timestamp ET": timestamps,
            "timestamp": timestamps.tz_convert("UTC"),
            "open": [102.0] * 4,
            "high": [104.0] * 4,
            "low": [100.0] * 4,
            "close": [102.0, 102.5, 102.0, 102.0],
            "volume": [1.0] * 4,
        }
    )
    bars["is_final_rth_bar"] = _mark_final_orb_rth_bars(bars)
    strategy = ORBStrategy()
    strategy._in_trade = True
    strategy._trade_state = ORBTradeState(
        entry_price=101.0,
        side="LONG",
        stop_price=99.0,
        target_price=105.0,
    )

    early_close = bars.iloc[1].to_dict()
    decision = strategy.on_market_data(early_close, position=object())
    assert bool(early_close["is_final_rth_bar"])
    assert decision.action is StrategyAction.EXIT
    assert "rth_close" in decision.reason
    strategy.on_exit()
    assert not strategy.in_trade
    assert strategy._trade_state is None

    # The early date closes at 12:59; it does not mark bars in the next date.
    assert not bool(bars.iloc[0]["is_final_rth_bar"])
    assert not bool(bars.iloc[2]["is_final_rth_bar"])
    assert bool(bars.iloc[3]["is_final_rth_bar"])


def _rth_bars_for_session(day: str, periods: int = 450) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{day} 09:30",
        periods=periods,
        freq="min",
        tz="America/New_York",
    )


@pytest.mark.parametrize(
    ("day", "expected_utc"),
    [
        ("2021-05-10", "2021-05-10 20:53Z"),
        ("2022-09-15", "2022-09-15 20:52Z"),
        ("2023-10-26", "2023-10-26 20:45Z"),
        ("2024-01-30", "2024-01-30 21:40Z"),
        ("2025-05-16", "2025-05-16 20:53Z"),
    ],
)
def test_s2r_research_horizon_excludes_five_known_extra_timestamps(
    day, expected_utc
):
    timestamps = _rth_bars_for_session(day)
    eligible = _mark_s2r_entry_eligible(
        pd.DataFrame({"timestamp ET": timestamps})
    )
    by_utc = pd.Series(eligible.to_numpy(), index=timestamps.tz_convert("UTC"))
    assert not bool(by_utc.loc[pd.Timestamp(expected_utc)])


def test_s2r_research_horizon_last_eligible_and_first_cutoff_bar():
    timestamps = _rth_bars_for_session("2024-01-30")
    eligible = _mark_s2r_entry_eligible(
        pd.DataFrame({"timestamp ET": timestamps})
    )
    by_local = pd.Series(eligible.to_numpy(), index=timestamps)
    assert bool(by_local.loc[pd.Timestamp("2024-01-30 16:39", tz="America/New_York")])
    assert not bool(by_local.loc[pd.Timestamp("2024-01-30 16:40", tz="America/New_York")])


def test_s2r_research_horizon_cutoff_is_session_local():
    first = _rth_bars_for_session("2024-01-08", periods=45)
    second = _rth_bars_for_session("2024-01-09", periods=50)
    timestamps = first.append(second)
    eligible = _mark_s2r_entry_eligible(
        pd.DataFrame({"timestamp ET": timestamps})
    )
    by_local = pd.Series(eligible.to_numpy(), index=timestamps)

    assert bool(by_local.loc[pd.Timestamp("2024-01-08 09:54", tz="America/New_York")])
    assert not bool(by_local.loc[pd.Timestamp("2024-01-08 09:55", tz="America/New_York")])
    assert bool(by_local.loc[pd.Timestamp("2024-01-09 09:59", tz="America/New_York")])
    assert not bool(by_local.loc[pd.Timestamp("2024-01-09 10:00", tz="America/New_York")])


def test_s2r_horizon_cutoff_prevents_qualifying_entry_signal():
    fitted = S2FittedModel(
        signal_model=S2SignalModel(
            thresholds={feature: 1.0 for feature in BASE_FEATURES},
            scales={feature: 1.0 for feature in BASE_FEATURES},
        ),
        volatility_reference=(1.0, 2.0, 3.0, 4.0),
    )
    strategy = S2RStrategy()
    strategy.set_fitted_models_for_bar({1: fitted})
    row = {
        "s2r_hmm_states": ((1, 2),),
        "realized_vol_30": 2.0,
        **{feature: 0.0 for feature in BASE_FEATURES},
    }

    row["s2r_entry_eligible"] = True
    assert strategy.generate_signal(row).value == "short"
    row["s2r_entry_eligible"] = False
    assert strategy.generate_signal(row).value == "flat"
    assert strategy.take_window_evaluations()[0]["reason"] == "research_horizon_cutoff"


def test_orb_attribution_omits_irrelevant_regime_features():
    timestamp = pd.Timestamp("2021-05-06 14:00:00Z")
    paper = pd.DataFrame(
        [{"strategy_name": "ORB", "entry_signal_timestamp": timestamp}]
    )
    market = pd.DataFrame(
        [{"timestamp": timestamp, "hmm_state": 2, "zscore": 1.0}]
    )

    attributed = add_paper_trade_attribution(paper, market, {})

    assert pd.isna(attributed.loc[0, "hmm_state"])
    assert pd.isna(attributed.loc[0, "zscore"])


def test_s2r_exit_reason_normalization_distinguishes_baseline_timeout():
    assert (
        _normalize_exit_reason(
            "S2 baseline timeout (NO_RECOVERY_ENRICHMENT / ORIGINAL_S2)"
        )
        == "timeout"
    )
    assert _normalize_exit_reason("S2R recovery recovered") == "recovered"
    assert _normalize_exit_reason("S2R recovery failed_to_recover") == (
        "failed_to_recover"
    )


def test_zscore_parity_compares_at_research_float32_artifact_precision():
    trade_key = "MRL1|2021-07-26T13:32:00Z"
    reference = pd.DataFrame(
        [{"trade_key": trade_key, "strategy_name": "MRL1", "zscore": -3.7524436}]
    )
    paper = pd.DataFrame(
        [
            {
                "trade_key": trade_key,
                "strategy_name": "MRL1",
                "zscore": -3.7524436549384985,
            }
        ]
    )
    assert compare_trade_ledgers(reference, paper).iloc[0]["status"] == "EXACT"

    paper.loc[0, "zscore"] = -3.7524
    assert compare_trade_ledgers(reference, paper).iloc[0]["first_difference_field"] == (
        "zscore"
    )
