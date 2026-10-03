from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd
import pytest

from src.models.regime import VolatilityRegimeModel
from src.paper.research_replay import (
    DEFAULT_END,
    DEFAULT_START,
    PARITY_FIELDS,
    ResearchReplayContextAdapter,
    _ReplayEventLogger,
    _build_paper_engine,
    _fit_s2r_models,
    _mark_final_orb_rth_bars,
    _paper_trades_from_events,
    add_paper_trade_attribution,
    build_argument_parser,
    build_reference_trades,
    compare_trade_ledgers,
    load_frozen_hmm_states,
    parse_oos_range,
    research_causal_expanding_percentile,
)
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


def test_s2r_schedule_assigns_shared_boundary_to_next_window():
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
            "hmm_state": [2] * len(timestamps),
            "realized_vol_30": [0.1, 0.2, 0.3, 0.4, 0.5],
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
                "validation_start": "2022-01-01T00:00:00Z",
                "validation_end": "2022-01-01T00:02:00Z",
            },
            {
                "window": 2,
                "validation_start": "2022-01-01T00:02:00Z",
                "validation_end": "2022-01-01T00:03:00Z",
            },
        ]
    )

    models, mapping = _fit_s2r_models(market, reference)

    assert sorted(models) == [1, 2]
    assert mapping["s2r_window"].tolist() == [-1, -1, 1, 1, 2]


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
        "MRL1": "MATCH",
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


def test_final_orb_rth_marker_uses_each_sessions_last_orb_bar():
    timestamps = pd.DatetimeIndex(
        [
            "2020-09-07 09:30",
            "2020-09-07 12:58",
            "2020-09-07 12:59",
            "2020-09-08 09:30",
            "2020-09-08 15:58",
            "2020-09-08 15:59",
            "2020-09-08 16:00",
            "2020-09-08 16:59",
        ]
    ).tz_localize("America/New_York")
    rth = pd.DataFrame({"timestamp ET": timestamps})

    assert _mark_final_orb_rth_bars(rth).tolist() == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]


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
