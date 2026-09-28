from __future__ import annotations

import pandas as pd

from src.models.windowed_regime import (
    build_research_hmm_schedule_metadata,
    load_frozen_research_hmm_windows,
    research_hmm_window_for_timestamp,
)
from src.paper.autonomous_runner import (
    FROZEN_HMM_SCHEDULE,
    identify_final_rth_bars,
)
from src.strategies.orb.config import ORBConfig, ORBExitReason
from src.strategies.orb.lifecycle import ORBLifecycle, ORBTradeState
from src.strategies.orb.strategy import is_final_rth_bar


def test_persisted_schedule_has_22_ordered_non_overlapping_windows():
    windows = load_frozen_research_hmm_windows(FROZEN_HMM_SCHEDULE)

    assert len(windows) == 22
    assert [window.window for window in windows] == list(range(1, 23))
    assert all(
        left.oos_start <= left.oos_end < right.oos_start
        for left, right in zip(windows, windows[1:])
    )
    assert research_hmm_window_for_timestamp(
        windows, windows[0].oos_start
    ) == windows[0]
    assert research_hmm_window_for_timestamp(
        windows, windows[10].oos_end
    ) == windows[10]
    assert research_hmm_window_for_timestamp(
        windows, windows[-1].oos_end + pd.Timedelta(minutes=1)
    ) is None


def test_frozen_schedule_boundaries_do_not_move_when_future_rows_are_appended(
    tmp_path,
):
    timestamps = pd.date_range(
        "2024-01-01",
        periods=44,
        freq="min",
        tz="UTC",
    )
    baseline = pd.DataFrame(
        {"timestamp": timestamps, "zscore_30": [0.0] * len(timestamps)}
    )
    persisted = build_research_hmm_schedule_metadata(baseline)
    schedule_path = tmp_path / "frozen_schedule.csv"
    persisted.to_csv(schedule_path, index=False)
    original_windows = load_frozen_research_hmm_windows(schedule_path)

    future_timestamps = pd.date_range(
        timestamps[-1] + pd.Timedelta(minutes=1),
        periods=44,
        freq="min",
    )
    appended = pd.concat(
        [
            baseline,
            pd.DataFrame(
                {"timestamp": future_timestamps, "zscore_30": [0.0] * 44}
            ),
        ],
        ignore_index=True,
    )
    recalculated = build_research_hmm_schedule_metadata(appended)
    assert not recalculated["oos_start"].equals(persisted["oos_start"])

    resolved_after_append = load_frozen_research_hmm_windows(schedule_path)
    assert resolved_after_append == original_windows


def test_final_rth_bar_uses_each_new_york_session_actual_last_bar():
    eastern = "America/New_York"
    timestamps = [
        pd.Timestamp("2024-01-08 09:30", tz=eastern),
        pd.Timestamp("2024-01-08 15:59", tz=eastern),
        pd.Timestamp("2024-01-08 16:00", tz=eastern),
        pd.Timestamp("2024-01-12 09:30", tz=eastern),
        pd.Timestamp("2024-01-12 12:59", tz=eastern),
        pd.Timestamp("2024-01-12 16:00", tz=eastern),
    ]
    final_bars = identify_final_rth_bars(
        pd.DataFrame({"timestamp": timestamps})
    )

    normal_close = timestamps[1].tz_convert("UTC")
    early_close = timestamps[4].tz_convert("UTC")
    assert final_bars == frozenset({normal_close, early_close})

    config = ORBConfig()
    assert is_final_rth_bar(
        {"timestamp": normal_close, "is_final_rth_bar": True}, config
    )
    assert is_final_rth_bar(
        {"timestamp": early_close, "is_final_rth_bar": True}, config
    )
    assert not is_final_rth_bar(
        {"timestamp": timestamps[3].tz_convert("UTC"), "is_final_rth_bar": False},
        config,
    )


def test_orb_target_and_stop_precede_final_session_close():
    lifecycle = ORBLifecycle(ORBConfig())
    state = lifecycle.create_trade(
        side="LONG",
        entry_price=101.0,
        or_high=101.0,
        or_low=99.0,
    )

    target = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=100.0,
        close=104.0,
        is_rth_close=True,
    )
    assert target is not None
    assert target.reason is ORBExitReason.TARGET
    assert target.exit_price == 105.0

    stop = lifecycle.evaluate_bar(
        state,
        high=105.0,
        low=99.0,
        close=104.0,
        is_rth_close=True,
    )
    assert stop is not None
    assert stop.reason is ORBExitReason.STOP
    assert stop.exit_price == 99.0


def test_orb_exits_at_close_of_marked_final_available_bar():
    lifecycle = ORBLifecycle(ORBConfig())
    state = ORBTradeState(
        entry_price=101.0,
        side="LONG",
        stop_price=99.0,
        target_price=105.0,
    )
    for is_final in (False, True):
        result = lifecycle.evaluate_bar(
            state,
            high=104.0,
            low=100.0,
            close=102.5,
            is_rth_close=is_final,
        )
        if not is_final:
            assert result is None
        else:
            assert result is not None
            assert result.reason is ORBExitReason.RTH_CLOSE
            assert result.exit_price == 102.5
