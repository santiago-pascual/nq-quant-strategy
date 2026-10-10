from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import pytest

from src.paper.cme_calendar import CalendarUnavailable, CMECalendarSnapshot, CMETradingCalendar
from src.paper.ibkr_delayed_acquisition import (
    AppendOnlyGapClassificationLedger,
    AcquisitionCursor,
    AtomicAcquisitionCursor,
    execute_with_bounded_reconnect,
    HistoricalRequestPacer,
    HistoricalRequestPlanner,
    DelayedAcquisitionSession,
    rehydrate_pending_from_observations,
    format_ibkr_end_time,
)
from src.paper.ibkr_delayed_market_data import (
    AppendOnlyFinalizationLedger,
    DelayedBarFinalizer,
    FinalizationPolicy,
    IBKRContractSchedule,
    IBKRContractWindow,
)
from src.paper.ibkr_observation_ledger import AppendOnlyBarObservationLedger
from src.paper.ibkr_observation_ledger import load_observation_ledger
from src.paper.ibkr_paper_runner import IBKRReconnectBackoff, IBKRCursorAcquisitionController, IBKRTransportUnavailable


@contextmanager
def _test_dir(prefix):
    parent = Path("results/paper/ibkr_diagnostics")
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f".test-{prefix}-{uuid.uuid4().hex}"
    target.mkdir()
    try:
        yield target
    finally:
        shutil.rmtree(target)


def test_confirmation_request_is_anchored_to_oldest_pending_not_moving_now():
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    planner = HistoricalRequestPlanner()
    cycle = planner.plan_cycle(now_utc=start + timedelta(minutes=45),
                               pending_bar_starts=[int(start.timestamp()),
                                                   int((start + timedelta(minutes=10)).timestamp())])
    assert [item.kind for item in cycle] == ["recent_discovery", "pending_confirmation"]
    confirm = cycle[1]
    assert confirm.duration_seconds == 11 * 60
    assert confirm.covered_pending_starts[0] == int(start.timestamp())
    assert confirm.end_time_utc == start + timedelta(minutes=11)
    assert confirm.end_time_utc < start + timedelta(minutes=45)
    assert format_ibkr_end_time(confirm.end_time_utc) == "20261009 00:14:00 UTC"


def test_pending_confirmation_window_does_not_reach_back_across_maintenance_break():
    start = datetime(2026, 10, 8, 22, 24, tzinfo=timezone.utc)
    pending = [int(start.timestamp()), int((start + timedelta(minutes=4)).timestamp())]
    planner = HistoricalRequestPlanner()
    request = planner.plan_cycle(
        now_utc=datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc),
        pending_bar_starts=pending,
        include_discovery=False,
    )[0]

    request_start = int(request.end_time_utc.timestamp()) - request.duration_seconds
    assert request.kind == "pending_confirmation"
    assert request_start == pending[0]
    assert request.duration_seconds == 5 * 60
    assert request.end_time_utc == start + timedelta(minutes=5)
    assert request.covered_pending_starts == tuple(pending)


def test_pending_confirmation_windows_rotate_fairly_without_dropping_timestamps():
    start = datetime(2026, 10, 9, tzinfo=timezone.utc)
    pending = [int((start + timedelta(minutes=offset)).timestamp()) for offset in (0, 5, 45, 50)]
    planner = HistoricalRequestPlanner()
    first = planner.plan_cycle(now_utc=start + timedelta(hours=2), pending_bar_starts=pending,
                               confirmation_offset=0)[1]
    second = planner.plan_cycle(now_utc=start + timedelta(hours=2), pending_bar_starts=pending,
                                confirmation_offset=1)[1]
    assert first.covered_pending_starts == tuple(pending[:2])
    assert second.covered_pending_starts == tuple(pending[2:])
    assert first.confirmation_group_count == second.confirmation_group_count == 2
    assert sorted(first.covered_pending_starts + second.covered_pending_starts) == pending


def test_request_planner_rejects_future_or_naive_cycle_and_pacer_bounds_volume():
    planner = HistoricalRequestPlanner()
    with pytest.raises(ValueError, match="timezone-aware"):
        planner.plan_cycle(now_utc=datetime(2026, 10, 9), pending_bar_starts=[])
    pacer = HistoricalRequestPacer()
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    pacer.begin_cycle(now, 2)
    with pytest.raises(RuntimeError, match="too frequent"):
        pacer.begin_cycle(now + timedelta(seconds=20), 2)
    with pytest.raises(ValueError, match="at most two"):
        HistoricalRequestPacer().begin_cycle(now, 3)
    recovered_pacer = HistoricalRequestPacer(prior_request_epochs=[now.timestamp()])
    with pytest.raises(RuntimeError, match="too frequent"):
        recovered_pacer.begin_cycle(now + timedelta(seconds=20), 1)


def test_bootstrap_planner_requests_bounded_chronological_history_before_recent_discovery():
    planner = HistoricalRequestPlanner()
    now = datetime(2026, 10, 9, 1, 0, tzinfo=timezone.utc)
    first_missing = int(datetime(2026, 10, 8, 13, 3, tzinfo=timezone.utc).timestamp())
    safe_end = now - timedelta(minutes=11)
    requests = planner.plan_cycle(
        now_utc=now, pending_bar_starts=(), include_discovery=True,
        bootstrap_backfill_next_start_epoch_utc=first_missing,
        bootstrap_backfill_target_end_epoch_utc=first_missing + 3600,
        bootstrap_safe_end_utc=safe_end,
    )
    assert [request.kind for request in requests] == ["bootstrap_backfill"]
    request = requests[0]
    assert request.duration_seconds == 1800
    assert request.end_time_utc == datetime.fromtimestamp(first_missing + 1800, timezone.utc)
    assert request.end_time_utc < safe_end

    # The next cycle confirms old observations independently while the
    # acquisition cursor advances through another bounded historical window.
    requests = planner.plan_cycle(
        now_utc=now, pending_bar_starts=[first_missing], include_discovery=True,
        bootstrap_backfill_next_start_epoch_utc=first_missing + 1800,
        bootstrap_backfill_target_end_epoch_utc=first_missing + 5400,
        bootstrap_safe_end_utc=safe_end,
    )
    assert [request.kind for request in requests] == ["bootstrap_backfill", "pending_confirmation"]
    assert requests[0].end_time_utc == datetime.fromtimestamp(first_missing + 3600, timezone.utc)
    assert requests[1].covered_pending_starts == (first_missing,)


def test_atomic_cursor_recovers_pinned_contract_and_request_budget():
  with _test_dir("cursor") as tmp_path:
    path = tmp_path / "cursor.json"
    cursor = AtomicAcquisitionCursor(path, contract_id=815824267,
                                     local_symbol="MNQZ6", expiry="20261218")
    cursor.cursor.oldest_pending_bar_start_epoch_utc = 1791504000
    cursor.cursor.recent_request_epochs = (1791504100.0, 1791504130.0)
    cursor.save()
    recovered = AtomicAcquisitionCursor(path, contract_id=815824267,
                                        local_symbol="MNQZ6", expiry="20261218")
    assert recovered.cursor.oldest_pending_bar_start_epoch_utc == 1791504000
    assert recovered.cursor.recent_request_epochs == (1791504100.0, 1791504130.0)
    assert recovered.cursor.next_request_id == cursor.cursor.next_request_id
    with pytest.raises(ValueError, match="different MNQ contract"):
        AtomicAcquisitionCursor(path, contract_id=123, local_symbol="MNQH7", expiry="20270319")


def test_bootstrap_cursor_advances_only_to_observed_bar_boundary_and_reserves_request_id():
    from types import SimpleNamespace
    from src.paper.ibkr_paper_runner import IBKRCursorAcquisitionController

    with _test_dir("backfill-observed-cursor") as root:
        cursor_store = AtomicAcquisitionCursor(
            root / "cursor.json", contract_id=815824267,
            local_symbol="MNQZ6", expiry="20261218",
        )
        start = datetime(2026, 10, 8, 20, 33, tzinfo=timezone.utc)
        start_epoch = int(start.timestamp())
        cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc = start_epoch
        cursor_store.cursor.bootstrap_backfill_target_end_epoch_utc = start_epoch + 3600
        cursor_store.cursor.bootstrap_backfill_complete = False
        cursor_store.save()
        observation_path = root / "observations.jsonl"
        observation_path.write_text("", encoding="utf-8")

        class Session:
            def __init__(self):
                self.cursor_store = cursor_store
                self.cursor = cursor_store.cursor
                self.observation_ledger = SimpleNamespace(path=observation_path)
                self.finalizer = SimpleNamespace(
                    calendar=SimpleNamespace(expected_globex_minute=lambda _stamp: True)
                )

            def pending_confirmation_starts(self):
                return []

            def persist_request_budget(self, pacer, *, now):
                self.cursor.recent_request_epochs = pacer.recent_request_epochs
                self.cursor.last_request_at_utc = now.astimezone(timezone.utc).isoformat()
                self.cursor_store.save()

            def ingest_response(self, rows, **_kwargs):
                return []

        now = start + timedelta(hours=2)
        observed_rows = [
            {"timestamp": start_epoch - 180 + minute * 60}
            for minute in range(30)
        ]
        requested_ids = []

        def request(_request, request_id):
            requested_ids.append(request_id)
            # The cursor allocation is durable before TWS can return a callback.
            assert cursor_store.cursor.next_request_id == request_id + 1
            return {"completed": True, "connected": True,
                    "rows": observed_rows, "errors": []}

        controller = IBKRCursorAcquisitionController(
            session=Session(), request_historical=request,
            clock=lambda: now,
        )
        cycle = controller.refresh()

        assert requested_ids == [1_000_000]
        assert cycle["requests"][0]["kind"] == "bootstrap_backfill"
        # The IBKR response ended three minutes before the requested exclusive
        # boundary. The next cursor follows observed data, avoiding a skipped
        # interval caused by advancing to the request boundary.
        assert cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc == start_epoch + 27 * 60
        assert cursor_store.cursor.next_request_id == 1_000_001
        restored = AtomicAcquisitionCursor(
            root / "cursor.json", contract_id=815824267,
            local_symbol="MNQZ6", expiry="20261218",
        )
        assert restored.cursor.next_request_id == 1_000_001


def test_bootstrap_cursor_skips_only_calendar_verified_maintenance_closure():
    from types import SimpleNamespace
    from src.paper.ibkr_paper_runner import IBKRCursorAcquisitionController

    with _test_dir("verified-maintenance-skip") as root:
        calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(
            Path("src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json")
        ))
        cursor_store = AtomicAcquisitionCursor(
            root / "cursor.json", contract_id=815824267,
            local_symbol="MNQZ6", expiry="20261218",
        )
        start = datetime(2026, 10, 8, 21, 3, tzinfo=timezone.utc)
        target = datetime(2026, 10, 9, 21, 0, tzinfo=timezone.utc)
        cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc = int(start.timestamp())
        cursor_store.cursor.bootstrap_backfill_target_end_epoch_utc = int(target.timestamp())
        cursor_store.cursor.bootstrap_backfill_complete = False
        cursor_store.save()

        controller = object.__new__(IBKRCursorAcquisitionController)
        controller.session = SimpleNamespace(
            cursor=cursor_store.cursor, cursor_store=cursor_store,
            finalizer=SimpleNamespace(calendar=calendar),
        )
        controller.clock = lambda: datetime(2026, 10, 10, 1, 0, tzinfo=timezone.utc)

        skipped = controller._advance_verified_calendar_closure()
        assert skipped is not None
        assert skipped["classification"] == "verified_exchange_closure"
        assert skipped["from_utc"] == "2026-10-08T21:03:00+00:00"
        assert skipped["to_utc"] == "2026-10-08T22:00:00+00:00"
        assert skipped["minutes"] == 57
        assert skipped["calendar_identity"] == calendar.snapshot.identity
        assert cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc == int(
            datetime(2026, 10, 8, 22, 0, tzinfo=timezone.utc).timestamp()
        )
        assert cursor_store.cursor.bootstrap_backfill_complete is False
        assert calendar.expected_globex_minute(datetime(2026, 10, 8, 22, 0, tzinfo=timezone.utc))


def test_atomic_cursor_retries_one_transient_windows_replace_lock(monkeypatch):
    import src.paper.ibkr_delayed_acquisition as acquisition

    with _test_dir("atomic-retry") as tmp_path:
        path = tmp_path / "cursor.json"
        cursor = AtomicAcquisitionCursor(path, contract_id=815824267,
                                         local_symbol="MNQZ6", expiry="20261218")
        cursor.cursor.last_observation_sequence = 7
        real_replace = acquisition.os.replace
        attempts = {"count": 0}

        def transient_lock(source, destination):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise PermissionError("simulated transient Windows sharing lock")
            return real_replace(source, destination)

        monkeypatch.setattr(acquisition.os, "replace", transient_lock)
        cursor.save()
        assert attempts["count"] == 2
        restored = AtomicAcquisitionCursor(path, contract_id=815824267,
                                           local_symbol="MNQZ6", expiry="20261218")
        assert restored.cursor.last_observation_sequence == 7


def test_pacer_refuses_more_than_configured_rolling_budget():
    now = datetime(2026, 10, 9, tzinfo=timezone.utc)
    pacer = HistoricalRequestPacer(max_requests_per_10_minutes=4)
    for cycle in range(2):
        pacer.begin_cycle(now + timedelta(seconds=cycle * 30), 2)
    with pytest.raises(RuntimeError, match="rolling budget"):
        pacer.begin_cycle(now + timedelta(seconds=60), 1)


def test_disconnect_reconnect_retries_once_but_provider_error_does_not_loop():
    attempts = iter(({"completed": False, "connected": False},
                     {"completed": True, "connected": True, "rows": ["bar"]}))
    reconnect_calls = []
    result, count = execute_with_bounded_reconnect(lambda: next(attempts),
        lambda: reconnect_calls.append("reconnected"))
    assert result["rows"] == ["bar"]
    assert count == 1
    assert reconnect_calls == ["reconnected"]

    reconnect_calls.clear()
    result, count = execute_with_bounded_reconnect(
        lambda: {"completed": False, "connected": True, "errors": ["provider pacing error"]},
        lambda: reconnect_calls.append("should-not-run"))
    assert count == 0
    assert result["errors"] == ["provider pacing error"]
    assert reconnect_calls == []


def test_reconnect_backoff_is_indefinite_bounded_and_jittered():
    policy = IBKRReconnectBackoff(random_value=lambda: 1.0)
    assert [policy.next_delay() for _ in range(8)] == [6.0, 12.0, 24.0, 48.0, 60.0, 60.0, 60.0, 60.0]


def test_controller_recovers_after_repeated_1100_and_handshake_failures_without_cursor_advance():
    from types import SimpleNamespace
    from src.paper.ibkr_paper_runner import _is_transient_ibkr_disconnect

    with _test_dir("indefinite-reconnect") as root:
        cursor_store = AtomicAcquisitionCursor(
            root / "cursor.json", contract_id=815824267,
            local_symbol="MNQZ6", expiry="20261218",
        )
        cursor_store.cursor.bootstrap_backfill_complete = True
        cursor_store.cursor.last_delivered_bar_start_epoch_utc = 1_791_575_940
        cursor_store.save()
        observations = root / "observations.jsonl"
        observations.write_text("", encoding="utf-8")

        class Session:
            def __init__(self):
                self.cursor = cursor_store.cursor
                self.cursor_store = cursor_store
                self.observation_ledger = SimpleNamespace(path=observations)
                self.finalizer = SimpleNamespace(last_delivered=None,
                    calendar=SimpleNamespace(expected_globex_minute=lambda _stamp: True))

            def pending_confirmation_starts(self):
                return []

            def persist_request_budget(self, pacer, *, now):
                self.cursor.recent_request_epochs = pacer.recent_request_epochs
                self.cursor.last_request_at_utc = now.astimezone(timezone.utc).isoformat()
                self.cursor_store.save()

            def ingest_response(self, rows, **kwargs):
                assert rows == []
                return []

        base = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
        now = [base]
        request_results = iter((
            {"completed": False, "connected": True,
             "errors": [{"request_id": -1, "code": 1100, "message": "connectivity lost"}]},
            {"completed": False, "connected": True,
             "errors": [{"request_id": -1, "code": 1100, "message": "connectivity lost again"}]},
            {"completed": True, "connected": True, "rows": [], "errors": []},
        ))
        reconnect_outcomes = iter((
            IBKRTransportUnavailable("TWS handshake timeout"), None, None,
        ))
        reconnect_attempts = []
        request_attempts = []
        events = []

        def reconnect():
            reconnect_attempts.append(now[0])
            outcome = next(reconnect_outcomes)
            if outcome is not None:
                raise outcome

        def request(*_):
            request_attempts.append(now[0])
            return next(request_results)

        controller = IBKRCursorAcquisitionController(
            session=Session(), request_historical=request, reconnect=reconnect,
            clock=lambda: now[0], initially_connected=True,
            backoff=IBKRReconnectBackoff(random_value=lambda: 0.0),
            on_event=lambda event_type, payload, **kw: events.append((event_type.value, dict(payload))),
        )
        before_cursor = cursor_store.cursor.last_delivered_bar_start_epoch_utc
        initial_request_id = cursor_store.cursor.next_request_id

        first = controller.refresh(datetime.fromtimestamp(
            cursor_store.cursor.last_delivered_bar_start_epoch_utc, timezone.utc))
        assert first["transport_available"] is False
        assert first["connection_state"] == "RECONNECTING"
        assert first["retry_count"] == 0
        assert first["next_retry_at_utc"] == (base + timedelta(seconds=5)).isoformat()
        assert cursor_store.cursor.bootstrap_backfill_complete is False
        assert cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc == before_cursor + 60
        assert _is_transient_ibkr_disconnect({"completed": False, "connected": True,
            "errors": [{"code": 1100, "message": "Connectivity lost"}]})

        persisted_incident = cursor_store.cursor.reconnect_incident_id
        persisted_retry = cursor_store.cursor.reconnect_next_retry_at_utc
        controller = IBKRCursorAcquisitionController(
            session=Session(), request_historical=request, reconnect=reconnect,
            clock=lambda: now[0], initially_connected=False,
            backoff=IBKRReconnectBackoff(random_value=lambda: 0.0),
            on_event=lambda event_type, payload, **kw: events.append((event_type.value, dict(payload))),
        )
        assert controller.incident_id == persisted_incident
        assert controller.next_retry_at_utc.isoformat() == persisted_retry
        assert controller.backoff.failures == 1
        assert [kind for kind, _ in events] == ["feed_disconnected"]

        now[0] = base + timedelta(seconds=4)
        early = controller.refresh()
        assert early["transport_available"] is False
        assert reconnect_attempts == []

        now[0] = base + timedelta(seconds=6)
        failed_handshake = controller.refresh()
        assert failed_handshake["retry_count"] == 1
        assert failed_handshake["next_retry_at_utc"] == (base + timedelta(seconds=16)).isoformat()
        assert len(request_attempts) == 1

        # Reconnect handshake succeeds, but historical pacing still prevents
        # another request until 30 seconds after the previous cycle.
        now[0] = base + timedelta(seconds=16)
        handshake_only = controller.refresh()
        assert handshake_only["connection_state"] == "RECOVERING"
        assert handshake_only["last_successful_handshake_utc"] == now[0].isoformat()
        assert handshake_only["next_retry_at_utc"] == (base + timedelta(seconds=30)).isoformat()
        assert len(request_attempts) == 1

        now[0] = base + timedelta(seconds=30)
        second_disconnect = controller.refresh()
        assert second_disconnect["connection_state"] == "RECONNECTING"
        assert second_disconnect["next_retry_at_utc"] == (base + timedelta(seconds=50)).isoformat()
        assert len(request_attempts) == 2

        now[0] = base + timedelta(seconds=50)
        second_handshake = controller.refresh()
        assert second_handshake["next_retry_at_utc"] == (base + timedelta(seconds=60)).isoformat()
        assert len(request_attempts) == 2

        now[0] = base + timedelta(seconds=60)
        caught_up = controller.refresh()
        assert caught_up["transport_available"] is True
        assert caught_up["last_successful_request_utc"] == now[0].isoformat()
        assert caught_up["connection_state"] == "RECOVERING"
        assert cursor_store.cursor.last_delivered_bar_start_epoch_utc == before_cursor
        assert cursor_store.cursor.next_request_id > initial_request_id
        assert len(request_attempts) == 3
        assert len(reconnect_attempts) == 3

        # Recovery is recorded once, only after the caller confirms that
        # acquisition and Paper delivery have caught up.
        cursor_store.cursor.bootstrap_backfill_complete = True
        cursor_store.cursor.bootstrap_backfill_target_end_epoch_utc = before_cursor + 60
        cursor_store.save()
        controller.mark_recovery_verified()
        controller.mark_recovery_verified()
        assert controller.connection_state == "CONNECTED"
        assert [kind for kind, _ in events] == ["feed_disconnected", "feed_connected"]


def test_ibkr_empty_hmds_response_is_closure_only_when_entire_window_is_calendar_closed():
    from types import SimpleNamespace
    from src.paper.ibkr_paper_runner import IBKRCursorAcquisitionController

    project = Path(__file__).resolve().parents[2]
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(
        project / "src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"))

    def run_at(now):
        with _test_dir("hmds-empty") as root:
            cursor_store = AtomicAcquisitionCursor(root / "cursor.json", contract_id=815824267,
                local_symbol="MNQZ6", expiry="20261218")
            cursor_store.cursor.bootstrap_backfill_complete = True
            cursor_store.save()
            observation_path = root / "observations.jsonl"
            observation_path.write_text("", encoding="utf-8")

            class Session:
                def __init__(self):
                    self.cursor_store = cursor_store
                    self.cursor = cursor_store.cursor
                    self.observation_ledger = SimpleNamespace(path=observation_path)
                    self.finalizer = SimpleNamespace(calendar=calendar)

                def pending_confirmation_starts(self):
                    return []

                def persist_request_budget(self, pacer, *, now):
                    self.cursor.recent_request_epochs = pacer.recent_request_epochs
                    self.cursor_store.save()

                def ingest_response(self, rows, **kwargs):
                    assert rows == []
                    return []

            controller = IBKRCursorAcquisitionController(
                session=Session(), request_historical=lambda *_: {
                    "completed": False, "connected": True,
                    "errors": [{"code": 162, "message": "HMDS query returned no data"}],
                }, clock=lambda: now, initially_connected=True,
                backoff=IBKRReconnectBackoff(random_value=lambda: 0.0),
            )
            before = (cursor_store.cursor.last_observation_sequence,
                      cursor_store.cursor.last_delivered_bar_start_epoch_utc)
            result = controller.refresh()
            after = (cursor_store.cursor.last_observation_sequence,
                     cursor_store.cursor.last_delivered_bar_start_epoch_utc)
            return result, before, after

    closed, closed_before, closed_after = run_at(datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc))
    assert closed["transport_available"] is True
    assert closed["acquisition_state"] != "WAITING_FOR_PROVIDER_DATA"
    assert closed_before == closed_after == (0, None)

    open_result, open_before, open_after = run_at(datetime(2026, 10, 12, 22, 30, tzinfo=timezone.utc))
    assert open_result["transport_available"] is True
    assert open_result["acquisition_state"] == "WAITING_FOR_PROVIDER_DATA"
    assert open_result["next_retry_at_utc"] is not None
    assert open_before == open_after == (0, None)


def test_tws_handshake_failure_is_reported_with_callback_diagnostics_and_cleaned(monkeypatch):
    import sys
    from types import ModuleType
    from src.paper.ibkr_paper_runner import IBKRTransportUnavailable, TWSHistoricalTRADESClient

    disconnected = []

    class FakeEClient:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.connected = False

        def connect(self, host, port, client_id):
            self.endpoint = (host, port, client_id)
            self.connected = True

        def run(self):
            self.wrapper.error(-1, 502, "Could not connect to TWS API socket")

        def isConnected(self):
            return self.connected

        def disconnect(self):
            disconnected.append(True)
            self.connected = False

    class FakeEWrapper:
        pass

    class FakeContract:
        pass

    ibapi = ModuleType("ibapi")
    client = ModuleType("ibapi.client")
    contract = ModuleType("ibapi.contract")
    wrapper = ModuleType("ibapi.wrapper")
    client.EClient = FakeEClient
    contract.Contract = FakeContract
    wrapper.EWrapper = FakeEWrapper
    monkeypatch.setitem(sys.modules, "ibapi", ibapi)
    monkeypatch.setitem(sys.modules, "ibapi.client", client)
    monkeypatch.setitem(sys.modules, "ibapi.contract", contract)
    monkeypatch.setitem(sys.modules, "ibapi.wrapper", wrapper)

    with pytest.raises(IBKRTransportUnavailable, match=r"127\.0\.0\.1:7497.*502.*Could not connect"):
        TWSHistoricalTRADESClient(
            host="127.0.0.1", port=7497, client_id=198,
            con_id=815824267, local_symbol="MNQZ6", expiry="20261218",
            timeout_seconds=0.5,
        )
    assert disconnected == [True]


def test_transient_tws_handshake_failure_schedules_retry_without_advancing_cursor():
    from types import SimpleNamespace
    from src.paper.ibkr_paper_runner import (
        IBKRCursorAcquisitionController,
        IBKRTransportUnavailable,
    )

    with _test_dir("handshake-retry") as root:
        cursor_store = AtomicAcquisitionCursor(
            root / "cursor.json", contract_id=815824267,
            local_symbol="MNQZ6", expiry="20261218",
        )
        cursor_store.cursor.bootstrap_backfill_complete = True
        cursor_store.save()
        observations = root / "observations.jsonl"
        observations.write_text("", encoding="utf-8")

        class Session:
            def __init__(self):
                self.cursor = cursor_store.cursor
                self.cursor_store = cursor_store
                self.observation_ledger = SimpleNamespace(path=observations)

            def pending_confirmation_starts(self):
                return []

            def persist_request_budget(self, pacer, *, now):
                self.cursor.recent_request_epochs = pacer.recent_request_epochs
                self.cursor.last_request_at_utc = now.astimezone(timezone.utc).isoformat()
                self.cursor_store.save()

        reconnects = []

        def reconnect():
            reconnects.append(True)
            raise IBKRTransportUnavailable(
                "TWS API handshake failed at 127.0.0.1:7497 (client_id=198): callbacks=[502]"
            )

        now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
        controller = IBKRCursorAcquisitionController(
            session=Session(),
            request_historical=lambda *_: {
                "completed": False, "connected": False, "rows": [], "errors": []
            },
            reconnect=reconnect, clock=lambda: now,
        )
        before = cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc
        cycle = controller.refresh()

        assert reconnects == []
        assert cycle["transport_available"] is False
        assert "disconnected" in cycle["last_transport_error"].lower()
        assert cycle["connection_state"] == "RECONNECTING"
        retry_at = datetime.fromisoformat(cycle["next_retry_at_utc"])
        assert 5 <= (retry_at - now).total_seconds() <= 6
        assert cursor_store.cursor.bootstrap_backfill_next_start_epoch_utc == before
        assert cursor_store.cursor.last_observation_sequence == 0


def test_duplicate_and_out_of_order_responses_are_sorted_before_finalization():
  with _test_dir("ledger") as tmp_path:
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    schedule = IBKRContractSchedule([IBKRContractWindow(815824267, "MNQZ6", start, end)])
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="test-oct", source="https://example.test/reviewed-fixture", coverage_start=date(2026, 10, 8),
        coverage_end=date(2026, 10, 31)))
    final_path = tmp_path / "final.jsonl"
    final_ledger = AppendOnlyFinalizationLedger(final_path)
    finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
                                    policy=FinalizationPolicy(), finalization_ledger=final_ledger)
    # A later bar arrives first. Once the earlier bar arrives it remains the
    # delivery cursor and both are emitted in exchange-time order after stable polls.
    values = {"open": 20000, "high": 20001, "low": 19999, "close": 20000, "volume": 5}
    for poll, seen in ((1, 700), (2, 730)):
        for stamp in (start + timedelta(minutes=1), start):
            finalizer.observe({"timestamp": int(stamp.timestamp()), **values}, contract_id=815824267,
                              local_symbol="MNQZ6", observed_at=start + timedelta(seconds=seen),
                              poll_number=poll)
        emitted = finalizer.finalize_ready(now=start + timedelta(seconds=seen))
    assert [item.exchange_bar_timestamp_utc for item in emitted] == [start, start + timedelta(minutes=1)]
    # Reconstructing from the durable delivery ledger prevents duplicate output.
    resumed = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
                                  policy=FinalizationPolicy(),
                                  finalization_ledger=AppendOnlyFinalizationLedger(final_path))
    assert resumed.finalize_ready(now=start + timedelta(minutes=20)) == []
    assert len(final_path.read_text(encoding="utf-8").splitlines()) == 2


def test_permanent_expected_gap_blocks_until_operator_classifies_with_evidence():
  with _test_dir("gap") as tmp_path:
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    schedule = IBKRContractSchedule([IBKRContractWindow(815824267, "MNQZ6", start,
                                                        start + timedelta(hours=1))])
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="test-oct", source="https://example.test/reviewed-fixture",
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31)))
    missing = start + timedelta(minutes=1)
    values = {"open": 20000, "high": 20001, "low": 19999, "close": 20000, "volume": 5}
    f = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar, policy=FinalizationPolicy())
    for stamp in (start, start + timedelta(minutes=2)):
        for poll, sec in ((1, 760), (2, 790)):
            f.observe({"timestamp": int(stamp.timestamp()), **values}, contract_id=815824267,
                      local_symbol="MNQZ6", observed_at=start + timedelta(seconds=sec), poll_number=poll)
    assert len(f.finalize_ready(now=start + timedelta(seconds=790))) == 1
    assert f.pending_timestamps == (start + timedelta(minutes=2),)
    with pytest.raises(ValueError, match="requires operator_reviewed"):
        AppendOnlyGapClassificationLedger(tmp_path / "gaps.jsonl").append(
            int(missing.timestamp()), classification="operator_reviewed_missing_data",
            evidence="", reviewed_by="test")
    gaps = AppendOnlyGapClassificationLedger(tmp_path / "gaps.jsonl")
    gaps.append(int(missing.timestamp()), classification="operator_reviewed_missing_data",
                evidence="provider response repeatedly omitted this minute", reviewed_by="tester")
    resumed = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
        policy=FinalizationPolicy(), gap_classifications=gaps.rows)
    for stamp in (start, start + timedelta(minutes=2)):
        for poll, sec in ((1, 760), (2, 790)):
            resumed.observe({"timestamp": int(stamp.timestamp()), **values}, contract_id=815824267,
                            local_symbol="MNQZ6", observed_at=start + timedelta(seconds=sec), poll_number=poll)
    emitted = resumed.finalize_ready(now=start + timedelta(seconds=790))
    assert [item.exchange_bar_timestamp_utc for item in emitted] == [start, start + timedelta(minutes=2)]
    assert any(event["kind"] == "explicitly_classified_expected_gap" for event in resumed.quality_events)


def test_calendar_coverage_expiration_fails_closed():
    start = datetime(2026, 11, 2, 0, 3, tzinfo=timezone.utc)
    f = DelayedBarFinalizer(contract_schedule=IBKRContractSchedule([
        IBKRContractWindow(815824267, "MNQZ6", start - timedelta(hours=1), start + timedelta(hours=1))]),
        calendar=CMETradingCalendar(CMECalendarSnapshot(
            version="test-oct", source="https://example.test/reviewed-fixture",
            coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31))))
    with pytest.raises(CalendarUnavailable, match="does not cover"):
        f.observe({"timestamp": int(start.timestamp()), "open": 20000, "high": 20001,
                   "low": 19999, "close": 20000, "volume": 2},
                  contract_id=815824267, local_symbol="MNQZ6",
                  observed_at=start + timedelta(minutes=12), poll_number=1)


def test_observation_ledger_keeps_revisions_and_cursor_checkpoint_is_separate():
  with _test_dir("revisions") as tmp_path:
    observation = AppendOnlyBarObservationLedger(tmp_path / "observed.jsonl", run_id="test")
    cursor = AtomicAcquisitionCursor(tmp_path / "cursor.json", contract_id=815824267,
                                     local_symbol="MNQZ6", expiry="20261218")
    bar_start = int(datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc).timestamp())
    base = {"timestamp": bar_start, "open": 20000, "high": 20001, "low": 19999,
            "close": 20000, "volume": 5}
    args = {"contract": {"con_id": 815824267, "local_symbol": "MNQZ6", "expiry": "20261218"},
            "observed_at": datetime(2026, 10, 9, 0, 20, tzinfo=timezone.utc),
            "request_id": 1, "poll_number": 1}
    observation.append(base, **args)
    observation.append({**base, "close": 19999}, **{**args, "request_id": 2, "poll_number": 2})
    cursor.cursor.last_observation_sequence = 2
    cursor.cursor.oldest_pending_bar_start_epoch_utc = bar_start
    cursor.save()
    lines = (tmp_path / "observed.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"is_revision":true' in lines[1]
    restored = AtomicAcquisitionCursor(tmp_path / "cursor.json", contract_id=815824267,
                                       local_symbol="MNQZ6", expiry="20261218")
    assert restored.cursor.last_observation_sequence == 2
    assert restored.cursor.oldest_pending_bar_start_epoch_utc == bar_start


def test_restart_rehydrates_pending_observations_and_never_delivers_twice():
  with _test_dir("restart") as tmp_path:
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    schedule = IBKRContractSchedule([IBKRContractWindow(815824267, "MNQZ6", start - timedelta(hours=1),
                                                        start + timedelta(hours=1))])
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="test-oct", source="https://example.test/reviewed-fixture",
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31)))
    observation_path, delivery_path, cursor_path = (tmp_path / "observations.jsonl",
                                                    tmp_path / "deliveries.jsonl", tmp_path / "cursor.json")
    observations = AppendOnlyBarObservationLedger(observation_path, run_id="restart-test")
    cursor_store = AtomicAcquisitionCursor(cursor_path, contract_id=815824267,
                                           local_symbol="MNQZ6", expiry="20261218")
    finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
        policy=FinalizationPolicy(), finalization_ledger=AppendOnlyFinalizationLedger(delivery_path))
    session = DelayedAcquisitionSession(cursor_store=cursor_store,
                                         observation_ledger=observations, finalizer=finalizer)
    values = {"open": 20000, "high": 20001, "low": 19999, "close": 20000, "volume": 5}
    base = {"con_id": 815824267, "timestamp": int(start.timestamp()), **values}
    session.ingest_response([base], observed_at=start + timedelta(seconds=660), poll_number=1, request_id=1)

    # Simulated process restart: load all durable state and reconstruct pending.
    resumed_cursor = AtomicAcquisitionCursor(cursor_path, contract_id=815824267,
                                              local_symbol="MNQZ6", expiry="20261218")
    resumed_observations = AppendOnlyBarObservationLedger(observation_path, run_id="restart-test")
    resumed_finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
        policy=FinalizationPolicy(), finalization_ledger=AppendOnlyFinalizationLedger(delivery_path))
    rehydrate_pending_from_observations(resumed_finalizer, load_observation_ledger(observation_path))
    resumed = DelayedAcquisitionSession(cursor_store=resumed_cursor,
        observation_ledger=resumed_observations, finalizer=resumed_finalizer)
    output = resumed.ingest_response([base], observed_at=start + timedelta(seconds=690),
                                     poll_number=2, request_id=2)
    assert [item.exchange_bar_timestamp_utc for item in output] == [start]

    # Another restart sees the durable delivery journal; a duplicate response
    # cannot produce a second Paper observation.
    last_finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
        policy=FinalizationPolicy(), finalization_ledger=AppendOnlyFinalizationLedger(delivery_path))
    rehydrate_pending_from_observations(last_finalizer, load_observation_ledger(observation_path))
    assert last_finalizer.finalize_ready(now=start + timedelta(minutes=20)) == []
    assert len(delivery_path.read_text(encoding="utf-8").splitlines()) == 1


def test_confirmation_window_observations_outside_pending_set_are_audited_but_not_candidates():
  with _test_dir("confirmation-filter") as tmp_path:
    start = datetime(2026, 10, 9, 0, 3, tzinfo=timezone.utc)
    schedule = IBKRContractSchedule([IBKRContractWindow(815824267, "MNQZ6", start - timedelta(hours=1),
                                                        start + timedelta(hours=1))])
    calendar = CMETradingCalendar(CMECalendarSnapshot(
        version="test-oct", source="https://example.test/reviewed-fixture",
        coverage_start=date(2026, 10, 8), coverage_end=date(2026, 10, 31)))
    obs_path = tmp_path / "observations.jsonl"
    values = {"open": 20000, "high": 20001, "low": 19999, "close": 20000, "volume": 5}
    rows = [{"con_id": 815824267, "timestamp": int((start + timedelta(minutes=offset)).timestamp()), **values}
            for offset in (0, 1)]
    ledger = AppendOnlyBarObservationLedger(obs_path, run_id="confirmation-filter")
    finalizer = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
                                    policy=FinalizationPolicy())
    session = DelayedAcquisitionSession(
        cursor_store=AtomicAcquisitionCursor(tmp_path / "cursor.json", contract_id=815824267,
                                             local_symbol="MNQZ6", expiry="20261218"),
        observation_ledger=ledger, finalizer=finalizer)
    session.ingest_response(rows, observed_at=start + timedelta(minutes=20),
        poll_number=1, request_id=1, finalize_timestamps={int(start.timestamp())})
    stored = load_observation_ledger(obs_path)
    assert len(stored) == 2
    assert [row["eligible_for_finalization"] for row in stored] == [True, False]
    resumed = DelayedBarFinalizer(contract_schedule=schedule, calendar=calendar,
                                  policy=FinalizationPolicy())
    rehydrate_pending_from_observations(resumed, stored)
    assert resumed.pending_timestamps == (start,)
