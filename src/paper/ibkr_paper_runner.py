"""Read-only IBKR acquisition controller for delayed Paper integration.

The acquisition/finalization algorithms remain in their existing modules. This
module connects their bounded request plan to a request transport and exposes
the resulting finalized journal through the recoverable Paper service.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
import threading
import time
from typing import Any, Callable, Mapping
from pathlib import Path
from uuid import uuid4

from src.paper.cme_calendar import CMETradingCalendar
from src.paper.ibkr_delayed_acquisition import (
    AppendOnlyGapClassificationLedger,
    AtomicAcquisitionCursor,
    DelayedAcquisitionSession,
    rehydrate_pending_from_observations,
)
from src.paper.ibkr_delayed_market_data import (
    AppendOnlyFinalizationLedger,
    DelayedBarFinalizer,
    FinalizationPolicy,
    IBKRContractSchedule,
    IBKRContractWindow,
)
from src.paper.ibkr_delayed_acquisition import (
    HistoricalRequest,
    HistoricalRequestPacer,
    HistoricalRequestPlanner,
    execute_with_bounded_reconnect,
    format_ibkr_end_time,
)
from src.paper.ibkr_observation_ledger import (
    AppendOnlyBarObservationLedger,
    load_observation_ledger,
)


# IBKR occasionally includes a small leading overlap before a historical
# request's nominal start. Keep the allowance narrow and observable; larger
# excursions remain a hard error. This is not permission to ignore rows.
MAX_HISTORICAL_LEADING_OVERLAP_SECONDS = 5 * 60
HISTORICAL_REQUEST_ID_FLOOR = 1_000_000


class IBKRTransportUnavailable(RuntimeError):
    """Transient TWS socket/handshake failure, distinct from data validation."""


class IBKRReconnectBackoff:
    """Indefinite, bounded exponential retry schedule with small positive jitter.

    A retry is scheduled rather than busy-looped. Historical requests remain
    separately governed by ``HistoricalRequestPacer``.
    """

    def __init__(self, *, base_seconds: float = 5.0, maximum_seconds: float = 60.0,
                 jitter_fraction: float = 0.20, random_value: Callable[[], float] | None = None) -> None:
        if base_seconds <= 0 or maximum_seconds < base_seconds or not 0 <= jitter_fraction <= 1:
            raise ValueError("invalid IBKR reconnect backoff configuration")
        self.base_seconds = float(base_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.jitter_fraction = float(jitter_fraction)
        self.random_value = random_value or random.random
        self.failures = 0

    def reset(self) -> None:
        self.failures = 0

    def next_delay(self) -> float:
        base = min(self.maximum_seconds, self.base_seconds * (2 ** self.failures))
        self.failures += 1
        jitter = max(0.0, min(1.0, float(self.random_value()))) * self.jitter_fraction
        return min(self.maximum_seconds, base * (1.0 + jitter))


_TRANSIENT_IBKR_CODES = {502, 504, 507, 1100, 1101, 1102, 1300}


def _is_transient_ibkr_disconnect(result: Mapping[str, Any]) -> bool:
    """Recognize transport loss even when EClient.isConnected() is stale."""
    if not bool(result.get("connected")):
        return True
    for item in result.get("errors", ()) or ():
        if isinstance(item, Mapping):
            code = item.get("code", item.get("errorCode"))
            message = str(item.get("message", item.get("errorString", ""))).lower()
            if code is not None and int(code) in _TRANSIENT_IBKR_CODES:
                return True
            if any(token in message for token in (
                "handshake timed out", "connection between ibkr and trader workstation has been lost",
                "socket connection", "connection reset", "connection refused", "broken pipe",
            )):
                return True
        else:
            text = str(item).lower()
            if any(token in text for token in ("1100", "1101", "1102", "502", "504", "socket", "handshake")):
                return True
    return False


def _is_ibkr_pacing_error(result: Mapping[str, Any]) -> bool:
    for item in result.get("errors", ()) or ():
        if isinstance(item, Mapping):
            code = item.get("code", item.get("errorCode"))
            message = str(item.get("message", item.get("errorString", ""))).lower()
            try:
                code = int(code)
            except (TypeError, ValueError):
                code = -1
            if code == 162 and any(token in message for token in ("pacing", "too many requests", "frequency")):
                return True
    return False


def _is_ibkr_no_data_error(result: Mapping[str, Any]) -> bool:
    for item in result.get("errors", ()) or ():
        if not isinstance(item, Mapping):
            continue
        try:
            code = int(item.get("code", item.get("errorCode", -1)))
        except (TypeError, ValueError):
            continue
        message = str(item.get("message", item.get("errorString", ""))).lower()
        if code == 162 and ("no data" in message or "returned no data" in message):
            return True
    return False


class IBKRCursorAcquisitionController:
    """Execute existing bounded acquisition cycles and persist each response.

    ``request_historical`` is a read-only transport callback returning
    ``completed``, ``connected``, ``rows`` and ``errors``. Transport failures
    pause the cycle and schedule indefinite, bounded-backoff reconnection;
    fail-closed data/identity errors still propagate. Acquisition and Paper
    acknowledgment cursors remain independent.
    """

    def __init__(
        self,
        *,
        session: DelayedAcquisitionSession,
        request_historical: Callable[[HistoricalRequest, int], Mapping[str, Any]],
        reconnect: Callable[[], None] | None = None,
        planner: HistoricalRequestPlanner | None = None,
        pacer: HistoricalRequestPacer | None = None,
        clock: Callable[[], datetime] | None = None,
        on_event: Callable[..., Any] | None = None,
        initially_connected: bool = True,
        backoff: IBKRReconnectBackoff | None = None,
    ) -> None:
        self.session = session
        self.request_historical = request_historical
        self.reconnect_transport = reconnect
        self.planner = planner or HistoricalRequestPlanner()
        self.pacer = pacer or HistoricalRequestPacer(
            prior_request_epochs=session.cursor.recent_request_epochs
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_event = on_event
        self.backoff = backoff or IBKRReconnectBackoff()
        self.poll_number = max(
            (int(row["poll_number"]) for row in load_observation_ledger(
                session.observation_ledger.path
            )),
            default=0,
        )
        self.next_request_id = max(
            HISTORICAL_REQUEST_ID_FLOOR,
            int(session.cursor.next_request_id),
        )
        self.session.cursor.next_request_id = self.next_request_id
        self.session.cursor_store.save()
        self.cycles = 0
        self.requests = 0
        self.finalized = 0
        self.last_cycle: dict[str, Any] = {}
        saved_cursor = session.cursor
        self.last_transport_error: str | None = getattr(saved_cursor, "reconnect_last_error", None)
        self.backoff.failures = int(getattr(saved_cursor, "reconnect_backoff_failures", 0) or 0)
        saved_state = getattr(saved_cursor, "reconnect_state", None)
        self.connection_state = (
            "CONNECTED" if initially_connected else
            (saved_state if saved_state in {"DISCONNECTED", "RECONNECTING", "RECOVERING"}
             else "DISCONNECTED")
        )
        self.transport_available: bool | None = bool(initially_connected)
        self.reconnect_attempt_count = int(getattr(saved_cursor, "reconnect_attempt_count", 0) or 0)
        saved_retry_at = getattr(saved_cursor, "reconnect_next_retry_at_utc", None)
        self.next_retry_at_utc = (datetime.fromisoformat(saved_retry_at)
                                  if saved_retry_at else None)
        self.last_successful_handshake_utc: datetime | None = (
            datetime.fromisoformat(saved_cursor.reconnect_last_handshake_at_utc)
            if getattr(saved_cursor, "reconnect_last_handshake_at_utc", None)
            else (self.clock().astimezone(timezone.utc) if initially_connected else None)
        )
        saved_request_at = getattr(saved_cursor, "reconnect_last_request_at_utc", None)
        self.last_successful_request_utc = datetime.fromisoformat(saved_request_at) if saved_request_at else None
        self._incident_active = bool(getattr(saved_cursor, "reconnect_incident_id", None))
        self.incident_id: str | None = getattr(saved_cursor, "reconnect_incident_id", None)
        self.incident_started_at_utc: str | None = getattr(saved_cursor, "reconnect_started_at_utc", None)
        self._retry_reason: str | None = None
        self.acquisition_state = "IDLE"
        self._lock = threading.Lock()
        if not initially_connected and self.connection_state == "CONNECTED":
            self.connection_state = "DISCONNECTED"
        self._persist_reconnect_state()

    def _persist_reconnect_state(self) -> None:
        cursor = self.session.cursor
        cursor.reconnect_state = self.connection_state
        cursor.reconnect_incident_id = self.incident_id
        cursor.reconnect_started_at_utc = self.incident_started_at_utc
        cursor.reconnect_attempt_count = self.reconnect_attempt_count
        cursor.reconnect_backoff_failures = self.backoff.failures
        cursor.reconnect_next_retry_at_utc = (
            self.next_retry_at_utc.astimezone(timezone.utc).isoformat()
            if self.next_retry_at_utc else None
        )
        cursor.reconnect_last_handshake_at_utc = (
            self.last_successful_handshake_utc.astimezone(timezone.utc).isoformat()
            if self.last_successful_handshake_utc else None
        )
        cursor.reconnect_last_request_at_utc = (
            self.last_successful_request_utc.astimezone(timezone.utc).isoformat()
            if self.last_successful_request_utc else None
        )
        cursor.reconnect_last_error = self.last_transport_error
        self.session.cursor_store.save()

    def _emit_transition(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            from src.paper.logger import PaperEventType
            self.on_event(PaperEventType(event_type), dict(payload), timestamp=self.clock())
        except Exception:
            # Monitoring persistence must not alter acquisition safety or
            # stop the simulated engine. Status remains available in memory.
            return

    def _schedule_retry(self, now: datetime, error: str, *, disconnected: bool = True) -> None:
        if disconnected:
            if not self._incident_active:
                self._incident_active = True
                self.incident_id = uuid4().hex
                self.incident_started_at_utc = now.astimezone(timezone.utc).isoformat()
                self.connection_state = "RECONNECTING"
                self.transport_available = False
                self.last_transport_error = error
                self._persist_reconnect_state()
                self._emit_transition("feed_disconnected", {
                    "incident_id": self.incident_id,
                    "incident_started_at_utc": self.incident_started_at_utc,
                    "provider": "IBKR_TWS", "state": "DISCONNECTED",
                    "error": error, "retry_policy": "exponential_with_jitter",
                })
            self.transport_available = False
            self.connection_state = "RECONNECTING"
        delay = self.backoff.next_delay()
        self.next_retry_at_utc = now.astimezone(timezone.utc) + timedelta(seconds=delay)
        self._retry_reason = error
        self.last_transport_error = error
        self._persist_reconnect_state()

    def reconnection_status(self, *, include_progress: bool = True) -> dict[str, Any]:
        return {
            "connection_state": self.connection_state,
            "retry_count": self.reconnect_attempt_count,
            "next_retry_at_utc": (self.next_retry_at_utc.isoformat()
                                   if self.next_retry_at_utc else None),
            "last_successful_handshake_utc": (
                self.last_successful_handshake_utc.isoformat()
                if self.last_successful_handshake_utc else None
            ),
            "last_successful_request_utc": (
                self.last_successful_request_utc.isoformat()
                if self.last_successful_request_utc else None
            ),
            "incident_id": self.incident_id,
            "incident_started_at_utc": self.incident_started_at_utc,
            "recovery_progress": self._cursor_recovery_status() if include_progress else None,
            "last_transport_error": self.last_transport_error,
            "acquisition_state": self.acquisition_state,
        }

    def _reconnect_if_due(self, now: datetime) -> bool:
        if self.connection_state not in {"DISCONNECTED", "RECONNECTING"}:
            return True
        if self.next_retry_at_utc is not None and now.astimezone(timezone.utc) < self.next_retry_at_utc:
            return False
        if self.reconnect_transport is None:
            self._schedule_retry(now, "TWS disconnected and no reconnect handler is configured")
            return False
        self.reconnect_attempt_count += 1
        self._persist_reconnect_state()
        try:
            self.reconnect_transport()
        except (IBKRTransportUnavailable, OSError, TimeoutError, ConnectionError) as exc:
            self._schedule_retry(now, f"{type(exc).__name__}: {exc}")
            return False
        # Contract resolution and read-only delayed mode are verified by the
        # transport's reconnect callback. Do not call the incident recovered
        # until historical catch-up and Paper delivery are also reconciled.
        self.transport_available = True
        self.connection_state = "RECOVERING"
        self.last_successful_handshake_utc = now.astimezone(timezone.utc)
        self.next_retry_at_utc = None
        self._persist_reconnect_state()
        return True

    def mark_recovery_verified(self) -> None:
        """Close one outage only after provider and Paper cursors are caught up."""
        if not self._incident_active:
            return
        if (not self.session.cursor.bootstrap_backfill_complete
                or self.session.cursor.pending_confirmation_group):
            return
        now = self.clock().astimezone(timezone.utc)
        incident_id = self.incident_id
        self._incident_active = False
        self.incident_id = None
        self.incident_started_at_utc = None
        self.connection_state = "CONNECTED"
        self.transport_available = True
        self.last_transport_error = None
        self._retry_reason = None
        self.next_retry_at_utc = None
        self.reconnect_attempt_count = 0
        self.backoff.reset()
        self._emit_transition("feed_connected", {
            "incident_id": incident_id,
            "provider": "IBKR_TWS", "state": "CONNECTED",
            "recovered_at_utc": now.isoformat(),
            "last_successful_handshake_utc": (
                self.last_successful_handshake_utc.isoformat()
                if self.last_successful_handshake_utc else None
            ),
            "last_successful_request_utc": (
                self.last_successful_request_utc.isoformat()
                if self.last_successful_request_utc else None
            ),
        })
        self._persist_reconnect_state()

    def refresh(self, _paper_cursor: datetime | None = None) -> Mapping[str, Any]:
        """Run one paced discovery/confirmation cycle; suitable as source.refresh."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("IBKR acquisition refresh is already in progress")
        try:
            now = self.clock()
            if now.tzinfo is None:
                raise ValueError("acquisition controller clock must be timezone-aware")
            self.poll_number += 1
            if self.next_retry_at_utc is not None and now.astimezone(timezone.utc) < self.next_retry_at_utc:
                if self.connection_state not in {"DISCONNECTED", "RECONNECTING"}:
                    self.last_cycle = {
                        "cycle": self.cycles + 1, "poll_number": self.poll_number,
                        "requests": [], "finalized": 0,
                        "transport_available": self.transport_available,
                        **self.reconnection_status(),
                    }
                    return dict(self.last_cycle)
            if not self._reconnect_if_due(now):
                self.last_cycle = {
                    "cycle": self.cycles + 1, "poll_number": self.poll_number,
                    "requests": [], "finalized": 0,
                    "transport_available": False,
                    "last_transport_error": self.last_transport_error,
                    **self.reconnection_status(),
                }
                return dict(self.last_cycle)
            if self.next_retry_at_utc is not None and now.astimezone(timezone.utc) >= self.next_retry_at_utc:
                self.next_retry_at_utc = None
                self.acquisition_state = "POLLING"
            if self.connection_state == "RECOVERING":
                pace_wait = self.pacer.seconds_until_next_cycle(now)
                if pace_wait > 0:
                    self.next_retry_at_utc = now.astimezone(timezone.utc) + timedelta(seconds=pace_wait)
                    self.last_cycle = {
                        "cycle": self.cycles + 1, "poll_number": self.poll_number,
                        "requests": [], "finalized": 0,
                        "transport_available": True,
                        "last_transport_error": self.last_transport_error,
                        **self.reconnection_status(),
                    }
                    return dict(self.last_cycle)
            # On first service startup, seed a durable chronological acquisition
            # cursor from the Paper/bootstrap cursor. This is deliberately a
            # separate cursor: acquisition may be ahead while bars are awaiting
            # stabilization, but Paper delivery remains strictly contiguous.
            if not self.session.cursor.bootstrap_backfill_complete:
                if self.session.cursor.bootstrap_backfill_next_start_epoch_utc is None:
                    if _paper_cursor is not None:
                        if _paper_cursor.tzinfo is None:
                            raise ValueError("Paper cursor must be timezone-aware")
                        self.session.cursor.bootstrap_backfill_next_start_epoch_utc = (
                            int(_paper_cursor.timestamp()) + 60
                        )
                        self.session.cursor_store.save()
                    else:
                        # No explicit activation/commit cursor means this is a
                        # fresh diagnostic acquisition, not permission to scan
                        # arbitrary history.
                        self.session.cursor.bootstrap_backfill_complete = True
                        self.session.cursor_store.save()
            self._advance_verified_calendar_closure()
            pending = self.session.pending_confirmation_starts()
            safe_backfill_end = now - timedelta(seconds=660)
            catchup_active = (
                not self.session.cursor.bootstrap_backfill_complete
                and self.session.cursor.bootstrap_backfill_next_start_epoch_utc is not None
            )
            requests = self.planner.plan_cycle(
                now_utc=now,
                pending_bar_starts=pending,
                include_discovery=True,
                # During backlog recovery, oldest-first confirmation is
                # required to advance the contiguous Paper prefix. Round-robin
                # fairness resumes after catch-up completes.
                confirmation_offset=(0 if catchup_active else self.session.cursor.pending_confirmation_offset),
                bootstrap_backfill_next_start_epoch_utc=(
                    None if self.session.cursor.bootstrap_backfill_complete
                    else self.session.cursor.bootstrap_backfill_next_start_epoch_utc
                ),
                bootstrap_backfill_target_end_epoch_utc=(
                    None if self.session.cursor.bootstrap_backfill_complete
                    else self.session.cursor.bootstrap_backfill_target_end_epoch_utc
                ),
                bootstrap_safe_end_utc=(
                    None if self.session.cursor.bootstrap_backfill_complete
                    else safe_backfill_end
                ),
            )
            if (not self.session.cursor.bootstrap_backfill_complete
                    and self.session.cursor.bootstrap_backfill_target_end_epoch_utc is not None
                    and self.session.cursor.bootstrap_backfill_next_start_epoch_utc is not None
                    and self.session.cursor.bootstrap_backfill_next_start_epoch_utc
                    >= self.session.cursor.bootstrap_backfill_target_end_epoch_utc):
                # The acquisition cursor has reached the current safe delayed
                # frontier. Future cycles use rolling recent discovery.
                self.session.cursor.bootstrap_backfill_complete = True
                self.session.cursor_store.save()
            confirmation = next(
                (item for item in requests if item.kind == "pending_confirmation"), None
            )
            if confirmation is not None and confirmation.confirmation_group_count:
                self.session.cursor.pending_confirmation_offset = (
                    0 if catchup_active else (
                        int(confirmation.confirmation_group_index) + 1
                    ) % int(confirmation.confirmation_group_count)
                )
            try:
                self.pacer.begin_cycle(now, len(requests))
            except RuntimeError as exc:
                if "pacing" not in str(exc).lower() and "too frequent" not in str(exc).lower():
                    raise
                self.acquisition_state = "PAUSED_PACING"
                self._schedule_retry(now, str(exc), disconnected=False)
                self.next_retry_at_utc = max(
                    self.next_retry_at_utc,
                    now.astimezone(timezone.utc) + timedelta(seconds=60),
                )
                self.last_cycle = {
                    "cycle": self.cycles + 1, "poll_number": self.poll_number,
                    "requests": [], "finalized": 0,
                    "transport_available": self.transport_available,
                    **self.reconnection_status(),
                }
                return dict(self.last_cycle)
            self.session.persist_request_budget(self.pacer, now=now)
            finalized_count = 0
            detail: list[dict[str, Any]] = []

            for request in requests:
                request_id = self.next_request_id
                # Persist the ID allocation before sending the request. If the
                # process crashes while TWS is still delivering callbacks, a
                # resumed client cannot attribute a late response to a reused
                # request ID.
                self.next_request_id += 1
                self.session.cursor.next_request_id = self.next_request_id
                self.session.cursor_store.save()
                retry_id = request_id
                self.requests += 1
                try:
                    result = self.request_historical(request, retry_id)
                except (IBKRTransportUnavailable, OSError, TimeoutError, ConnectionError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if not self._incident_active and _paper_cursor is not None:
                        self.begin_catchup(_paper_cursor)
                    self._schedule_retry(now, error)
                    self.last_cycle = {
                        "cycle": self.cycles + 1, "poll_number": self.poll_number,
                        "requests": [{"request_id": request_id, "kind": request.kind,
                                      "completed": False, "transport_available": False,
                                      "error": error}], "finalized": finalized_count,
                        "transport_available": False, "last_transport_error": error,
                        "paper_cursor": _paper_cursor.isoformat() if _paper_cursor else None,
                        **self.reconnection_status(),
                    }
                    return dict(self.last_cycle)
                if not result.get("completed"):
                    if _is_transient_ibkr_disconnect(result):
                        errors = list(result.get("errors", []))
                        error = (
                            f"TWS disconnected during {request.kind}; "
                            f"errors={errors or ['no provider callback received']}"
                        )
                        if not self._incident_active and _paper_cursor is not None:
                            self.begin_catchup(_paper_cursor)
                        self._schedule_retry(now, error)
                        self.last_cycle = {
                            "cycle": self.cycles + 1, "poll_number": self.poll_number,
                            "requests": [{
                                "request_id": request_id,
                                "kind": request.kind,
                                "completed": False,
                                "transport_available": False,
                                "error": error,
                            }],
                            "finalized": finalized_count,
                            "transport_available": False,
                            "last_transport_error": error,
                            "paper_cursor": _paper_cursor.isoformat() if _paper_cursor else None,
                            **self.reconnection_status(),
                        }
                        return dict(self.last_cycle)
                    if _is_ibkr_pacing_error(result):
                        error = f"IBKR historical pacing response during {request.kind}: {result.get('errors', [])}"
                        self.acquisition_state = "PAUSED_PACING"
                        self._schedule_retry(now, error, disconnected=False)
                        self.next_retry_at_utc = max(
                            self.next_retry_at_utc,
                            now.astimezone(timezone.utc) + timedelta(seconds=60),
                        )
                        self.last_cycle = {
                            "cycle": self.cycles + 1, "poll_number": self.poll_number,
                            "requests": [{"request_id": request_id, "kind": request.kind,
                                          "completed": False, "transport_available": self.transport_available,
                                          "error": error}], "finalized": finalized_count,
                            "transport_available": self.transport_available,
                            **self.reconnection_status(),
                        }
                        return dict(self.last_cycle)
                    if _is_ibkr_no_data_error(result):
                        # An empty HMDS response is a verified closure only
                        # when the entire requested minute range is closed in
                        # the reviewed calendar. During an expected open
                        # interval, wait and retry without moving cursors or
                        # pretending the bars existed.
                        verified_closed = False
                        if request.kind == "recent_discovery":
                            calendar = self.session.finalizer.calendar
                            end_epoch = int(now.astimezone(timezone.utc).timestamp())
                            end_epoch -= end_epoch % 60
                            start_epoch = end_epoch - int(request.duration_seconds)
                            expected = [calendar.expected_globex_minute(
                                datetime.fromtimestamp(stamp, timezone.utc)
                            ) for stamp in range(start_epoch, end_epoch, 60)]
                            verified_closed = bool(expected) and not any(expected)
                        if verified_closed:
                            result = {"completed": True, "connected": True, "rows": [],
                                      "errors": [], "provider_empty_classification": "verified_market_closure"}
                        else:
                            error = f"IBKR returned no historical bars during a range not proven closed: {result.get('errors', [])}"
                            if _paper_cursor is not None:
                                self.begin_catchup(_paper_cursor)
                            self.acquisition_state = "WAITING_FOR_PROVIDER_DATA"
                            self._schedule_retry(now, error, disconnected=False)
                            self.next_retry_at_utc = max(
                                self.next_retry_at_utc,
                                now.astimezone(timezone.utc) + timedelta(seconds=60),
                            )
                            self.last_cycle = {
                                "cycle": self.cycles + 1, "poll_number": self.poll_number,
                                "requests": [{"request_id": request_id, "kind": request.kind,
                                              "completed": False, "transport_available": True,
                                              "error": error}], "finalized": finalized_count,
                                "transport_available": True,
                                **self.reconnection_status(),
                            }
                            return dict(self.last_cycle)
                    if not result.get("completed"):
                        errors = result.get("errors", [])
                        raise RuntimeError(
                            f"IBKR {request.kind} request did not complete; "
                            f"connected={result.get('connected')}; errors={errors}"
                        )
                rows = sorted(
                    list(result.get("rows", [])), key=lambda row: int(row["timestamp"])
                )
                deferred_finalization: set[int] = set()
                leading_overlaps: list[int] = []
                if request.end_time_utc is not None:
                    request_end = int(request.end_time_utc.timestamp())
                    request_start = request_end - int(request.duration_seconds)
                    too_early = [int(row["timestamp"]) for row in rows
                                 if int(row["timestamp"]) < request_start - MAX_HISTORICAL_LEADING_OVERLAP_SECONDS]
                    too_late = [int(row["timestamp"]) for row in rows
                                if int(row["timestamp"]) > request_end]
                    if too_early or too_late:
                        offending = min(too_early) if too_early else max(too_late)
                        first = datetime.fromtimestamp(offending, timezone.utc).isoformat()
                        raise RuntimeError(
                            f"IBKR {request.kind} response exceeded bounded request overlap; "
                            f"allowed=[{datetime.fromtimestamp(request_start - MAX_HISTORICAL_LEADING_OVERLAP_SECONDS, timezone.utc).isoformat()}, "
                            f"{request.end_time_utc.isoformat()}]; offending={first}"
                        )
                    leading_overlaps = [int(row["timestamp"]) for row in rows
                                        if int(row["timestamp"]) < request_start]
                    # The request end is exclusive. Preserve an exact-boundary
                    # row in the append-only observation ledger, but require a
                    # later overlapping response before finalizing it.
                    deferred_finalization = {int(row["timestamp"]) for row in rows
                                             if int(row["timestamp"]) == request_end}
                observed_values = [
                    datetime.fromisoformat(str(row["arrival_utc"]))
                    for row in rows if row.get("arrival_utc")
                ]
                observed_at = max(observed_values, default=self.clock())
                finalized = self.session.ingest_response(
                    rows,
                    observed_at=observed_at,
                    poll_number=self.poll_number,
                    request_id=int(request_id),
                    finalize_timestamps=(
                        request.covered_pending_starts
                        if request.kind == "pending_confirmation" else None
                    ),
                    defer_finalization_timestamps=deferred_finalization,
                )
                if request.kind == "bootstrap_backfill":
                    # Move the cursor to the observed response boundary, not
                    # the requested boundary. IBKR can round historical query
                    # end times; advancing by the request window can skip the
                    # trailing minutes that the provider did not return.
                    request_start = int(request.end_time_utc.timestamp()) - int(request.duration_seconds)
                    request_end = int(request.end_time_utc.timestamp())
                    observed_starts = [int(row["timestamp"]) for row in rows]
                    if not observed_starts:
                        raise RuntimeError(
                            "IBKR bootstrap_backfill returned no bars; acquisition cursor was not advanced"
                        )
                    observed_end = min(request_end, max(observed_starts) + 60)
                    if observed_end <= request_start:
                        raise RuntimeError(
                            "IBKR bootstrap_backfill made no forward timestamp progress; cursor was not advanced"
                        )
                    self.session.cursor.bootstrap_backfill_next_start_epoch_utc = observed_end
                    if (self.session.cursor.bootstrap_backfill_target_end_epoch_utc is not None
                            and self.session.cursor.bootstrap_backfill_next_start_epoch_utc
                            >= self.session.cursor.bootstrap_backfill_target_end_epoch_utc):
                        self.session.cursor.bootstrap_backfill_complete = True
                    self.session.cursor_store.save()
                elif (request.kind == "recent_discovery"
                      and not self.session.cursor.bootstrap_backfill_complete
                      and self.session.cursor.bootstrap_backfill_target_end_epoch_utc is None
                      and rows):
                    cursor_start = self.session.cursor.bootstrap_backfill_next_start_epoch_utc
                    frontier_rows = [int(row["timestamp"]) for row in rows
                                     if cursor_start is None or int(row["timestamp"]) >= cursor_start]
                    if frontier_rows:
                        # Target is the exclusive end of the latest observed
                        # complete minute, not wall-clock now.
                        self.session.cursor.bootstrap_backfill_target_end_epoch_utc = max(frontier_rows) + 60
                        self.session.cursor_store.save()
                finalized_count += len(finalized)
                detail.append({
                    "request_id": int(request_id),
                    "kind": request.kind,
                    "rows": len(rows),
                    "leading_overlap_rows": len(leading_overlaps),
                    "exclusive_end_rows_deferred": len(deferred_finalization),
                    "finalized": len(finalized),
                    "reconnects": self.reconnect_attempt_count,
                    "errors": list(result.get("errors", [])),
                })
                self.next_request_id = max(self.next_request_id, int(request_id) + 1)
                self.session.cursor.next_request_id = self.next_request_id
                self.session.cursor_store.save()

            self.cycles += 1
            self.finalized += finalized_count
            self.last_successful_request_utc = self.clock().astimezone(timezone.utc)
            if not self._incident_active:
                self.connection_state = "CONNECTED"
                self.transport_available = True
                self.last_transport_error = None
            elif self.connection_state != "RECOVERING":
                self.connection_state = "RECOVERING"
            self._persist_reconnect_state()
            self.last_cycle = {
                "cycle": self.cycles,
                "poll_number": self.poll_number,
                "requests": detail,
                "finalized": finalized_count,
                "acquisition_cursor": self.session.cursor.last_delivered_bar_start_epoch_utc,
                "paper_cursor": _paper_cursor.isoformat() if _paper_cursor else None,
                "transport_available": True,
                **self.reconnection_status(),
            }
            return dict(self.last_cycle)
        finally:
            self._lock.release()

    def begin_catchup(self, paper_cursor: datetime | None) -> None:
        """Seed/restart acquisition catch-up from the first uncommitted Paper minute."""
        if paper_cursor is not None and paper_cursor.tzinfo is None:
            raise ValueError("Paper cursor must be timezone-aware")
        if paper_cursor is None:
            # On a fresh service this is normally source.bootstrap_after_timestamp;
            # subscribe passes that cursor through the source hook below.
            next_start = None
        else:
            next_start = int(paper_cursor.timestamp()) + 60
        cursor = self.session.cursor
        if cursor.last_delivered_bar_start_epoch_utc is not None:
            next_start = max(next_start or 0, cursor.last_delivered_bar_start_epoch_utc + 60)
        if cursor.bootstrap_backfill_complete:
            cursor.bootstrap_backfill_next_start_epoch_utc = next_start
            cursor.bootstrap_backfill_target_end_epoch_utc = None
            cursor.bootstrap_backfill_complete = next_start is None
        elif cursor.bootstrap_backfill_next_start_epoch_utc is None:
            cursor.bootstrap_backfill_next_start_epoch_utc = next_start
            cursor.bootstrap_backfill_target_end_epoch_utc = None
            cursor.bootstrap_backfill_complete = next_start is None
        elif next_start is not None and cursor.bootstrap_backfill_next_start_epoch_utc < next_start:
            # A durable Paper checkpoint is stronger evidence than the
            # acquisition cursor; never reacquire committed historical range.
            cursor.bootstrap_backfill_next_start_epoch_utc = next_start
            cursor.bootstrap_backfill_target_end_epoch_utc = None
        if paper_cursor is not None and self.session.finalizer.last_delivered is None:
            self.session.finalizer.last_delivered = paper_cursor.astimezone(timezone.utc)
        self.session.cursor_store.save()

    def _advance_verified_calendar_closure(self) -> dict[str, Any] | None:
        """Skip only a calendar-proven closure at the backfill cursor.

        IBKR may answer a historical query wholly inside the daily maintenance
        break with the preceding active bars. Those rows are not evidence that
        the requested interval was covered. Move to the next calendar-open
        minute instead; unexplained gaps inside an open session still fail the
        normal bounded-response checks.
        """
        cursor = self.session.cursor
        start = cursor.bootstrap_backfill_next_start_epoch_utc
        if cursor.bootstrap_backfill_complete or start is None:
            return None
        target = cursor.bootstrap_backfill_target_end_epoch_utc
        if target is not None and start >= target:
            cursor.bootstrap_backfill_complete = True
            self.session.cursor_store.save()
            return None

        calendar = self.session.finalizer.calendar
        start_dt = datetime.fromtimestamp(int(start), timezone.utc)
        if calendar.expected_globex_minute(start_dt):
            return None

        # The supported historical snapshots span multiple-day holiday/weekend
        # closures. Keep the scan bounded and let an out-of-coverage date raise
        # from the calendar rather than treating it as a closure.
        next_open: int | None = None
        for offset in range(1, 14 * 24 * 60 + 1):
            candidate = int(start) + offset * 60
            if target is not None and candidate >= target:
                next_open = int(target)
                break
            stamp = datetime.fromtimestamp(candidate, timezone.utc)
            if calendar.expected_globex_minute(stamp):
                next_open = candidate
                break
        if next_open is None:
            raise RuntimeError(
                "no next verified Globex open found within the bounded 14-day calendar scan"
            )

        skipped = {
            "classification": "verified_exchange_closure",
            "from_utc": start_dt.isoformat(),
            "to_utc": datetime.fromtimestamp(next_open, timezone.utc).isoformat(),
            "minutes": max(0, (next_open - int(start)) // 60),
            "calendar_version": calendar.snapshot.version,
            "calendar_identity": calendar.snapshot.identity,
            "recorded_at_utc": self.clock().astimezone(timezone.utc).isoformat(),
        }
        cursor.verified_closure_skips = (*cursor.verified_closure_skips, skipped)
        cursor.bootstrap_backfill_next_start_epoch_utc = next_open
        if target is not None and next_open >= target:
            cursor.bootstrap_backfill_complete = True
        self.session.cursor_store.save()
        return skipped

    def recovery_status(self) -> dict[str, Any]:
        progress = self._cursor_recovery_status()
        return {**progress, **self.reconnection_status(include_progress=False),
                "recovery_progress": progress}

    def _cursor_recovery_status(self) -> dict[str, Any]:
        cursor = self.session.cursor
        return {
            "catchup_active": not cursor.bootstrap_backfill_complete,
            "backfill_next_start_epoch_utc": cursor.bootstrap_backfill_next_start_epoch_utc,
            "backfill_target_end_epoch_utc": cursor.bootstrap_backfill_target_end_epoch_utc,
            "backfill_complete": cursor.bootstrap_backfill_complete,
            "latest_observed_bar_epoch_utc": cursor.last_discovered_bar_start_epoch_utc,
            "verified_closure_skips": list(cursor.verified_closure_skips),
            "pending_confirmation_count": len(cursor.pending_confirmation_group),
        }

    def is_recovered_bar(self, bar_start_epoch_utc: int) -> bool:
        cursor = self.session.cursor
        if not cursor.bootstrap_backfill_complete:
            return True
        target = cursor.bootstrap_backfill_target_end_epoch_utc
        return target is not None and int(bar_start_epoch_utc) < target


class TWSHistoricalTRADESClient:
    """Small read-only TWS historical request transport for one pinned future."""

    def __init__(self, *, host: str, port: int, client_id: int, con_id: int,
                 local_symbol: str, expiry: str, timeout_seconds: float = 20.0,
                 connect_on_init: bool = True) -> None:
        try:
            from ibapi.client import EClient
            from ibapi.contract import Contract
            from ibapi.wrapper import EWrapper
        except ImportError as exc:
            raise RuntimeError("IBKR API package is unavailable in this Python environment") from exc

        self.host, self.port, self.client_id = host, int(port), int(client_id)
        self.con_id, self.local_symbol, self.expiry = int(con_id), local_symbol, expiry
        self.timeout = float(timeout_seconds)
        self._thread: threading.Thread | None = None
        self._next_id = 100

        class App(EWrapper, EClient):
            def __init__(self, owner: "TWSHistoricalTRADESClient") -> None:
                EClient.__init__(self, self)
                self.owner = owner
                self.connected_event = threading.Event()
                self.socket_ack_event = threading.Event()
                self.handshake_failed_event = threading.Event()
                self.contract_event = threading.Event()
                self.request_event = threading.Event()
                self.contract_rows: list[Any] = []
                self.rows: list[dict[str, Any]] = []
                self.errors: list[dict[str, Any]] = []
                self.reader_error: str | None = None
                self.active_request: int | None = None

            def nextValidId(self, orderId: int) -> None:
                self.owner._next_id = max(self.owner._next_id, int(orderId))
                self.connected_event.set()

            def connectAck(self) -> None:
                self.socket_ack_event.set()

            # The EClient base class exposes order APIs. Delayed Paper is
            # historical-data-only, so make accidental order routing fail at
            # the transport boundary even if a future caller reaches `app`.
            def placeOrder(self, *args: Any, **kwargs: Any) -> None:
                raise PermissionError("IBKR delayed Paper transport is read-only; placeOrder is disabled")

            def cancelOrder(self, *args: Any, **kwargs: Any) -> None:
                raise PermissionError("IBKR delayed Paper transport is read-only; cancelOrder is disabled")

            def reqGlobalCancel(self, *args: Any, **kwargs: Any) -> None:
                raise PermissionError("IBKR delayed Paper transport is read-only; global cancel is disabled")

            def contractDetails(self, reqId: int, details: Any) -> None:
                if reqId == 1:
                    self.contract_rows.append(details.contract)

            def contractDetailsEnd(self, reqId: int) -> None:
                if reqId == 1:
                    self.contract_event.set()

            def historicalData(self, reqId: int, bar: Any) -> None:
                if reqId != self.active_request:
                    return
                try:
                    stamp = int(bar.date)
                    self.rows.append({
                        "con_id": self.owner.con_id,
                        "timestamp": stamp,
                        "open": float(bar.open), "high": float(bar.high),
                        "low": float(bar.low), "close": float(bar.close),
                        "volume": float(bar.volume),
                        "arrival_utc": datetime.now(timezone.utc).isoformat(),
                    })
                except (TypeError, ValueError, OverflowError) as exc:
                    self.errors.append({"code": None, "message": f"invalid IBKR bar: {exc}"})

            def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
                if reqId == self.active_request:
                    self.request_event.set()

            def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:
                record = {"request_id": int(reqId), "code": int(errorCode),
                          "message": str(errorString)}
                self.errors.append(record)
                if not self.connected_event.is_set() and int(errorCode) in {502, 504, 1100, 1300}:
                    self.handshake_failed_event.set()
                code = int(errorCode)
                if code in {502, 504, 507, 1100, 1101, 1102, 1300} or (
                    reqId == self.active_request and code in {162, 200, 321, 366}
                ):
                    self.request_event.set()

            def connectionClosed(self) -> None:
                handshake_was_complete = self.connected_event.is_set()
                self.connected_event.clear()
                if not handshake_was_complete:
                    self.handshake_failed_event.set()
                self.request_event.set()

        self._app_type = App
        self.app: Any = None
        self.contract: Any = None
        if connect_on_init:
            self._connect_and_resolve()

    @property
    def is_connected(self) -> bool:
        if self.app is None:
            return False
        try:
            return bool(self.app.isConnected()) and self.contract is not None
        except Exception:
            return False

    def _connect_and_resolve(self) -> None:
        self.app = self._app_type(self)
        try:
            self.app.connect(self.host, self.port, self.client_id)
            self._thread = threading.Thread(
                target=self._run_reader, args=(self.app,),
                name="ibkr-paper-historical", daemon=True,
            )
            self._thread.start()
            deadline = time.monotonic() + self.timeout
            while not self.app.connected_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.app.handshake_failed_event.is_set():
                    details = self._handshake_diagnostic()
                    raise IBKRTransportUnavailable(
                        f"TWS API handshake failed at {self.host}:{self.port} "
                        f"(client_id={self.client_id}, timeout={self.timeout:g}s): {details}"
                    )
                if not self._thread.is_alive():
                    raise IBKRTransportUnavailable(
                        f"TWS API reader stopped before handshake at {self.host}:{self.port} "
                        f"(client_id={self.client_id}): {self._handshake_diagnostic()}"
                    )
                self.app.connected_event.wait(min(0.1, remaining))
        except BaseException:
            self._close_failed_connection()
            raise
        try:
            query = self._new_contract()
            self.app.reqContractDetails(1, query)
            if not self.app.contract_event.wait(self.timeout):
                raise IBKRTransportUnavailable(
                    f"TWS contract-details response timed out at {self.host}:{self.port} "
                    f"(client_id={self.client_id}); callbacks={self.app.errors[-5:]}"
                )
            matches = [item for item in self.app.contract_rows if int(item.conId) == self.con_id]
            if len(matches) != 1:
                raise RuntimeError(f"expected one TWS contract for conId {self.con_id}; got {len(matches)}")
            contract = matches[0]
            if (contract.symbol != "MNQ" or contract.secType != "FUT"
                    or contract.localSymbol != self.local_symbol
                    or str(contract.lastTradeDateOrContractMonth) != self.expiry):
                raise RuntimeError("TWS resolved contract differs from approved MNQ mapping")
            self.contract = contract
            self.app.reqMarketDataType(3)
        except BaseException:
            self._close_failed_connection()
            raise

    def _run_reader(self, app: Any) -> None:
        try:
            app.run()
        except BaseException as exc:
            app.reader_error = f"{type(exc).__name__}: {exc}"
            app.handshake_failed_event.set()
            app.request_event.set()

    def _handshake_diagnostic(self) -> str:
        app = self.app
        if app is None:
            return "no TWS client instance"
        errors = list(getattr(app, "errors", []))[-5:]
        reader_error = getattr(app, "reader_error", None)
        connected = False
        try:
            connected = bool(app.isConnected())
        except Exception:
            pass
        return f"socket_connected={connected}; callbacks={errors}; reader_error={reader_error}"

    def _close_failed_connection(self) -> None:
        app = self.app
        if app is not None:
            try:
                app.disconnect()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def _new_contract(self) -> Any:
        from ibapi.contract import Contract
        contract = Contract()
        contract.conId = self.con_id
        contract.exchange = "CME"
        return contract

    def request(self, request: HistoricalRequest, request_id: int) -> Mapping[str, Any]:
        if not self.is_connected:
            return {"completed": False, "connected": False, "rows": [],
                    "errors": [{"code": 0, "message": "TWS API is not connected"}]}
        self.app.rows = []
        self.app.errors = []
        self.app.request_event.clear()
        self.app.active_request = int(request_id)
        self._next_id = max(self._next_id, int(request_id) + 1)
        end_time = (
            "" if request.end_time_utc is None
            else format_ibkr_end_time(request.end_time_utc)
        )
        self.app.reqHistoricalData(
            int(request_id), self.contract, end_time,
            f"{request.duration_seconds} S", "1 min", "TRADES", 0, 2, False, [],
        )
        completed = self.app.request_event.wait(self.timeout)
        if not completed:
            try:
                self.app.cancelHistoricalData(int(request_id))
            except Exception:
                pass
        errors = list(self.app.errors)
        return {
            "completed": bool(completed and not errors),
            "connected": bool(self.app.isConnected()),
            "rows": list(self.app.rows), "errors": errors,
        }

    def reconnect(self) -> None:
        old = self.app
        thread = self._thread
        if old is not None and old.isConnected():
            old.disconnect()
        if thread is not None:
            thread.join(timeout=2)
        try:
            self._connect_and_resolve()
        except IBKRTransportUnavailable:
            raise
        except (OSError, TimeoutError, ConnectionError) as exc:
            self._close_failed_connection()
            raise IBKRTransportUnavailable(
                f"TWS reconnect failed at {self.host}:{self.port} "
                f"(client_id={self.client_id}): {type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        if self.app is not None:
            try:
                if self.app.isConnected():
                    self.app.disconnect()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)


def build_cursor_aware_pipeline(
    *,
    output_dir: str | Path,
    calendar: CMETradingCalendar,
    request_historical: Callable[[HistoricalRequest, int], Mapping[str, Any]],
    reconnect: Callable[[], None] | None = None,
    on_event: Callable[..., Any] | None = None,
    initially_connected: bool = True,
    contract_id: int = 815824267,
    local_symbol: str = "MNQZ6",
    expiry: str = "20261218",
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Compose existing acquisition journals/controller with Paper delivery.

    The pinned contract window is deliberately bounded to the reviewed CME
    snapshot, with one preceding local day for the Globex evening open. The CME
    calendar remains the authority and rejects any timestamp outside coverage.
    """
    from datetime import time as wall_time, timedelta

    from src.paper.ibkr_paper_recovery import (
        AppendOnlyPaperDeliveryLedger,
        IBKRFinalizedLedgerSource,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = calendar.snapshot
    coverage_start_local = datetime.combine(
        snapshot.coverage_start - timedelta(days=1), wall_time(0), timezone.utc
    )
    coverage_end_local = datetime.combine(
        snapshot.coverage_end + timedelta(days=2), wall_time(0), timezone.utc
    )
    contract_schedule = IBKRContractSchedule([
        IBKRContractWindow(
            contract_id, local_symbol, coverage_start_local, coverage_end_local
        )
    ])
    observations = AppendOnlyBarObservationLedger(
        root / "ibkr_observations.jsonl", run_id="delayed-paper"
    )
    finalized = AppendOnlyFinalizationLedger(root / "ibkr_finalized.jsonl")
    cursor_store = AtomicAcquisitionCursor(
        root / "ibkr_acquisition_cursor.json", contract_id=contract_id,
        local_symbol=local_symbol, expiry=expiry,
    )
    gap_ledger = AppendOnlyGapClassificationLedger(root / "ibkr_gap_classifications.jsonl")
    finalizer = DelayedBarFinalizer(
        contract_schedule=contract_schedule, calendar=calendar,
        policy=FinalizationPolicy(), finalization_ledger=finalized,
        gap_classifications=gap_ledger.rows,
    )
    rehydrate_pending_from_observations(
        finalizer, load_observation_ledger(observations.path)
    )
    acquisition_session = DelayedAcquisitionSession(
        cursor_store=cursor_store, observation_ledger=observations,
        finalizer=finalizer,
    )
    acquisition_session.pending_confirmation_starts()
    delivery = AppendOnlyPaperDeliveryLedger(
        root / "paper_delivery.jsonl", contract_id=contract_id,
        local_symbol=local_symbol, expiry=expiry,
        calendar_identity=snapshot.identity,
    )
    controller = IBKRCursorAcquisitionController(
        session=acquisition_session, request_historical=request_historical,
        reconnect=reconnect, clock=clock, on_event=on_event,
        initially_connected=initially_connected,
    )
    source = IBKRFinalizedLedgerSource(
        finalization_ledger=finalized, delivery_ledger=delivery,
        contract_schedule=contract_schedule, calendar=calendar,
        refresh=controller.refresh, begin_catchup=controller.begin_catchup,
        recovery_status=controller.recovery_status,
        is_recovered_bar=controller.is_recovered_bar,
        mark_recovery_verified=controller.mark_recovery_verified,
    )
    return {
        "source": source,
        "controller": controller,
        "delivery_ledger": delivery,
        "acquisition_session": acquisition_session,
        "observation_ledger": observations,
        "finalization_ledger": finalized,
        "cursor_store": cursor_store,
        "contract_schedule": contract_schedule,
    }
