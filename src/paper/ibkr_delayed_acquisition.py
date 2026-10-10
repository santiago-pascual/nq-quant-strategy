"""Persistent cursor and bounded request planning for delayed IBKR bars.

This module is transport-neutral. It does not connect to TWS or submit orders;
the owner of an IBKR client executes the returned bounded historical requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable

from src.paper.ibkr_delayed_market_data import DelayedBarFinalizer, FinalizedIBKRBar
from src.paper.ibkr_observation_ledger import AppendOnlyBarObservationLedger


UTC = timezone.utc


@dataclass(frozen=True)
class HistoricalRequest:
    kind: str
    duration_seconds: int
    end_time_utc: datetime | None
    covered_pending_starts: tuple[int, ...] = ()
    confirmation_group_index: int | None = None
    confirmation_group_count: int | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds < 60 or self.duration_seconds > 1800:
            raise ValueError("historical request duration must be within 60..1800 seconds")
        if self.kind not in {"recent_discovery", "pending_confirmation", "bootstrap_backfill"}:
            raise ValueError("unsupported historical request kind")
        if self.kind in {"pending_confirmation", "bootstrap_backfill"} and self.end_time_utc is None:
            raise ValueError(f"{self.kind} requires a UTC end time")
        if self.end_time_utc is not None:
            if self.end_time_utc.tzinfo is None:
                raise ValueError("request end time must be timezone-aware")
            object.__setattr__(self, "end_time_utc", self.end_time_utc.astimezone(UTC))


@dataclass
class AcquisitionCursor:
    contract_id: int
    local_symbol: str
    expiry: str
    last_request_at_utc: str | None = None
    last_observation_sequence: int = 0
    last_discovered_bar_start_epoch_utc: int | None = None
    oldest_pending_bar_start_epoch_utc: int | None = None
    last_delivered_bar_start_epoch_utc: int | None = None
    bootstrap_backfill_next_start_epoch_utc: int | None = None
    bootstrap_backfill_target_end_epoch_utc: int | None = None
    bootstrap_backfill_complete: bool = False
    pending_confirmation_group: tuple[int, ...] = ()
    pending_confirmation_offset: int = 0
    recent_request_epochs: tuple[float, ...] = ()
    # Request IDs are durable across process restarts. A high floor avoids
    # reusing IDs from pre-cursor versions of the runner after a disconnect.
    next_request_id: int = 1_000_000
    verified_closure_skips: tuple[dict, ...] = ()
    reconnect_state: str | None = None
    reconnect_incident_id: str | None = None
    reconnect_started_at_utc: str | None = None
    reconnect_attempt_count: int = 0
    reconnect_backoff_failures: int = 0
    reconnect_next_retry_at_utc: str | None = None
    reconnect_last_handshake_at_utc: str | None = None
    reconnect_last_request_at_utc: str | None = None
    reconnect_last_error: str | None = None

    def __post_init__(self) -> None:
        if self.contract_id <= 0 or not self.local_symbol or not self.expiry:
            raise ValueError("cursor requires pinned contract identity and expiry")
        self.pending_confirmation_group = tuple(int(v) for v in self.pending_confirmation_group)
        self.recent_request_epochs = tuple(float(v) for v in self.recent_request_epochs)
        self.next_request_id = int(self.next_request_id)
        if self.next_request_id < 1:
            raise ValueError("next historical request ID must be positive")
        self.verified_closure_skips = tuple(dict(item) for item in self.verified_closure_skips)


class AtomicAcquisitionCursor:
    """Crash-safe cursor file; observations and deliveries remain their own journals.

    The append-only observation and finalization ledgers are authoritative. This
    small atomic snapshot is an index/checkpoint and can be reconstructed from
    those journals after a crash.
    """

    VERSION = 1

    def __init__(self, path: str | Path, *, contract_id: int, local_symbol: str, expiry: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != self.VERSION:
                raise ValueError("unsupported acquisition cursor schema")
            cursor = AcquisitionCursor(**payload["cursor"])
            if (cursor.contract_id, cursor.local_symbol, cursor.expiry) != (contract_id, local_symbol, expiry):
                raise ValueError("persisted acquisition cursor belongs to a different MNQ contract")
            self.cursor = cursor
        else:
            self.cursor = AcquisitionCursor(contract_id, local_symbol, expiry)

    def save(self) -> None:
        payload = {"schema_version": self.VERSION, "cursor": asdict(self.cursor)}
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Windows antivirus/indexers can briefly hold an existing cursor
            # file open. Retry only that transient replace failure; persistent
            # permission or filesystem errors remain fatal and fail closed.
            replace_delays = (0.02, 0.05, 0.1, 0.2, 0.4)
            for attempt, delay in enumerate((*replace_delays, None)):
                try:
                    os.replace(temp_name, self.path)
                    break
                except PermissionError:
                    if os.name != "nt" or delay is None:
                        raise
                    time.sleep(delay)
            # Directory fsync is not available on all Windows filesystems.
            if os.name != "nt":
                dir_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


class HistoricalRequestPlanner:
    """Plan recent discovery plus bounded confirmation for oldest pending bars.

    At most one discovery and one confirmation are scheduled per cycle. A
    confirmation request is bounded to the exact pending block, keeping all
    pending bars in the server query without spanning unrelated closures at
    the left edge. IBKR can answer queries that straddle daily maintenance
    with the preceding session's final bars.
    """

    DISCOVERY_SECONDS = 1800
    CONFIRMATION_SECONDS = 1800
    MAX_GROUP_SPAN_SECONDS = 1700

    @staticmethod
    def _epoch(value: int | datetime) -> int:
        if isinstance(value, int):
            return value
        if value.tzinfo is None:
            raise ValueError("bar cursor timestamps must be timezone-aware")
        return int(value.astimezone(UTC).timestamp())

    def plan_cycle(self, *, now_utc: datetime, pending_bar_starts: Iterable[int | datetime],
                   include_discovery: bool = True, confirmation_offset: int = 0,
                   bootstrap_backfill_next_start_epoch_utc: int | None = None,
                   bootstrap_backfill_target_end_epoch_utc: int | None = None,
                   bootstrap_safe_end_utc: datetime | None = None) -> list[HistoricalRequest]:
        if now_utc.tzinfo is None:
            raise ValueError("cycle time must be timezone-aware")
        now = now_utc.astimezone(UTC)
        pending = sorted(set(self._epoch(item) for item in pending_bar_starts))
        requests: list[HistoricalRequest] = []
        backfill_active = bootstrap_backfill_next_start_epoch_utc is not None
        backfill_added = False
        if (backfill_active and bootstrap_backfill_target_end_epoch_utc is not None
                and bootstrap_safe_end_utc is not None):
            if bootstrap_safe_end_utc.tzinfo is None:
                raise ValueError("bootstrap backfill safe end must be timezone-aware")
            start = int(bootstrap_backfill_next_start_epoch_utc)
            safe_end = min(int(bootstrap_safe_end_utc.astimezone(UTC).timestamp()),
                           int(bootstrap_backfill_target_end_epoch_utc))
            safe_end -= safe_end % 60
            duration = min(self.DISCOVERY_SECONDS, safe_end - start)
            # Request only complete minute bars up to the delayed/finalization frontier.
            duration -= duration % 60
            if duration >= 60:
                end = datetime.fromtimestamp(start + duration, UTC)
                requests.append(HistoricalRequest("bootstrap_backfill", duration, end))
                backfill_added = True
        # First discover the provider's actual delayed frontier, then backfill
        # chronologically only through that observed frontier. Wall-clock time
        # is never treated as proof that historical bars exist.
        if include_discovery and not backfill_added and (not backfill_active or bootstrap_backfill_target_end_epoch_utc is None):
            requests.append(HistoricalRequest("recent_discovery", self.DISCOVERY_SECONDS, None))
        if pending:
            # Partition all pending times into bounded requests. A persisted
            # round-robin offset prevents one unstable old window from starving
            # confirmation of later pending windows. Every time remains in the
            # queue until finalized or explicitly classified.
            groups: list[list[int]] = []
            for stamp in pending:
                if not groups or stamp - groups[-1][0] > self.MAX_GROUP_SPAN_SECONDS:
                    groups.append([stamp])
                else:
                    groups[-1].append(stamp)
            group_index = int(confirmation_offset) % len(groups)
            group = groups[group_index]
            group_start = group[0]
            end_epoch = group[-1] + 60
            end = datetime.fromtimestamp(end_epoch, UTC)
            if end > now:
                # Do not ask the provider for future history. The bar remains
                # pending and will be requested on a subsequent cycle.
                end = now.replace(second=0, microsecond=0)
                end_epoch = int(end.timestamp())
            duration = end_epoch - group_start
            if duration < 60:
                raise RuntimeError("pending confirmation window has no completed minute to request")
            requests.append(HistoricalRequest("pending_confirmation", duration,
                end, tuple(group), group_index, len(groups)))
        if len(requests) > 2:
            raise AssertionError("request planner exceeded its per-cycle bound")
        return requests


class HistoricalRequestPacer:
    """Conservative two-request-per-30s loop guard and rolling request budget."""

    def __init__(self, *, minimum_cycle_seconds: int = 30, max_requests_per_10_minutes: int = 40,
                 prior_request_epochs: Iterable[float] = ()) -> None:
        if minimum_cycle_seconds < 30 or max_requests_per_10_minutes > 60 or max_requests_per_10_minutes < 1:
            raise ValueError("pacing policy exceeds conservative historical request limits")
        self.minimum_cycle_seconds = minimum_cycle_seconds
        self.max_requests_per_10_minutes = max_requests_per_10_minutes
        self._request_epochs: list[float] = list(prior_request_epochs)
        self._last_cycle_epoch: float | None = max(self._request_epochs, default=None)

    def begin_cycle(self, now: datetime, request_count: int) -> None:
        if now.tzinfo is None:
            raise ValueError("pacer time must be timezone-aware")
        epoch = now.astimezone(UTC).timestamp()
        if request_count < 0 or request_count > 2:
            raise ValueError("at most two historical requests are allowed per cycle")
        if self._last_cycle_epoch is not None and epoch - self._last_cycle_epoch < self.minimum_cycle_seconds:
            raise RuntimeError("historical polling cycle is too frequent")
        recent = [item for item in self._request_epochs if epoch - item < 600]
        if len(recent) + request_count > self.max_requests_per_10_minutes:
            raise RuntimeError("historical request rolling budget exceeded")
        self._request_epochs = recent + [epoch] * request_count
        self._last_cycle_epoch = epoch

    def reserve_reconnect_retry(self, now: datetime) -> None:
        """Count one bounded same-request retry after a detected disconnect."""
        if now.tzinfo is None:
            raise ValueError("pacer time must be timezone-aware")
        epoch = now.astimezone(UTC).timestamp()
        recent = [item for item in self._request_epochs if epoch - item < 600]
        if len(recent) + 1 > self.max_requests_per_10_minutes:
            raise RuntimeError("historical request rolling budget exceeded on reconnect retry")
        self._request_epochs = recent + [epoch]

    @property
    def recent_request_epochs(self) -> tuple[float, ...]:
        return tuple(self._request_epochs)

    def seconds_until_next_cycle(self, now: datetime) -> float:
        """Seconds remaining before another historical cycle is permitted."""
        if now.tzinfo is None:
            raise ValueError("pacer time must be timezone-aware")
        if self._last_cycle_epoch is None:
            return 0.0
        return max(0.0, self.minimum_cycle_seconds -
                   (now.astimezone(UTC).timestamp() - self._last_cycle_epoch))


def format_ibkr_end_time(value: datetime) -> str:
    """IBKR endDateTime in explicitly UTC format; bar timestamps stay epoch UTC."""
    if value.tzinfo is None:
        raise ValueError("IBKR request end time must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y%m%d %H:%M:%S UTC")


class DelayedAcquisitionSession:
    """Persist observations, advance the finalizer, then atomically checkpoint.

    Crash ordering is intentional: append raw observations first, append final
    delivery records through the finalizer second, and update this cursor last.
    Both journals are fsynced before this checkpoint. On restart their records
    are authoritative and can reconstruct any cursor snapshot lost mid-cycle.
    """

    def __init__(self, *, cursor_store: AtomicAcquisitionCursor,
                 observation_ledger: AppendOnlyBarObservationLedger,
                 finalizer: DelayedBarFinalizer) -> None:
        self.cursor_store = cursor_store
        self.observation_ledger = observation_ledger
        self.finalizer = finalizer
        self.cursor = cursor_store.cursor

    def ingest_response(self, rows: Iterable[dict], *, observed_at: datetime,
                        poll_number: int, request_id: int,
                        finalize_timestamps: Iterable[int] | None = None,
                        defer_finalization_timestamps: Iterable[int] = ()) -> list[FinalizedIBKRBar]:
        if observed_at.tzinfo is None:
            raise ValueError("response observation time must be timezone-aware")
        observations = []
        allowed_to_finalize = (None if finalize_timestamps is None
                               else {int(value) for value in finalize_timestamps})
        deferred = {int(value) for value in defer_finalization_timestamps}
        for row in rows:
            if int(row["con_id"]) != self.cursor.contract_id:
                raise ValueError("historical response contains an unapproved contract ID")
            timestamp = int(row["timestamp"])
            eligible = (timestamp not in deferred and
                        (allowed_to_finalize is None or timestamp in allowed_to_finalize))
            record = self.observation_ledger.append(
                row, contract={"con_id": self.cursor.contract_id,
                               "local_symbol": self.cursor.local_symbol,
                               "expiry": self.cursor.expiry},
                observed_at=row.get("arrival_utc", observed_at), request_id=request_id, poll_number=poll_number,
                eligible_for_finalization=eligible,
            )
            observations.append(record)
            self.cursor.last_observation_sequence = max(
                self.cursor.last_observation_sequence, int(record["observation_sequence"]))
        # Process in exchange-time order independent of provider row ordering.
        for record in sorted(observations, key=lambda item: int(item["bar_start_epoch_utc"])):
            timestamp = int(record["bar_start_epoch_utc"])
            if (timestamp in deferred or
                    (allowed_to_finalize is not None and timestamp not in allowed_to_finalize)):
                continue
            result = self.finalizer.observe(
                {"timestamp": int(record["bar_start_epoch_utc"]), **record["ohlcv"]},
                contract_id=int(record["contract_id"]), local_symbol=str(record["local_symbol"]),
                observed_at=record["observed_at_utc"], poll_number=poll_number,
            )
            if allowed_to_finalize is None:
                self.cursor.last_discovered_bar_start_epoch_utc = max(
                    self.cursor.last_discovered_bar_start_epoch_utc or 0,
                    int(record["bar_start_epoch_utc"]))
            # Invalid/unknown/out-of-session bars remain in the append-only
            # ledger and are visible as quality events; they are not delivered.
            if result in {"incomplete", "outside_session"}:
                continue
        finalized = self.finalizer.finalize_ready(now=observed_at)
        pending = [int(item.timestamp()) for item in self.finalizer.pending_timestamps]
        self.cursor.oldest_pending_bar_start_epoch_utc = min(pending) if pending else None
        self.cursor.pending_confirmation_group = tuple(pending)
        if finalized:
            self.cursor.last_delivered_bar_start_epoch_utc = int(finalized[-1].exchange_bar_timestamp_utc.timestamp())
        self.cursor.last_request_at_utc = observed_at.astimezone(UTC).isoformat()
        self.cursor_store.save()
        return finalized

    def pending_confirmation_starts(self) -> tuple[int, ...]:
        # Derive from live finalizer state rather than trusting a stale cursor
        # snapshot; this is also the restart-recovery consistency check.
        current = tuple(int(item.timestamp()) for item in self.finalizer.pending_timestamps)
        if current != self.cursor.pending_confirmation_group:
            self.cursor.pending_confirmation_group = current
            self.cursor.oldest_pending_bar_start_epoch_utc = min(current) if current else None
            self.cursor_store.save()
        return current

    def persist_request_budget(self, pacer: HistoricalRequestPacer, *, now: datetime) -> None:
        """Persist rolling pacing history before issuing planned API calls."""
        self.cursor.recent_request_epochs = pacer.recent_request_epochs
        self.cursor.last_request_at_utc = now.astimezone(UTC).isoformat()
        self.cursor_store.save()


def rehydrate_pending_from_observations(finalizer: DelayedBarFinalizer,
                                        records: Iterable[dict]) -> None:
    """Restore unfinalized candidates from the append-only observation journal.

    Replaying delivered records is safe: the finalizer recognizes its durable
    delivery journal and only records any later revisions as diagnostics.
    """
    ordered = sorted(records, key=lambda row: (int(row["poll_number"]),
                                                int(row["observation_sequence"])))
    for record in ordered:
        if not bool(record.get("eligible_for_finalization", True)):
            continue
        finalizer.observe(
            {"timestamp": int(record["bar_start_epoch_utc"]), **record["ohlcv"]},
            contract_id=int(record["contract_id"]), local_symbol=str(record["local_symbol"]),
            observed_at=record["observed_at_utc"], poll_number=int(record["poll_number"]),
        )


def request_plan_identity(request: HistoricalRequest) -> tuple[str, int, int | None]:
    """Stable key for diagnostic/idempotency reporting of scheduled requests."""
    return (request.kind, request.duration_seconds,
            int(request.end_time_utc.timestamp()) if request.end_time_utc else None)


def execute_with_bounded_reconnect(attempt, reconnect, *, max_reconnects: int = 1):
    """Retry a read-only historical request only after confirmed disconnection.

    `attempt` returns a mapping with `completed` and `connected`. Provider/API
    errors on a healthy socket are returned to the caller without retry.
    """
    if max_reconnects < 0 or max_reconnects > 3:
        raise ValueError("reconnect attempts must be bounded between zero and three")
    result = attempt()
    reconnects = 0
    while (not bool(result.get("completed")) and not bool(result.get("connected"))
           and reconnects < max_reconnects):
        reconnect()
        reconnects += 1
        result = attempt()
    return result, reconnects


class AppendOnlyGapClassificationLedger:
    """Explicit human-reviewed classification for an otherwise blocking gap."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[int, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._rows[int(row["bar_start_epoch_utc"])] = row

    @property
    def rows(self) -> dict[int, dict]:
        return dict(self._rows)

    def append(self, bar_start_epoch_utc: int, *, classification: str,
               evidence: str, reviewed_by: str) -> None:
        if classification != "operator_reviewed_missing_data" or not evidence.strip() or not reviewed_by.strip():
            raise ValueError("expected-gap override requires operator_reviewed_missing_data, evidence and reviewer")
        key = int(bar_start_epoch_utc)
        row = {"bar_start_epoch_utc": key,
               "bar_start_utc": datetime.fromtimestamp(key, UTC).isoformat(),
               "classification": classification, "evidence": evidence.strip(),
               "reviewed_by": reviewed_by.strip(), "recorded_at_utc": datetime.now(UTC).isoformat()}
        prior = self._rows.get(key)
        if prior is not None:
            if prior != row:
                raise RuntimeError("gap classification is append-only and already exists")
            return
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[key] = row
