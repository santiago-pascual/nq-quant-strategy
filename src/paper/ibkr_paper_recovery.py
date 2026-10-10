"""Crash-recoverable, PAPER-only handoff from finalized IBKR bars.

The module stays separate from the base Paper service so it can be tested and
versioned without changing the active historical runner. It never uses broker
order APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Iterable, Iterator, Mapping

import pandas as pd

from src.paper.cme_calendar import CMETradingCalendar
from src.paper.ibkr_delayed_market_data import AppendOnlyFinalizationLedger, IBKRContractSchedule
from src.paper.ibkr_observation_ledger import bar_value_hash
from src.paper.logger import PaperEventType
from src.paper.realtime_market_data import CanonicalBar
from src.paper.realtime_checkpoint import AtomicCheckpointStore, restore_engine_state, runtime_identity
from src.paper.realtime_service import RealtimePaperConfig, RealtimePaperService


def _utc(value: str | datetime) -> str:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("recovery timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


class AppendOnlyPaperDeliveryLedger:
    """Persist finalized bars as pending, then append Paper commit acknowledgments."""

    SCHEMA_VERSION = 1
    # The active r3 run created its append-only journal with this exact module
    # fingerprint. The current module only adds bounded acquisition/finalizer
    # fixes; the journal schema and commit protocol are unchanged. Accept this
    # one known predecessor fingerprint while preserving it in every record.
    # Unknown fingerprints continue to fail closed.
    LEGACY_COMPATIBLE_MODULE_SHA256 = frozenset({
        "4d2f4c5c4bc4ba1ecf47d819329da7c4320b50b84709162e0f65cf6bfac4dcf9",
    })

    def __init__(self, path: str | Path, *, contract_id: int, local_symbol: str,
                 expiry: str, calendar_identity: str) -> None:
        if contract_id <= 0 or not local_symbol or not expiry or not calendar_identity:
            raise ValueError("recovery ledger requires pinned contract and reviewed calendar identity")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "schema_version": self.SCHEMA_VERSION,
            "contract_id": int(contract_id), "local_symbol": local_symbol,
            "expiry": expiry, "calendar_identity": calendar_identity,
            "recovery_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.writer_module_sha256 = self.identity["recovery_module_sha256"]
        self.compatibility_mode = False
        self._bars: dict[int, dict[str, Any]] = {}
        self._committed: dict[int, str] = {}
        if self.path.exists():
            self._load()

    def _append(self, row: Mapping[str, Any]) -> None:
        encoded = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _load(self) -> None:
        identity_seen = False
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_identity = row.get("ledger_identity")
                if row_identity != self.identity and not identity_seen:
                    legacy_identity = dict(self.identity)
                    recorded_hash = row_identity.get("recovery_module_sha256") if isinstance(row_identity, dict) else None
                    legacy_identity["recovery_module_sha256"] = recorded_hash
                    if (recorded_hash in self.LEGACY_COMPATIBLE_MODULE_SHA256
                            and row_identity == legacy_identity):
                        # Keep the journal's original identity instead of
                        # rewriting history or pretending it was written by
                        # the current code. Subsequent appends remain clearly
                        # tagged with the writer hash below.
                        self.identity = dict(row_identity)
                        self.compatibility_mode = True
                if row_identity != self.identity:
                    raise ValueError("contract/calendar/schema identity differs")
                identity_seen = True
                epoch = int(row["bar_start_epoch_utc"])
                if row["kind"] == "finalized_pending":
                    prior = self._bars.get(epoch)
                    if prior and prior["bar_value_sha256"] != row["bar_value_sha256"]:
                        raise ValueError("conflicting finalized versions for one pending timestamp")
                    self._bars[epoch] = row
                elif row["kind"] == "paper_commit":
                    if epoch not in self._bars:
                        raise ValueError("commit acknowledgment has no staged finalized bar")
                    if row["bar_value_sha256"] != self._bars[epoch]["bar_value_sha256"]:
                        raise ValueError("commit acknowledgment hash differs from staged bar")
                    self._committed[epoch] = str(row["checkpoint_sha256"])
                else:
                    raise ValueError("unknown recovery journal record kind")
            except Exception as exc:
                raise ValueError(f"invalid Paper delivery journal record at line {number}: {exc}") from exc
        epochs = sorted(self._bars)
        if any(left >= right for left, right in zip(epochs, epochs[1:])):
            raise ValueError("staged delivery journal is not strictly ordered")

    def stage(self, finalized_bar: Mapping[str, Any], *, recovered: bool) -> bool:
        """Durably stage a validated finalized bar; conflicting revisions fail closed."""
        contract_id = int(finalized_bar["contract_id"])
        if contract_id != self.identity["contract_id"]:
            raise ValueError("bar contract ID differs from pinned recovery contract")
        stamp = _utc(str(finalized_bar["exchange_bar_timestamp_utc"]))
        epoch = int(datetime.fromisoformat(stamp).timestamp())
        if epoch % 60:
            raise ValueError("exchange bar timestamp must be aligned to a UTC minute")
        status = str(finalized_bar.get("data_quality_status", ""))
        if status != "validated_stable_completed_calendar_covered":
            raise ValueError("only finalized, stable, calendar-covered bars may be staged")
        if not finalized_bar.get("provider") or not finalized_bar.get("local_symbol"):
            raise ValueError("bar provider and contract symbol are required")
        ohlcv = {name: float(finalized_bar[name]) for name in ("open", "high", "low", "close", "volume")}
        if (not all(math.isfinite(value) for value in ohlcv.values())
                or ohlcv["volume"] < 0
                or ohlcv["high"] < max(ohlcv["open"], ohlcv["close"])
                or ohlcv["low"] > min(ohlcv["open"], ohlcv["close"])
                or ohlcv["high"] < ohlcv["low"]):
            raise ValueError("finalized bar contains invalid OHLCV values")
        if str(finalized_bar["local_symbol"]) != self.identity["local_symbol"]:
            raise ValueError("bar local symbol differs from pinned recovery contract")
        if str(finalized_bar.get("expiry", self.identity["expiry"])) != self.identity["expiry"]:
            raise ValueError("bar expiry differs from pinned recovery contract")
        digest = bar_value_hash({"timestamp": epoch, **ohlcv})
        prior = self._bars.get(epoch)
        if prior:
            if prior["bar_value_sha256"] != digest:
                raise RuntimeError("pending finalized bar changed; preserve it and audit the revision separately")
            return False
        if self._bars and epoch <= max(self._bars):
            raise ValueError("finalized bars must be staged in strict exchange-time order")
        first_seen = _utc(str(finalized_bar["first_observed_at_utc"]))
        finalized_at = _utc(str(finalized_bar["finalized_at_utc"]))
        row = {
            "kind": "finalized_pending", "ledger_identity": self.identity,
            "writer_recovery_module_sha256": self.writer_module_sha256,
            "bar_start_epoch_utc": epoch, "exchange_bar_timestamp_utc": stamp,
            "exchange_bar_end_utc": _utc(str(finalized_bar.get(
                "exchange_bar_end_utc", datetime.fromtimestamp(epoch + 60, timezone.utc).isoformat()
            ))),
            "first_observed_at_utc": first_seen, "finalized_at_utc": finalized_at,
            "provider": str(finalized_bar["provider"]),
            "contract_id": contract_id, "local_symbol": str(finalized_bar["local_symbol"]),
            "expiry": self.identity["expiry"], "data_quality_status": status,
            "bar_value_sha256": digest, "ohlcv": ohlcv,
            "provenance": "RECOVERED_PAPER" if recovered else "CONTINUOUS_PAPER",
        }
        self._append(row)
        self._bars[epoch] = row
        return True

    def pending_after(self, checkpoint_timestamp: str | datetime | None) -> list[dict[str, Any]]:
        """Return unacknowledged staged bars after a validated Paper checkpoint."""
        cutoff = int(datetime.fromisoformat(_utc(checkpoint_timestamp)).timestamp()) if checkpoint_timestamp else -1
        if any(epoch > cutoff for epoch in self._committed):
            raise ValueError("Paper checkpoint is older than an acknowledged delivery; refusing rollback")
        return [dict(row) for epoch, row in sorted(self._bars.items())
                if epoch > cutoff and epoch not in self._committed]

    def acknowledge_through(self, checkpoint_timestamp: str | datetime,
                            checkpoint_sha256: str) -> int:
        """Acknowledge staged bars only when the checkpoint timestamp covers them."""
        if len(checkpoint_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_sha256.lower()):
            raise ValueError("Paper checkpoint SHA-256 is required")
        cutoff = int(datetime.fromisoformat(_utc(checkpoint_timestamp)).timestamp())
        eligible = [epoch for epoch in sorted(self._bars)
                    if epoch <= cutoff and epoch not in self._committed]
        for epoch in eligible:
            row = {
                "kind": "paper_commit", "ledger_identity": self.identity,
                "writer_recovery_module_sha256": self.writer_module_sha256,
                "bar_start_epoch_utc": epoch,
                "bar_value_sha256": self._bars[epoch]["bar_value_sha256"],
                "paper_checkpoint_timestamp_utc": _utc(checkpoint_timestamp),
                "checkpoint_sha256": checkpoint_sha256.lower(),
                "acknowledged_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._append(row)
            self._committed[epoch] = row["checkpoint_sha256"]
        return len(eligible)

    def reconcile_checkpoint(self, checkpoint_timestamp: str | datetime,
                             checkpoint_sha256: str) -> int:
        """Repair a crash after Paper checkpoint replacement but before cursor ack."""
        return self.acknowledge_through(checkpoint_timestamp, checkpoint_sha256)

    @property
    def committed_timestamps(self) -> tuple[int, ...]:
        return tuple(sorted(self._committed))


class IBKRFinalizedLedgerSource:
    """Read finalized IBKR bars and redeliver anything beyond Paper's commit.

    ``refresh`` is an injected callback that uses the existing bounded/paced
    acquisition code to append newly finalized bars. Without it this source is
    a finite, journal-only source suitable for isolated recovery validation.
    It contains no TWS connection or order API calls.
    """

    name = "ibkr_delayed_recoverable_paper"

    def __init__(self, *, finalization_ledger: AppendOnlyFinalizationLedger,
                 delivery_ledger: AppendOnlyPaperDeliveryLedger,
                 contract_schedule: IBKRContractSchedule,
                 calendar: CMETradingCalendar,
                 refresh: Callable[[datetime | None], None] | None = None,
                 begin_catchup: Callable[[datetime | None], None] | None = None,
                 recovery_status: Callable[[], Mapping[str, Any]] | None = None,
                 is_recovered_bar: Callable[[int], bool] | None = None,
                 mark_recovery_verified: Callable[[], None] | None = None,
                 poll_interval_seconds: float = 30.0,
                 bootstrap_after_timestamp: datetime | None = None) -> None:
        if poll_interval_seconds < 30:
            raise ValueError("IBKR historical refresh interval must be at least 30 seconds")
        if delivery_ledger.identity["calendar_identity"] != calendar.snapshot.identity:
            raise ValueError("delivery ledger calendar identity differs from runtime calendar")
        self.finalization_ledger = finalization_ledger
        self.delivery_ledger = delivery_ledger
        self.contract_schedule = contract_schedule
        self.calendar = calendar
        self.refresh_callback = refresh
        self.begin_catchup_callback = begin_catchup
        self.recovery_status_callback = recovery_status
        self.is_recovered_bar_callback = is_recovered_bar
        self.mark_recovery_verified_callback = mark_recovery_verified
        self.poll_interval_seconds = float(poll_interval_seconds)
        if bootstrap_after_timestamp is not None and bootstrap_after_timestamp.tzinfo is None:
            raise ValueError("causal bootstrap cutoff must be timezone-aware")
        self.bootstrap_after_timestamp = (
            bootstrap_after_timestamp.astimezone(timezone.utc)
            if bootstrap_after_timestamp is not None else None
        )
        self._started = False
        self._stopped = threading.Event()
        self._wake = threading.Event()
        self._after: datetime | None = None
        self._cursor: datetime | None = None
        self._audit: dict[int, dict[str, Any]] = {}
        self._last_error: str | None = None
        self._transport_available: bool | None = None
        self._bars_emitted = 0

    @staticmethod
    def _record_epoch(record: Mapping[str, Any]) -> int:
        return int(record.get("bar_start_epoch_utc", datetime.fromisoformat(
            str(record["exchange_bar_timestamp_utc"])
        ).timestamp()))

    def _read_finalized_rows(self) -> list[dict[str, Any]]:
        path = self.finalization_ledger.path
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.sort(key=self._record_epoch)
        return rows

    def _committed_cursor(self) -> datetime | None:
        """Latest Paper-acknowledged timestamp, independent of source emission."""
        committed = self.delivery_ledger.committed_timestamps
        latest = datetime.fromtimestamp(committed[-1], timezone.utc) if committed else None
        if self._after is None:
            return latest
        if latest is None or self._after > latest:
            return self._after
        return latest

    def _stage_finalized_rows(self, *, recovered: bool) -> None:
        cutoff = int(self._after.timestamp()) if self._after is not None else -1
        rows = self._read_finalized_rows()
        previous = cutoff
        for record in rows:
            epoch = self._record_epoch(record)
            if epoch <= cutoff:
                continue
            stamp = datetime.fromtimestamp(epoch, timezone.utc)
            contract = self.contract_schedule.resolve(stamp)
            if (contract.con_id != int(record["contract_id"])
                    or contract.local_symbol != str(record["contract_symbol"])):
                raise ValueError("finalized IBKR bar does not match reviewed contract schedule")
            if not self.calendar.expected_globex_minute(pd.Timestamp(stamp)):
                raise ValueError("finalized IBKR bar is outside reviewed CME Globex coverage")
            ohlcv = dict(record["ohlcv"])
            staged = {
                **ohlcv,
                "contract_id": int(record["contract_id"]),
                "local_symbol": str(record["contract_symbol"]),
                "expiry": self.delivery_ledger.identity["expiry"],
                "exchange_bar_timestamp_utc": record["exchange_bar_timestamp_utc"],
                "exchange_bar_end_utc": record["exchange_bar_end_utc"],
                "first_observed_at_utc": record["first_observed_at_utc"],
                "finalized_at_utc": record["finalized_at_utc"],
                "provider": record["provider"],
                "data_quality_status": record["data_quality_status"],
            }
            recovered_bar = recovered or bool(
                self.is_recovered_bar_callback and self.is_recovered_bar_callback(epoch)
            )
            self.delivery_ledger.stage(staged, recovered=recovered_bar)
            self._audit[epoch] = dict(record)
            previous = epoch
        # Validate every expected Globex minute between the Paper cursor and
        # the staged frontier. Closures are excluded by the reviewed calendar.
        committed_cursor = self._committed_cursor()
        pending = self.delivery_ledger.pending_after(committed_cursor)
        available = {int(row["bar_start_epoch_utc"]) for row in pending}
        cursor = int(committed_cursor.timestamp()) if committed_cursor is not None else -1
        for row in pending:
            epoch = int(row["bar_start_epoch_utc"])
            if cursor >= 0:
                missing = self.calendar.expected_missing_minutes(
                    datetime.fromtimestamp(cursor, timezone.utc),
                    datetime.fromtimestamp(epoch, timezone.utc),
                )
                absent = [int(item.timestamp()) for item in missing if int(item.timestamp()) not in available]
                if absent:
                    raise RuntimeError(
                        f"IBKR finalized journal has {len(absent)} unverified open-session bars missing; "
                        f"first={datetime.fromtimestamp(absent[0], timezone.utc).isoformat()}"
                    )
            cursor = epoch

    def start(self) -> None:
        self._started = True
        self._stopped.clear()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()

    def subscribe(self, symbols: tuple[str, ...], *, after_timestamp: datetime | None = None) -> None:
        if symbols != ("MNQ",):
            raise ValueError("IBKR finalized source is pinned to MNQ")
        if after_timestamp is not None and after_timestamp.tzinfo is None:
            raise ValueError("Paper resume timestamp must be timezone-aware")
        self._after = (
            after_timestamp.astimezone(timezone.utc) if after_timestamp
            else self.bootstrap_after_timestamp
        )
        self._cursor = self._after
        if self.begin_catchup_callback is not None:
            self.begin_catchup_callback(self._after)
        self._stage_finalized_rows(recovered=True)

    def _canonical(self, row: Mapping[str, Any]) -> CanonicalBar:
        epoch = int(row["bar_start_epoch_utc"])
        stamp = datetime.fromtimestamp(epoch, timezone.utc)
        metadata = self._audit.get(epoch, row)
        final_rth = self.calendar.is_final_rth_bar(stamp) if self.calendar.is_rth(stamp) else False
        provenance = str(row["provenance"])
        return CanonicalBar(
            symbol="MNQ", timestamp=stamp,
            open=float(row["ohlcv"]["open"]), high=float(row["ohlcv"]["high"]),
            low=float(row["ohlcv"]["low"]), close=float(row["ohlcv"]["close"]),
            volume=float(row["ohlcv"]["volume"]), provider=str(row["provider"]),
            source_id=f"{row['contract_id']}:{provenance}",
            is_session_final=final_rth, contract_symbol=str(row["local_symbol"]),
            provider_timestamp=datetime.fromisoformat(str(metadata["finalized_at_utc"])).astimezone(timezone.utc),
        )

    def audit_metadata(self, timestamp: datetime) -> Mapping[str, Any] | None:
        epoch = int(timestamp.astimezone(timezone.utc).timestamp())
        return self._audit.get(epoch) or self.delivery_ledger._bars.get(epoch)

    def bars(self) -> Iterator[CanonicalBar]:
        if not self._started:
            raise RuntimeError("IBKR finalized source must be started before iteration")
        first_cycle = True
        first_refresh = True
        while not self._stopped.is_set():
            self._stage_finalized_rows(recovered=first_cycle)
            first_cycle = False
            pending = self.delivery_ledger.pending_after(self._committed_cursor())
            for row in pending:
                if self._stopped.is_set():
                    return
                stamp = datetime.fromtimestamp(int(row["bar_start_epoch_utc"]), timezone.utc)
                if self._cursor is not None and stamp <= self._cursor:
                    continue
                self.contract_schedule.resolve(stamp)
                if not self.calendar.expected_globex_minute(pd.Timestamp(stamp)):
                    raise RuntimeError("CME calendar coverage unavailable for staged IBKR bar")
                self._cursor = stamp
                self._bars_emitted += 1
                yield self._canonical(row)
            if self.refresh_callback is None:
                return
            wait_seconds = 0.0 if first_refresh else self.poll_interval_seconds
            first_refresh = False
            if wait_seconds and self.recovery_status_callback is not None:
                reconnection = dict(self.recovery_status_callback())
                retry_at = reconnection.get("next_retry_at_utc")
                if retry_at:
                    try:
                        retry_time = datetime.fromisoformat(str(retry_at).replace("Z", "+00:00"))
                        if retry_time.tzinfo is not None:
                            wait_seconds = min(wait_seconds, max(0.1, (retry_time - datetime.now(timezone.utc)).total_seconds()))
                    except ValueError:
                        pass
            self._wake.wait(wait_seconds)
            self._wake.clear()
            if not self._stopped.is_set():
                try:
                    refresh_result = self.refresh_callback(self._cursor)
                    # A TWS handshake loss is a recoverable transport state:
                    # keep the Paper loop alive, expose the error, and retry
                    # on the next paced poll. Clear it only after a complete
                    # provider cycle, never merely because the callback returned.
                    if isinstance(refresh_result, Mapping) and refresh_result.get("transport_available") is False:
                        self._transport_available = False
                        self._last_error = str(
                            refresh_result.get("last_transport_error")
                            or "IBKR transport unavailable; waiting for next retry"
                        )
                    else:
                        self._transport_available = True
                        self._last_error = None
                except BaseException as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    raise

    def backfill(self, *, after_timestamp: datetime, before_timestamp: datetime) -> Iterable[CanonicalBar]:
        if after_timestamp.tzinfo is None or before_timestamp.tzinfo is None or before_timestamp <= after_timestamp:
            raise ValueError("IBKR backfill requires an aware, increasing exclusive interval")
        after = int(after_timestamp.astimezone(timezone.utc).timestamp())
        before = int(before_timestamp.astimezone(timezone.utc).timestamp())
        return [self._canonical(row) for row in self.delivery_ledger.pending_after(after_timestamp)
                if after < int(row["bar_start_epoch_utc"]) < before]

    def health(self) -> Mapping[str, Any]:
        committed_cursor = self._committed_cursor()
        pending = self.delivery_ledger.pending_after(committed_cursor)
        recovery = dict(self.recovery_status_callback()) if self.recovery_status_callback else {}
        finalized_rows = self._read_finalized_rows()
        journal_latest = max((row["exchange_bar_timestamp_utc"] for row in finalized_rows), default=None)
        observed_epoch = recovery.get("latest_observed_bar_epoch_utc")
        observed_latest = (datetime.fromtimestamp(int(observed_epoch), timezone.utc).isoformat()
                           if observed_epoch is not None else None)
        latest_available = max((item for item in (journal_latest, observed_latest) if item), default=None)
        connection_state = str(recovery.get("connection_state", "UNKNOWN")).upper()
        if connection_state in {"DISCONNECTED", "RECONNECTING"}:
            mode = connection_state
        elif str(recovery.get("acquisition_state", "")).upper() in {
            "PAUSED_PACING", "WAITING_FOR_PROVIDER_DATA"
        }:
            mode = "RECOVERING"
        elif connection_state == "RECOVERING" or recovery.get("catchup_active") or pending:
            mode = "RECOVERING"
        elif latest_available is None:
            mode = "WAITING_FOR_DATA"
        else:
            mode = "CAUGHT_UP"
        recovery_verified = bool(
            self._started and not self._stopped.is_set()
            and self._transport_available is True
            and not recovery.get("catchup_active")
            and not pending
            and int(recovery.get("pending_confirmation_count", 0) or 0) == 0
            and recovery.get("last_successful_request_utc")
        )
        if (recovery_verified and self.mark_recovery_verified_callback is not None
                and self.recovery_status_callback is not None):
            self.mark_recovery_verified_callback()
            recovery.update(dict(self.recovery_status_callback()))
            connection_state = str(recovery.get("connection_state", "CONNECTED")).upper()
            mode = "CAUGHT_UP"
        return {
            # ``connected`` preserves the MarketDataSource liveness contract:
            # the reader can remain running while the provider is down.
            # Provider truth is separately reported and is UNKNOWN until a
            # handshake/request has actually been observed.
            "connected": (self._started and not self._stopped.is_set()
                          and self._transport_available is not False),
            "provider_connected": self._transport_available,
            "stale": False, "source": self.name,
            "mode": mode,
            "last_processed_bar": self._cursor.isoformat() if self._cursor else None,
            "latest_available_bar": latest_available,
            "backlog_bars": len(pending), "bars_emitted": self._bars_emitted,
            "last_error": self._last_error or recovery.get("last_transport_error"),
            **recovery,
        }


class AcknowledgedDelayedPaperService(RealtimePaperService):
    """Paper service whose feed acknowledgment follows a durable per-bar checkpoint."""

    def __init__(self, *, delivery_ledger: AppendOnlyPaperDeliveryLedger, **kwargs: Any) -> None:
        config: RealtimePaperConfig = kwargs["config"]
        if config.checkpoint_every_bars != 1:
            raise ValueError("recoverable delayed Paper requires checkpoint_every_bars=1")
        if kwargs.get("source") is None or kwargs["source"].delivery_ledger is not delivery_ledger:
            raise ValueError("Paper source and service must share one delivery ledger")
        self.delivery_ledger = delivery_ledger
        super().__init__(**kwargs)

    def _checkpoint_sha256(self) -> str:
        for path in (self.checkpoint_store.path, self.checkpoint_store.backup_path):
            if not path.exists():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                envelope = {"schema_version": document["schema_version"], "payload": document["payload"]}
                body = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=True)
                if hashlib.sha256(body.encode()).hexdigest() == document.get("sha256"):
                    return hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        raise ValueError("no valid Paper checkpoint exists for delivery acknowledgment")

    def restore(self) -> None:
        has_bar_checkpoint = (self.checkpoint_store.path.exists()
                              or self.checkpoint_store.backup_path.exists())
        if has_bar_checkpoint:
            super().restore()
        else:
            payload = self.bootstrap_store.load()
            if payload.get("mode") != "PAPER" or payload.get("symbol") != self.config.symbol:
                raise ValueError("bootstrap checkpoint mode or symbol differs from runtime config")
            from src.paper.runtime_compatibility import validate_runtime_identity
            compatibility = validate_runtime_identity(
                payload.get("system", {}).get("runtime_identity"), runtime_identity()
            )
            if not compatibility.accepted:
                raise ValueError(f"bootstrap checkpoint runtime identity rejected: {compatibility.reason}")
            expected_calendar = self.calendar.snapshot.identity if self.calendar else None
            if payload.get("system", {}).get("calendar_identity") != expected_calendar:
                raise ValueError("bootstrap checkpoint CME calendar differs from runtime")
            expected_cost = self.cost_policy.identity if self.cost_policy else None
            if payload.get("system", {}).get("cost_profile_identity") != expected_cost:
                raise ValueError("bootstrap checkpoint cost profile differs from runtime")
            self.context_adapter.context.load_state_dict(payload["context"])
            restore_engine_state(self.engine, payload["engine"])
            runtime = payload["runtime"]
            self.run_id = str(runtime["run_id"])
            self._last_bar = None
            self._last_bar_digest = None
            self._bars_processed = 0
            self._risk_per_contract = {}
            self._equity_peak = float(self.engine.account_equity)
            self._portfolio_max_drawdown = 0.0
            self._daily_date = None
            self._daily_counts.clear()
            self.logger.set_event_context(run_id=self.run_id, severity="INFO", symbol=self.config.symbol)
            self._emit(PaperEventType.SYSTEM_RECOVERED, {
                "run_id": self.run_id, "severity": "INFO", "symbol": self.config.symbol,
                "last_processed_bar": None, "recovery_source": "causal_bootstrap_checkpoint",
                "replay_from_start": True,
                "runtime_compatibility": compatibility.as_dict(),
            }, timestamp=datetime.now(timezone.utc))
            self._write_status()
        if self._last_bar is not None:
            self.delivery_ledger.reconcile_checkpoint(
                self._last_bar.timestamp, self._checkpoint_sha256()
            )

    def save_checkpoint(self) -> None:
        if self._last_bar is None:
            return super().save_checkpoint()
        # The base service appends most bar events before checkpointing; force
        # those JSONL records durable before the state checkpoint can commit.
        with self.logger.path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        # CanonicalBar.provider_timestamp is a datetime. The current base
        # checkpoint's runtime.last_bar JSON path does not encode that field,
        # although the exchange timestamp itself is serialized explicitly.
        # Keep the provider observation/finalization time in the durable
        # delivery journal and event context, and omit it only from this
        # redundant checkpoint copy.
        committed_bar = self._last_bar
        self._last_bar = replace(committed_bar, provider_timestamp=None)
        try:
            super().save_checkpoint()
        finally:
            self._last_bar = committed_bar
        self.delivery_ledger.acknowledge_through(
            self._last_bar.timestamp, self._checkpoint_sha256()
        )

    def _append_refit_ledger(self, row: Mapping[str, Any]) -> None:
        path = self.refit_ledger_path
        identity = (str(row.get("stream")), str(row.get("model_version")),
                    str(row.get("first_live_timestamp", row.get("fit_timestamp"))))
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                prior = json.loads(line)
                prior_identity = (str(prior.get("stream")), str(prior.get("model_version")),
                                 str(prior.get("first_live_timestamp", prior.get("fit_timestamp"))))
                if prior_identity == identity:
                    if prior.get("model_hash") != row.get("model_hash"):
                        raise RuntimeError("replayed HMM refit identity produced a different model hash")
                    return
        super()._append_refit_ledger(row)

    def _process(self, bar: CanonicalBar) -> bool:
        _, separator, provenance = str(bar.source_id or "").partition(":")
        if separator and provenance:
            self.logger.set_event_context(
                market_data_provenance=provenance,
                market_data_first_observed_at_utc=(
                    self.source.audit_metadata(bar.timestamp).get("first_observed_at_utc")
                    if isinstance(self.source, IBKRFinalizedLedgerSource)
                    and self.source.audit_metadata(bar.timestamp) else None
                ),
                market_data_finalized_at_utc=bar.provider_timestamp.isoformat()
                if bar.provider_timestamp else None,
                market_data_contract_id=(
                    self.source.audit_metadata(bar.timestamp).get("contract_id")
                    if isinstance(self.source, IBKRFinalizedLedgerSource)
                    and self.source.audit_metadata(bar.timestamp) else None
                ),
            )
        return super()._process(bar)
