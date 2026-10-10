"""Standalone delayed IBKR bar validation/finalization primitives.

This module is not wired into the Paper runner. An explicit reviewed contract
schedule is required; listed-contract data is never silently treated as a
continuous MNQ series.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.paper.cme_calendar import CMETradingCalendar
from src.paper.ibkr_observation_ledger import bar_value_hash
from src.paper.realtime_market_data import CanonicalBar


class ContractScheduleUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class IBKRContractWindow:
    con_id: int
    local_symbol: str
    start_utc: datetime
    end_utc: datetime

    def __post_init__(self) -> None:
        start, end = pd.Timestamp(self.start_utc), pd.Timestamp(self.end_utc)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("contract schedule boundaries must be timezone-aware")
        if not self.local_symbol or self.con_id <= 0 or start >= end:
            raise ValueError("invalid IBKR contract window")
        object.__setattr__(self, "start_utc", start.tz_convert("UTC").to_pydatetime())
        object.__setattr__(self, "end_utc", end.tz_convert("UTC").to_pydatetime())


class IBKRContractSchedule:
    """Operator-reviewed continuous mapping; gaps and overlaps fail closed."""

    def __init__(self, windows: list[IBKRContractWindow]) -> None:
        if not windows:
            raise ContractScheduleUnavailable("no reviewed IBKR contract windows configured")
        ordered = sorted(windows, key=lambda item: item.start_utc)
        for left, right in zip(ordered, ordered[1:]):
            if left.end_utc != right.start_utc:
                raise ContractScheduleUnavailable("contract schedule must be contiguous with no overlap or uncovered roll interval")
        self.windows = tuple(ordered)

    def resolve(self, timestamp: datetime | pd.Timestamp) -> IBKRContractWindow:
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError("bar timestamps must be timezone-aware")
        utc = stamp.tz_convert("UTC").to_pydatetime()
        matches = [window for window in self.windows if window.start_utc <= utc < window.end_utc]
        if len(matches) != 1:
            raise ContractScheduleUnavailable(f"no unique reviewed contract mapping covers {utc.isoformat()}")
        return matches[0]


@dataclass(frozen=True)
class FinalizationPolicy:
    minimum_age_seconds: int = 600
    required_identical_observations: int = 2
    minimum_observation_spacing_seconds: int = 30

    def __post_init__(self) -> None:
        if self.minimum_age_seconds < 60:
            raise ValueError("minimum_age_seconds must be at least one bar")
        if self.required_identical_observations < 2:
            raise ValueError("at least two matching observations are required")
        if self.minimum_observation_spacing_seconds < 15:
            raise ValueError("observation spacing must respect IBKR identical-request pacing")


@dataclass(frozen=True)
class FinalizedIBKRBar:
    bar: CanonicalBar
    contract_id: int
    exchange_bar_timestamp_utc: datetime
    first_observed_at_utc: datetime
    finalized_at_utc: datetime
    data_quality_status: str
    stabilization_observations: int


class AppendOnlyFinalizationLedger:
    """Durable delivery journal used to resume without replaying emitted bars."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[tuple[int, int], str] = {}
        self._records: dict[tuple[int, int], dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["contract_id"]), int(row["bar_start_epoch_utc"]))
                self._rows[key] = str(row["bar_value_sha256"])
                self._records[key] = row

    @property
    def rows(self) -> Mapping[tuple[int, int], str]:
        return dict(self._rows)

    @property
    def records(self) -> Mapping[tuple[int, int], Mapping[str, Any]]:
        return {key: dict(value) for key, value in self._records.items()}

    def append(self, item: FinalizedIBKRBar) -> bool:
        key = (item.contract_id, int(item.exchange_bar_timestamp_utc.timestamp()))
        digest = bar_value_hash({"timestamp": key[1], "open": item.bar.open, "high": item.bar.high,
                                 "low": item.bar.low, "close": item.bar.close, "volume": item.bar.volume})
        prior = self._rows.get(key)
        if prior == digest:
            return False
        if prior is not None:
            raise RuntimeError("attempted to revise an already finalized Paper bar")
        if self._rows and key[1] <= max(stamp for _, stamp in self._rows):
            raise RuntimeError("finalized bar journal would violate exchange-time order")
        record = {
            "contract_id": item.contract_id,
            "bar_start_epoch_utc": key[1],
            "exchange_bar_timestamp_utc": item.exchange_bar_timestamp_utc.isoformat(),
            "exchange_bar_end_utc": (item.exchange_bar_timestamp_utc + timedelta(minutes=1)).isoformat(),
            "first_observed_at_utc": item.first_observed_at_utc.isoformat(),
            "finalized_at_utc": item.finalized_at_utc.isoformat(),
            "provider": item.bar.provider,
            "source_id": item.bar.source_id,
            "contract_symbol": item.bar.contract_symbol,
            "data_quality_status": item.data_quality_status,
            "stabilization_observations": item.stabilization_observations,
            "ohlcv": {"open": item.bar.open, "high": item.bar.high, "low": item.bar.low,
                      "close": item.bar.close, "volume": item.bar.volume},
            "bar_value_sha256": digest,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[key] = digest
        self._records[key] = record
        return True


class DelayedBarFinalizer:
    """Stabilize observed versions without rewriting already delivered bars."""

    def __init__(self, *, contract_schedule: IBKRContractSchedule,
                 calendar: CMETradingCalendar, policy: FinalizationPolicy = FinalizationPolicy(),
                 after_timestamp: datetime | None = None,
                 finalization_ledger: AppendOnlyFinalizationLedger | None = None,
                 gap_classifications: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        self.contract_schedule = contract_schedule
        self.calendar = calendar
        self.policy = policy
        self.last_delivered = (pd.Timestamp(after_timestamp).tz_convert("UTC").to_pydatetime()
                               if after_timestamp is not None else None)
        self._pending: dict[tuple[int, int], dict[str, Any]] = {}
        self._delivered: dict[tuple[int, int], str] = {}
        self._delivered_finalized_at: dict[tuple[int, int], datetime] = {}
        self.finalization_ledger = finalization_ledger
        self._reported_gaps: set[datetime] = set()
        self.gap_classifications = {int(key): dict(value)
                                    for key, value in (gap_classifications or {}).items()}
        if finalization_ledger is not None:
            journal_rows = finalization_ledger.rows
            for delivered_key, record in getattr(finalization_ledger, "records", {}).items():
                self._delivered_finalized_at[delivered_key] = self._utc(record["finalized_at_utc"])
            if journal_rows:
                latest_start = max(stamp for _, stamp in journal_rows)
                journal_timestamp = datetime.fromtimestamp(latest_start, timezone.utc)
                if self.last_delivered is None or journal_timestamp > self.last_delivered:
                    self.last_delivered = journal_timestamp
                self._delivered.update(journal_rows)
        self.late_revisions: list[dict[str, Any]] = []
        self.quality_events: list[dict[str, Any]] = []

    @property
    def pending_timestamps(self) -> tuple[datetime, ...]:
        return tuple(datetime.fromtimestamp(key[1], timezone.utc)
                     for key in sorted(self._pending, key=lambda item: item[1]))

    @staticmethod
    def _utc(value: datetime | str | pd.Timestamp) -> datetime:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("observation timestamps must be timezone-aware")
        return stamp.tz_convert("UTC").to_pydatetime()

    @staticmethod
    def _validate_ohlcv(row: Mapping[str, Any]) -> dict[str, float]:
        values = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite OHLCV")
        if min(values[key] for key in ("open", "high", "low", "close")) <= 0 or values["volume"] < 0:
            raise ValueError("prices must be positive and volume non-negative")
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError("high is inconsistent with OHLC")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError("low is inconsistent with OHLC")
        return values

    def observe(self, row: Mapping[str, Any], *, contract_id: int,
                local_symbol: str, observed_at: datetime | str | pd.Timestamp,
                poll_number: int) -> str:
        stamp = pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC")
        if stamp.second != 0 or stamp.microsecond != 0:
            raise ValueError("one-minute bar timestamp must align to minute start")
        start = stamp.to_pydatetime()
        observed = self._utc(observed_at)
        schedule = self.contract_schedule.resolve(start)
        if schedule.con_id != int(contract_id) or schedule.local_symbol != local_symbol:
            raise ContractScheduleUnavailable("bar contract does not match reviewed mapping for its timestamp")
        values = self._validate_ohlcv(row)
        if start + timedelta(minutes=1) > observed:
            self.quality_events.append({"kind": "incomplete_bar_observed", "timestamp": start.isoformat()})
            return "incomplete"
        # Calendar coverage is mandatory; unknown sessions do not become bars.
        if not self.calendar.expected_globex_minute(pd.Timestamp(start)):
            self.quality_events.append({"kind": "bar_outside_verified_globex_session", "timestamp": start.isoformat()})
            return "outside_session"
        key = (int(contract_id), int(stamp.timestamp()))
        digest = bar_value_hash({"timestamp": row["timestamp"], **values})
        if self.last_delivered is not None and start <= self.last_delivered:
            previous = self._delivered.get(key)
            if previous is not None and previous != digest:
                delivered_at = self._delivered_finalized_at.get(key)
                if delivered_at is not None and observed <= delivered_at:
                    return "observed_before_delivery_during_recovery"
                self.late_revisions.append({"contract_id": contract_id, "timestamp": start.isoformat(),
                                            "observed_at": observed.isoformat(), "new_hash": digest,
                                            "delivered_hash": previous})
                return "revision_after_delivery"
            return "already_delivered"
        prior = self._pending.get(key)
        if prior is None:
            self._pending[key] = {"values": values, "hash": digest, "first_observed": observed,
                                  "last_observed": observed, "stable_count": 1,
                                  "last_poll": int(poll_number), "contract_symbol": local_symbol}
            return "new_observation"
        if prior["hash"] != digest:
            prior.update(values=values, hash=digest, first_observed=prior["first_observed"],
                         last_observed=observed, stable_count=1, last_poll=int(poll_number))
            self.quality_events.append({"kind": "pre_finalization_revision", "timestamp": start.isoformat(),
                                        "observed_at": observed.isoformat()})
            return "revised_before_finalization"
        if int(poll_number) != prior["last_poll"] and (observed - prior["last_observed"]).total_seconds() >= self.policy.minimum_observation_spacing_seconds:
            prior["stable_count"] += 1
            prior["last_observed"] = observed
            prior["last_poll"] = int(poll_number)
        return "unchanged_observation"

    def finalize_ready(self, *, now: datetime | str | pd.Timestamp) -> list[FinalizedIBKRBar]:
        current = self._utc(now)
        output: list[FinalizedIBKRBar] = []
        for key in sorted(self._pending, key=lambda item: item[1]):
            pending = self._pending[key]
            start = datetime.fromtimestamp(key[1], timezone.utc)
            if pending["last_observed"] > current:
                break
            if pending["stable_count"] < self.policy.required_identical_observations:
                break
            if (current - (start + timedelta(minutes=1))).total_seconds() < self.policy.minimum_age_seconds:
                break
            if self.last_delivered is not None and start <= self.last_delivered:
                continue
            if self.last_delivered is not None:
                cursor = self.last_delivered + timedelta(minutes=1)
                # Expected session minutes absent from the pending set block
                # chronological delivery; closures are skipped by calendar.
                pending_starts = {datetime.fromtimestamp(k[1], timezone.utc) for k in self._pending}
                while cursor < start:
                    if self.calendar.expected_globex_minute(pd.Timestamp(cursor)) and cursor not in pending_starts:
                        classification = self.gap_classifications.get(int(cursor.timestamp()))
                        if classification and classification.get("classification") == "operator_reviewed_missing_data":
                            self.quality_events.append({"kind": "explicitly_classified_expected_gap",
                                "timestamp": cursor.isoformat(), "evidence": classification.get("evidence"),
                                "reviewed_by": classification.get("reviewed_by")})
                            cursor += timedelta(minutes=1)
                            continue
                        if cursor not in self._reported_gaps:
                            self.quality_events.append({"kind": "unresolved_expected_minute_gap", "timestamp": cursor.isoformat()})
                            self._reported_gaps.add(cursor)
                        break
                    cursor += timedelta(minutes=1)
                else:
                    cursor = None
                if cursor is not None:
                    break
            stamp = pd.Timestamp(start)
            session_final = self.calendar.is_final_rth_bar(stamp)
            canonical = CanonicalBar(
                symbol="MNQ", timestamp=start,
                open=pending["values"]["open"], high=pending["values"]["high"],
                low=pending["values"]["low"], close=pending["values"]["close"],
                volume=pending["values"]["volume"], completed=True,
                provider="ibkr_tws_delayed_historical", source_id=str(key[0]),
                is_session_final=session_final, contract_symbol=pending["contract_symbol"],
                provider_timestamp=pending["last_observed"],
            )
            finalized = FinalizedIBKRBar(
                bar=canonical, contract_id=key[0], exchange_bar_timestamp_utc=start,
                first_observed_at_utc=pending["first_observed"], finalized_at_utc=current,
                data_quality_status="validated_stable_completed_calendar_covered",
                stabilization_observations=pending["stable_count"],
            )
            if self.finalization_ledger is not None:
                self.finalization_ledger.append(finalized)
            self.last_delivered = start
            self._delivered[key] = pending["hash"]
            self._delivered_finalized_at[key] = current
            del self._pending[key]
            output.append(finalized)
        return output


def finalize_observation_sequence(records: list[Mapping[str, Any]], *,
                                 contract_schedule: IBKRContractSchedule,
                                 calendar: CMETradingCalendar,
                                 policy: FinalizationPolicy = FinalizationPolicy(),
                                 finalization_ledger: AppendOnlyFinalizationLedger | None = None) -> tuple[DelayedBarFinalizer, list[FinalizedIBKRBar]]:
    """Replay recorded observation times through the gate, poll by poll.

    Intended for deterministic adapter smoke/forensics. It does not execute
    Paper decisions and does not reinterpret exchange timestamps as arrival
    times.
    """
    finalizer = DelayedBarFinalizer(contract_schedule=contract_schedule, calendar=calendar,
                                    policy=policy, finalization_ledger=finalization_ledger)
    emitted: list[FinalizedIBKRBar] = []
    by_poll: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_poll.setdefault(int(record["poll_number"]), []).append(record)
    for poll_number in sorted(by_poll):
        poll_records = sorted(by_poll[poll_number], key=lambda item: int(item["observation_sequence"]))
        for record in poll_records:
            row = {"timestamp": int(record["bar_start_epoch_utc"]), **record["ohlcv"]}
            finalizer.observe(row, contract_id=int(record["contract_id"]),
                              local_symbol=str(record["local_symbol"]),
                              observed_at=record["observed_at_utc"], poll_number=poll_number)
        if poll_records:
            poll_time = max(str(record["observed_at_utc"]) for record in poll_records)
            emitted.extend(finalizer.finalize_ready(now=poll_time))
    return finalizer, emitted
