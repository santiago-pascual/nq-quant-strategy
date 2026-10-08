"""Deterministic shadow replay over persisted canonical Paper bars."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.paper.logger import PaperEventType
from src.paper.realtime_checkpoint import AtomicCheckpointStore, restore_engine_state, runtime_identity
from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar
from src.paper.realtime_market_data import CanonicalBar, ReplayMarketDataSource
from src.paper.realtime_service import RealtimePaperConfig, RealtimePaperService
from src.paper.costs import PaperCostPolicy
from src.paper.run_autonomous import build_real_paper_engine


DETERMINISTIC_EVENT_TYPES = {
    PaperEventType.MARKET_DATA.value,
    PaperEventType.HMM_STATE.value,
    PaperEventType.STRATEGY_DECISION.value,
    PaperEventType.RISK_REQUEST.value,
    PaperEventType.RISK_DECISION.value,
    PaperEventType.ORDER_CREATED.value,
    PaperEventType.ORDER_SUBMITTED.value,
    PaperEventType.ORDER_FILLED.value,
    PaperEventType.FILL.value,
    PaperEventType.POSITION_OPENED.value,
    PaperEventType.POSITION_UPDATED.value,
    PaperEventType.POSITION_CLOSED.value,
    PaperEventType.RISK_REJECTION.value,
    PaperEventType.CANDIDATE_CREATED.value,
    PaperEventType.CANDIDATE_REJECTED.value,
    PaperEventType.PAPER_ORDER_CREATED.value,
    PaperEventType.PAPER_ORDER_FILLED.value,
    PaperEventType.PAPER_POSITION_OPENED.value,
    PaperEventType.PAPER_POSITION_CLOSED.value,
}


def _read_events(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed event log at line {line_number}") from exc
    return result


def _canonical_bars(events: list[dict[str, Any]]) -> list[CanonicalBar]:
    bars: list[CanonicalBar] = []
    for event in events:
        if event.get("event_type") != PaperEventType.MARKET_DATA.value:
            continue
        payload = event["payload"]
        bars.append(CanonicalBar(
            symbol=str(payload.get("symbol", "MNQ")),
            timestamp=pd.Timestamp(payload["timestamp"]).to_pydatetime(),
            open=float(payload["open"]), high=float(payload["high"]),
            low=float(payload["low"]), close=float(payload["close"]),
            volume=float(payload["volume"]),
            provider=payload.get("provider"), source_id=payload.get("source_id"),
            is_session_final=payload.get("is_session_final"),
            contract_symbol=payload.get("contract_symbol"),bid=payload.get("bid"),ask=payload.get("ask"),
            provider_timestamp=pd.Timestamp(payload["provider_timestamp"]).to_pydatetime() if payload.get("provider_timestamp") else None,
        ))
    return bars


def _normalized(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for event in events:
        if event.get("event_type") not in DETERMINISTIC_EVENT_TYPES:
            continue
        payload = dict(event.get("payload", {}))
        for key in ("_idempotency_key", "run_id", "severity"):
            payload.pop(key, None)
        normalized.append({
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
            "payload": payload,
        })
    return normalized


def run_shadow_replay(
    source_dir: str | Path,
    *,
    initial_equity: float = 50_000.0,
    commission_per_contract: float = 0.0,
    price_offset: float = 0.0,
) -> dict[str, Any]:
    """Recompute a persisted stream from its pre-stream bootstrap checkpoint.

    The seed contains only causal historical state as of stream start. Replayed
    canonical bars pass through the same context, HMM, strategies, risk, and
    simulated execution path as the original Paper run.
    """
    source_dir = Path(source_dir)
    event_path = source_dir / "events.jsonl"
    seed_path = source_dir / "bootstrap_checkpoint.json"
    if not event_path.exists() or not seed_path.exists():
        raise FileNotFoundError("Shadow replay requires events.jsonl and bootstrap_checkpoint.json")
    seed = AtomicCheckpointStore(seed_path).load()
    cost_file = source_dir / "cost_profile.json"
    cost_policy = PaperCostPolicy.from_json(cost_file) if cost_file.exists() else None
    if seed.get("system", {}).get("cost_profile_identity") != (cost_policy.identity if cost_policy else None):
        raise ValueError("Shadow replay cost profile differs from the original run")
    if seed.get("system", {}).get("runtime_identity") != runtime_identity():
        raise ValueError("Shadow bootstrap code/dependency identity differs from this runtime")
    original_events = _read_events(event_path)
    bars = _canonical_bars(original_events)
    if not bars:
        raise ValueError("No persisted canonical market bars are available for shadow replay")

    shadow_dir = source_dir / "shadow_replay"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    engine, adapter = build_real_paper_engine(
        shadow_dir, initial_equity=initial_equity,
        commission_per_contract=cost_policy.commission_per_contract_side if cost_policy else commission_per_contract,
        exchange_fee_per_contract=cost_policy.exchange_fee_per_contract_side if cost_policy else 0.0,
        regulatory_fee_per_contract=cost_policy.regulatory_fee_per_contract_side if cost_policy else 0.0,
        price_offset=0.0 if cost_policy else price_offset,
    )
    adapter.context.load_state_dict(seed["context"])
    restore_engine_state(engine, seed["engine"])
    source = ReplayMarketDataSource(bars)
    service = RealtimePaperService(
        source=source, engine=engine, context_adapter=adapter,
        config=RealtimePaperConfig(mode="PAPER", output_dir=shadow_dir),
        run_id=str(seed["runtime"]["run_id"]),
        calendar=(
            CMETradingCalendar(CMECalendarSnapshot.from_mapping(seed["system"]["calendar_snapshot"]))
            if seed.get("system", {}).get("calendar_snapshot") else None
        ),
        cost_policy=cost_policy,
    )
    service.run()

    replay_events = _read_events(shadow_dir / "events.jsonl")
    expected = _normalized(original_events)
    actual = _normalized(replay_events)
    differences: list[dict[str, Any]] = []
    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"compared": 0, "differences": 0, "by_event_type": {}}
    )
    for row in expected:
        day = pd.Timestamp(row["timestamp"]).tz_convert("America/New_York").date().isoformat()
        by_day[day]["compared"] += 1
    for index in range(max(len(expected), len(actual))):
        expected_row = expected[index] if index < len(expected) else None
        actual_row = actual[index] if index < len(actual) else None
        if expected_row == actual_row:
            continue
        example = {"index": index, "expected": expected_row, "actual": actual_row}
        if len(differences) < 20:
            differences.append(example)
        reference = expected_row or actual_row
        if reference:
            day = pd.Timestamp(reference["timestamp"]).tz_convert("America/New_York").date().isoformat()
            daily = by_day[day]
            daily["differences"] += 1
            event_type = reference.get("event_type", "unknown")
            daily["by_event_type"][event_type] = daily["by_event_type"].get(event_type, 0) + 1
    first_difference = differences[0] if differences else None
    report = {
        "status": "GREEN" if not differences and len(expected) == len(actual) else "RED",
        "source_event_log": str(event_path),
        "shadow_event_log": str(shadow_dir / "events.jsonl"),
        "bars_replayed": len(bars),
        "deterministic_events_expected": len(expected),
        "deterministic_events_actual": len(actual),
        "difference_count": sum(day["differences"] for day in by_day.values()),
        "first_difference": first_difference,
        "difference_examples": differences,
        "daily": dict(by_day),
    }
    report_path = source_dir / "daily_shadow_parity.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
