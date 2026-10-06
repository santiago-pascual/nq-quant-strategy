from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.paper.logger import PaperEventType
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG


TERMINAL_OUTCOMES = (
    "ACCEPTED",
    "RISK_REJECTED",
    "CONFLICT_REJECTED",
    "NO_ORDER_SUBMITTED",
    "ORDER_NO_FILL",
    "FILL_NO_POSITION",
    "CLOSED",
    "OPEN_AT_END",
)


def _candidate_key(strategy: str, timestamp: pd.Timestamp) -> str:
    return f"{strategy.upper()}|{timestamp.tz_convert('UTC').isoformat()}"


def build_candidate_ledger(
    events: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build exactly one terminal outcome per Paper ENTER event."""
    candidates: dict[str, dict[str, Any]] = {}
    latest_by_strategy: dict[str, str] = {}
    candidate_by_order: dict[str, str] = {}
    active_by_strategy: dict[str, str] = {}

    for event in events:
        kind = str(event["event_type"])
        timestamp = pd.Timestamp(event["timestamp"])
        if timestamp.tzinfo is None:
            raise ValueError("Candidate events must have timezone-aware timestamps.")
        timestamp = timestamp.tz_convert("UTC")
        payload = event["payload"]
        strategy = str(payload.get("strategy_name", ""))
        if not strategy and isinstance(payload.get("context"), Mapping):
            strategy = str(payload["context"].get("strategy_name", ""))

        if kind == PaperEventType.STRATEGY_DECISION.value and payload.get("action") == "enter":
            key = _candidate_key(strategy, timestamp)
            candidates[key] = {
                "trade_key": key,
                "strategy_name": strategy,
                "candidate_id": {
                    "MRL1": MRL1_CONFIG.candidate_id,
                    "MRS2": MRS2_CONFIG.candidate_id,
                    "S2R": "S2R",
                    "ORB": "ORB",
                }.get(strategy, strategy),
                "entry_timestamp": timestamp,
                "decision_observed": True,
                "decision_reason": payload.get("reason", ""),
                "risk_decision": "",
                "risk_reason": "",
                "risk_quantity": 0,
                "theoretical_quantity": np.nan,
                "executable_quantity": 0,
                "adaptive_quantity": None,
                "risk_per_contract": np.nan,
                "total_risk": np.nan,
                "sizing_rule": "",
                "order_submitted": False,
                "broker_order_id": "",
                "fill_quantity": 0,
                "position_opened": False,
                "position_closed": False,
                "error": "",
                "rejection_reason": "",
            }
            latest_by_strategy[strategy] = key
            continue

        key = latest_by_strategy.get(strategy)
        record = candidates.get(key) if key else None
        if record is None:
            continue
        if kind == PaperEventType.RISK_DECISION.value:
            record.update(
                risk_decision=payload.get("decision", ""),
                risk_reason=payload.get("reason", ""),
                risk_quantity=int(payload.get("quantity", 0)),
                theoretical_quantity=payload.get("theoretical_quantity", np.nan),
                executable_quantity=int(payload.get("executable_quantity", 0)),
                adaptive_quantity=payload.get("adaptive_quantity"),
                risk_per_contract=payload.get("risk_per_contract", np.nan),
                total_risk=payload.get("total_risk", np.nan),
                sizing_rule=payload.get("sizing_rule", ""),
            )
        elif kind == PaperEventType.ERROR.value:
            record["error"] = str(payload.get("message", payload.get("error", "")))
            context = payload.get("context")
            if isinstance(context, Mapping) and context.get("strategy_name"):
                latest_by_strategy[str(context["strategy_name"])] = record["trade_key"]
        elif kind == PaperEventType.ORDER_CREATED.value:
            order_id = str(payload.get("broker_order_id", ""))
            record["broker_order_id"] = order_id
            if order_id:
                candidate_by_order[order_id] = record["trade_key"]
        elif kind == PaperEventType.ORDER_SUBMITTED.value:
            record["order_submitted"] = True
        elif kind in (PaperEventType.FILL.value, PaperEventType.ORDER_PARTIALLY_FILLED.value):
            order_id = str(payload.get("broker_order_id", ""))
            fill_record = candidates.get(candidate_by_order.get(order_id, ""))
            if fill_record is not None:
                fill_record["fill_quantity"] += int(payload.get("quantity", 0))
        elif kind == PaperEventType.ORDER_REJECTED.value:
            record["rejection_reason"] = str(payload.get("reason", payload.get("message", "")))
        elif kind == PaperEventType.ORDER_CANCELLED.value:
            record["rejection_reason"] = str(payload.get("reason", "order cancelled before fill"))
        elif kind == PaperEventType.POSITION_OPENED.value:
            record["position_opened"] = True
            active_by_strategy[strategy] = record["trade_key"]
        elif kind == PaperEventType.POSITION_CLOSED.value:
            active = candidates.get(active_by_strategy.pop(strategy, ""))
            if active is not None:
                active["position_closed"] = True

    rows = []
    for record in candidates.values():
        error = str(record["error"] or "")
        if record["risk_decision"] == "rejected":
            outcome = "RISK_REJECTED"
            reason = str(record["risk_reason"] or "risk engine rejected candidate without a reason")
        elif "portfolio conflict rejected" in error.lower():
            outcome = "CONFLICT_REJECTED"
            reason = error
        elif record["risk_decision"] != "approved":
            outcome = "NO_ORDER_SUBMITTED"
            reason = error or str(record["risk_reason"] or "risk decision was not observed")
        elif not record["order_submitted"]:
            outcome = "NO_ORDER_SUBMITTED"
            reason = error or "approved candidate had no order submission event"
        elif record["fill_quantity"] <= 0:
            outcome = "ORDER_NO_FILL"
            reason = str(record["rejection_reason"] or "submitted order had no fill by replay end")
        elif not record["position_opened"]:
            outcome = "FILL_NO_POSITION"
            reason = "fill observed without a position opened event"
        elif record["position_closed"]:
            outcome = "CLOSED"
            reason = ""
        else:
            outcome = "OPEN_AT_END"
            reason = "position remained open at replay end"

        record["terminal_outcome"] = outcome
        record["status"] = outcome
        accepted = record["risk_decision"] == "approved" and record["order_submitted"] and outcome not in {
            "CONFLICT_REJECTED", "NO_ORDER_SUBMITTED",
        }
        record["candidate_decision"] = "ACCEPTED" if accepted else "REJECTED"
        record["rejection_reason"] = reason if outcome in {
            "RISK_REJECTED", "CONFLICT_REJECTED", "NO_ORDER_SUBMITTED",
            "ORDER_NO_FILL", "FILL_NO_POSITION",
        } else ""
        record["candidate_outcome"] = (
            "ACCEPTED" if accepted else f"REJECTED({record['rejection_reason']})"
        )
        rows.append(record)

    columns = [
        "trade_key", "strategy_name", "candidate_id", "entry_timestamp",
        "decision_observed", "candidate_decision", "candidate_outcome",
        "decision_reason", "risk_decision", "risk_reason", "risk_quantity",
        "theoretical_quantity", "executable_quantity", "adaptive_quantity",
        "risk_per_contract", "total_risk", "sizing_rule", "order_submitted",
        "broker_order_id", "fill_quantity", "position_opened", "position_closed",
        "terminal_outcome", "status", "rejection_reason", "error",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    if not result["terminal_outcome"].isin(TERMINAL_OUTCOMES).all():
        raise RuntimeError("Candidate ledger contains an unknown terminal outcome.")
    if result["trade_key"].duplicated().any():
        raise RuntimeError("Candidate ledger contains duplicate Paper ENTER decisions.")
    return result.sort_values(
        ["entry_timestamp", "strategy_name"], kind="mergesort"
    ).reset_index(drop=True)
