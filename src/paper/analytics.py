"""Deterministic reports from durable realtime Paper facts."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import json
import math
import sqlite3
from typing import Any, Mapping
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote


STRATEGIES = ("MRL1", "MRS2", "S2R", "ORB")


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def _drawdown(values: list[float]) -> tuple[float, float]:
    peak = values[0] if values else 0.0
    max_dd = 0.0
    current = 0.0
    for value in values:
        peak = max(peak, value)
        current = value - peak
        max_dd = min(max_dd, current)
    return max_dd, current


def _trade_metrics(trades: list[Mapping[str, Any]]) -> dict[str, Any]:
    net = [float(t["net_pnl"]) for t in trades if t.get("net_pnl") is not None]
    gross = [float(t["gross_pnl"]) for t in trades if t.get("gross_pnl") is not None]
    rs = [float(t["realized_r"]) for t in trades if t.get("realized_r") is not None]
    winners = [x for x in net if x > 0]
    losers = [x for x in net if x < 0]
    equity = []
    running = 0.0
    for value in net:
        running += value
        equity.append(running)
    max_dd, current_dd = _drawdown(equity)
    durations = [float(t["duration_seconds"]) for t in trades if t.get("duration_seconds") is not None]
    maes = [float(t["mae_points"]) for t in trades if t.get("mae_points") is not None]
    mfes = [float(t["mfe_points"]) for t in trades if t.get("mfe_points") is not None]
    mean_r = sum(rs) / len(rs) if rs else None
    stdev_r = (sum((x - mean_r) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5 if len(rs) > 1 else None
    downside = [min(0.0, x) for x in rs]
    downside_dev = (sum(x*x for x in downside) / len(downside)) ** 0.5 if downside else None
    run_max = run = 0
    run_sign = 0
    for value in net:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        run = run + 1 if sign and sign == run_sign else (1 if sign else 0)
        run_sign = sign
        run_max = max(run_max, run)
    wins_run = losses_run = wins_max = losses_max = 0
    for value in net:
        if value > 0:
            wins_run += 1; losses_run = 0
        elif value < 0:
            losses_run += 1; wins_run = 0
        else:
            wins_run = losses_run = 0
        wins_max = max(wins_max, wins_run); losses_max = max(losses_max, losses_run)
    return {
        "trades": len(trades), "gross_pnl": sum(gross),
        "total_costs": sum(float(t["total_costs"]) for t in trades if t.get("total_costs") is not None),
        "net_pnl": sum(net), "cumulative_r": sum(rs),
        "win_rate": len(winners) / len(net) if net else None,
        "profit_factor": _ratio(sum(winners), abs(sum(losers))),
        "expectancy_usd": sum(net) / len(net) if net else None,
        "expectancy_r": mean_r,
        "average_winner": sum(winners) / len(winners) if winners else None,
        "median_winner": sorted(winners)[len(winners)//2] if winners else None,
        "average_loser": sum(losers) / len(losers) if losers else None,
        "median_loser": sorted(losers)[len(losers)//2] if losers else None,
        "average_win_loss_ratio": _ratio(sum(winners)/len(winners), abs(sum(losers)/len(losers))) if winners and losers else None,
        "sharpe_per_trade_unannualized": mean_r/stdev_r if mean_r is not None and stdev_r else None,
        "sortino_per_trade_unannualized": mean_r/downside_dev if mean_r is not None and downside_dev else None,
        "max_drawdown_usd": max_dd, "current_drawdown_usd": current_dd,
        "max_consecutive_wins": wins_max, "max_consecutive_losses": losses_max,
        "duration_seconds": {"mean": sum(durations)/len(durations) if durations else None,
                              "median": sorted(durations)[len(durations)//2] if durations else None},
        "mae_points": _distribution(maes), "mfe_points": _distribution(mfes),
        "r_distribution": _distribution(rs),
        "quantity_distribution": _distribution([int(t["quantity"]) for t in trades if t.get("quantity") is not None]),
        "sampling_convention": "Per-trade closed net R; unannualized Sharpe and Sortino; downside target R=0.",
        "max_consecutive_any_sign": run_max,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None}
    ordered = sorted(values)
    def q(p: float) -> float:
        index = (len(ordered)-1)*p
        lower = math.floor(index); upper = math.ceil(index)
        return ordered[lower] + (ordered[upper]-ordered[lower])*(index-lower)
    return {"count": len(values), "mean": sum(values)/len(values),
            "median": q(.5), "p05": q(.05), "p25": q(.25), "p75": q(.75), "p95": q(.95)}


def _breakdown(trades: list[Mapping[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for trade in trades:
        key = key_fn(trade)
        groups.setdefault(str(key if key is not None else "UNAVAILABLE"), []).append(trade)
    return {key: {"trades": len(rows), "net_pnl": sum(float(t.get("net_pnl") or 0) for t in rows),
                  "expectancy_r": _ratio(sum(float(t.get("realized_r") or 0) for t in rows), len([t for t in rows if t.get("realized_r") is not None]))}
            for key, rows in sorted(groups.items())}


def build_daily_report(store: Any, day: date) -> dict[str, Any]:
    local = ZoneInfo("America/New_York")
    start = datetime.combine(day, time.min, local).astimezone(timezone.utc).isoformat()
    end = datetime.combine(day + timedelta(days=1), time.min, local).astimezone(timezone.utc).isoformat()
    trades = store.closed_trades(start=start, end=end)
    by_strategy = {name: _trade_metrics([t for t in trades if t["strategy"] == name]) for name in STRATEGIES}
    all_events = [dict(row) for row in store._db.execute("SELECT * FROM events WHERE timestamp_utc>=? AND timestamp_utc<? ORDER BY sequence", (start,end))]
    candidate_rows = [json.loads(row["payload_json"]) for row in store._db.execute("SELECT payload_json FROM candidates WHERE timestamp_utc>=? AND timestamp_utc<?", (start,end))]
    rejected: dict[str, Counter] = {name: Counter() for name in STRATEGIES}
    accepted = Counter()
    for row in candidate_rows:
        strategy = str(row.get("strategy_name", row.get("strategy", "UNKNOWN")))
        status = row.get("status")
        if status == "REJECTED": rejected.setdefault(strategy, Counter())[str(row.get("reason", "unspecified"))] += 1
        else: accepted[strategy] += 1
    risk_rows = list(store._db.execute("SELECT strategy,approved,reason FROM risk_decisions WHERE timestamp_utc>=? AND timestamp_utc<?", (start,end)))
    latest = store._db.execute("SELECT * FROM account_snapshots WHERE timestamp_utc<? ORDER BY timestamp_utc DESC LIMIT 1", (end,)).fetchone()
    health_types = Counter(row["event_type"] for row in all_events)
    hmm_rows = [dict(row) for row in store._db.execute(
        "SELECT stream,raw_state,timestamp_utc FROM hmm_observations WHERE timestamp_utc>=? AND timestamp_utc<? ORDER BY stream,timestamp_utc",
        (start, end),
    )]
    hmm_by_stream: dict[str, dict[str, Any]] = {}
    for stream in {str(row.get("stream") or "unknown") for row in hmm_rows}:
        rows = [row for row in hmm_rows if str(row.get("stream") or "unknown") == stream]
        occupancy = Counter(str(row["raw_state"]) for row in rows if row.get("raw_state") is not None)
        transitions = Counter(
            f"{before['raw_state']}->{after['raw_state']}"
            for before, after in zip(rows, rows[1:])
            if before.get("raw_state") is not None and after.get("raw_state") is not None
        )
        hmm_by_stream[stream] = {
            "observations": len(rows), "state_occupancy": dict(occupancy),
            "transitions": dict(transitions),
        }
    refit_rows = [dict(row) for row in store._db.execute(
        "SELECT stream,event_type,model_version,model_hash,training_start,training_end,fit_duration,payload_json FROM model_events WHERE timestamp_utc>=? AND timestamp_utc<? ORDER BY timestamp_utc",
        (start, end),
    )]
    technical = "RED" if health_types.get("system_critical") or health_types.get("system_error") else "GREEN"
    operational = "RED" if health_types.get("backfill_failed") or health_types.get("feed_disconnected") else "YELLOW" if health_types.get("feed_stale") or health_types.get("checkpoint_failed") else "GREEN"
    statistical = "RED" if health_types.get("parity_warning") else "YELLOW" if len(trades) < 25 else "GREEN"
    if len(trades) < 25:
        statistical = "YELLOW"
    return {
        "date": day.isoformat(), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": {"technical": technical, "operational": operational, "statistical": statistical,
                   "rules": {"technical_red": "Any system_critical or system_error event.",
                             "operational_red": "Any failed backfill or disconnect event; yellow for stale feed/checkpoint warning.",
                             "statistical_red": "Any deterministic shadow parity discrepancy.",
                             "statistical_yellow": "Fewer than 25 closed trades; P&L alone never causes red."}},
        "portfolio": _trade_metrics(trades),
        "strategies": by_strategy,
        "candidate_accounting": {name:{"candidates":sum(1 for x in candidate_rows if x.get("strategy_name",x.get("strategy"))==name),
                                       "accepted":accepted[name],"rejected":sum(rejected[name].values()),
                                       "rejection_reasons":dict(rejected[name])} for name in STRATEGIES},
        "risk_decisions": {"approved":sum(int(r["approved"]) for r in risk_rows),
                           "rejected":sum(not int(r["approved"]) for r in risk_rows),
                           "reasons":dict(Counter(str(r["reason"] or "unspecified") for r in risk_rows if not int(r["approved"])))} ,
        "performance_by": {
            "weekday_ny": _breakdown(trades, lambda t: datetime.fromisoformat(t["entry_timestamp_utc"]).astimezone(local).strftime("%A") if t.get("entry_timestamp_utc") else None),
            "entry_hour_ny": _breakdown(trades, lambda t: datetime.fromisoformat(t["entry_timestamp_utc"]).astimezone(local).strftime("%H:00 ET") if t.get("entry_timestamp_utc") else None),
            "entry_session": _breakdown(trades, lambda t: "RTH" if t.get("entry_timestamp_utc") and time(9,30) <= datetime.fromisoformat(t["entry_timestamp_utc"]).astimezone(local).time().replace(tzinfo=None) < time(16,0) else "ETH" if t.get("entry_timestamp_utc") else None),
            "week": _breakdown(trades, lambda t: f"{datetime.fromisoformat(t['entry_timestamp_utc']).astimezone(local).isocalendar().year}-W{datetime.fromisoformat(t['entry_timestamp_utc']).astimezone(local).isocalendar().week:02d}" if t.get("entry_timestamp_utc") else None),
            "month": _breakdown(trades, lambda t: datetime.fromisoformat(t["entry_timestamp_utc"]).astimezone(local).strftime("%Y-%m") if t.get("entry_timestamp_utc") else None),
            "hmm_raw_state": _breakdown(trades, lambda t: t.get("hmm_state")),
        },
        "rolling_closed_trade_summaries": {str(n): _trade_metrics(trades[-n:]) for n in (25,50,100)},
        "account_snapshot": dict(latest) if latest else None,
        "system_events": dict(health_types),
        "hmm": {"streams": hmm_by_stream, "refit_events": refit_rows},
        "historical_baseline_comparison": {"status":"UNAVAILABLE", "reason":"No frozen baseline mapping was configured for this realtime report."},
    }


class PaperAnalyticsReader:
    """Stable read-only query interface for future dashboards."""
    def __init__(self, path: str) -> None:
        uri_path = quote(str(Path(path).resolve()).replace("\\", "/"), safe="/:\\")
        self._db = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row

    def close(self) -> None:
        self._db.close()

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        row = self._db.execute(sql, params).fetchone()
        return dict(row) if row else None

    def status(self) -> dict[str, Any]:
        return self._one("SELECT * FROM account_snapshots ORDER BY timestamp_utc DESC LIMIT 1") or {"status":"UNAVAILABLE"}

    def positions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY entry_timestamp_utc")]

    def strategy_statistics(self, strategy: str | None = None) -> dict[str, Any]:
        # Reuse the same metric definitions against read-only rows.
        sql = "SELECT * FROM trades WHERE status='CLOSED'" + (" AND strategy=?" if strategy else "") + " ORDER BY exit_timestamp_utc"
        rows = [dict(r) for r in self._db.execute(sql, (strategy,) if strategy else ())]
        names = [strategy] if strategy else list(STRATEGIES)
        return {name: _trade_metrics([r for r in rows if r["strategy"]==name]) for name in names}

    def portfolio_statistics(self) -> dict[str, Any]:
        rows = [dict(r) for r in self._db.execute(
            "SELECT * FROM trades WHERE status='CLOSED' ORDER BY exit_timestamp_utc"
        )]
        return _trade_metrics(rows)

    def performance_series(self, granularity: str = "day", strategy: str | None = None) -> dict[str, Any]:
        if granularity not in {"day", "week", "month"}:
            raise ValueError("granularity must be day, week, or month")
        clauses = ["status='CLOSED'"]
        params: tuple[Any, ...] = ()
        if strategy is not None:
            clauses.append("strategy=?")
            params = (strategy,)
        rows = [dict(r) for r in self._db.execute(
            "SELECT * FROM trades WHERE " + " AND ".join(clauses) + " ORDER BY exit_timestamp_utc",
            params,
        )]
        local = ZoneInfo("America/New_York")
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not row.get("exit_timestamp_utc"):
                continue
            stamp = datetime.fromisoformat(row["exit_timestamp_utc"]).astimezone(local)
            if granularity == "day":
                key = stamp.date().isoformat()
            elif granularity == "week":
                iso = stamp.isocalendar()
                key = f"{iso.year}-W{iso.week:02d}"
            else:
                key = stamp.strftime("%Y-%m")
            groups.setdefault(key, []).append(row)
        return {key: _trade_metrics(items) for key, items in groups.items()}

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM trades ORDER BY entry_timestamp_utc DESC LIMIT ?", (limit,))]

    def trade_detail(self, trade_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM trades WHERE trade_id=?", (trade_id,))

    def hmm_state(self) -> dict[str, Any] | None:
        return self._one("SELECT * FROM hmm_observations ORDER BY timestamp_utc DESC LIMIT 1")

    def risk_status(self) -> dict[str, Any] | None:
        return self._one("SELECT open_risk,gross_exposure,timestamp_utc FROM account_snapshots ORDER BY timestamp_utc DESC LIMIT 1")

    def event_timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT event_id,sequence,event_type,timestamp_utc,strategy,severity,payload_json FROM events ORDER BY sequence DESC LIMIT ?", (limit,))]

    def daily_reports(self, limit: int = 30) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT report_date,generated_at_utc,json_path,markdown_path,report_json FROM daily_reports ORDER BY report_date DESC LIMIT ?", (limit,))]

    def shadow_parity(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM shadow_comparisons ORDER BY timestamp_utc DESC")]

    def model_refits(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute(
            "SELECT * FROM model_events ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
        )]

    def latest_checkpoint(self) -> dict[str, Any] | None:
        return self._one("SELECT * FROM checkpoint_events ORDER BY timestamp_utc DESC LIMIT 1")

    def feed_status(self) -> dict[str, Any]:
        bar = self._one("SELECT timestamp_utc,symbol,contract,source,feed_timestamp_utc FROM market_bars ORDER BY timestamp_utc DESC LIMIT 1")
        event = self._one(
            "SELECT event_type,timestamp_utc,severity,payload_json FROM events "
            "WHERE event_type LIKE 'FEED_%' ORDER BY sequence DESC LIMIT 1"
        )
        return {"last_bar": bar, "last_feed_event": event}

    def candidate_diagnostics(self, limit: int = 100) -> dict[str, Any]:
        totals = [dict(r) for r in self._db.execute(
            "SELECT strategy,status,reason,COUNT(*) AS count FROM candidates "
            "GROUP BY strategy,status,reason ORDER BY strategy,status,reason"
        )]
        recent = [dict(r) for r in self._db.execute(
            "SELECT event_id,timestamp_utc,strategy,status,reason,payload_json FROM candidates "
            "ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
        )]
        return {"totals": totals, "recent": recent}
