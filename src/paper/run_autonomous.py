from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.paper.autonomous_runner import (
    AutonomousPaperRunner,
    AutonomousRunConfig,
    RESULTS_DIR,
    load_canonical_raw_mnq,
)
from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.candidate_ledger import build_candidate_ledger
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.paper.logger import PaperEventLogger
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk import RiskEngine
from src.risk.policy import XFA_50K_PRODUCTION_POLICY
from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.strategy import S2RStrategy


def build_real_paper_engine(
    output_dir: Path,
    *,
    initial_equity: float = 50_000.0,
    commission_per_contract: float = 0.0,
    exchange_fee_per_contract: float = 0.0,
    regulatory_fee_per_contract: float = 0.0,
    price_offset: float = 0.0,
    recover_trailing_event: bool = False,
) -> tuple[PaperTradingEngine, PaperMarketContextAdapter]:
    context_adapter = PaperMarketContextAdapter()
    strategies = (
        MeanReversionStrategy(MRL1_CONFIG),
        MeanReversionStrategy(MRS2_CONFIG),
        S2RStrategy(),
        ORBStrategy(),
    )
    broker = InMemoryBrokerAdapter()
    engine = PaperTradingEngine(
        strategies=strategies,
        execution=ExecutionEngine(),
        risk=RiskEngine(XFA_50K_PRODUCTION_POLICY.to_risk_limits()),
        conflict=PortfolioConflictEngine(max_concurrent_positions=3),
        broker=broker,
        logger=PaperEventLogger(
            output_dir / "events.jsonl",
            recover_trailing_partial=recover_trailing_event,
        ),
        context_adapter=context_adapter,
        config=PaperEngineConfig(
            initial_equity=initial_equity,
            point_value=2.0,
            tick_size=0.25,
            price_offset=price_offset,
            commission_per_contract=commission_per_contract,
            exchange_fee_per_contract=exchange_fee_per_contract,
            regulatory_fee_per_contract=regulatory_fee_per_contract,
            automatic_simulated_fills=True,
        ),
    )
    return engine, context_adapter


def _event_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_run_artifacts(
    *,
    engine: PaperTradingEngine,
    runner: AutonomousPaperRunner,
    output_dir: Path,
) -> None:
    event_path = output_dir / "events.jsonl"
    signals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    risk_violations: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    current_market: dict[str, Any] = {}
    pending_signal: dict[str, dict[str, Any]] = {}
    entry_fills: dict[str, list[dict[str, Any]]] = {}
    active_trades: dict[str, dict[str, Any]] = {}
    exit_reasons: dict[str, str] = {}

    for event in _event_records(event_path):
        kind = event["event_type"]
        timestamp = event["timestamp"]
        payload = event["payload"]
        if kind == "market_data":
            current_market = payload
            continue
        if kind == "strategy_decision":
            strategy = payload.get("strategy_name", "")
            action = payload.get("action", "")
            if action in {"enter", "exit"}:
                signals.append(
                    {
                        "timestamp": timestamp,
                        "strategy_name": strategy,
                        "action": action,
                        "signal": payload.get("signal"),
                        "reason": payload.get("reason"),
                        "hmm_state": current_market.get("hmm_state"),
                        "hmm_window": current_market.get("hmm_window"),
                    }
                )
            if action == "enter":
                pending_signal[strategy] = {
                    "signal_timestamp": timestamp,
                    "signal": payload.get("signal"),
                    "signal_reason": payload.get("reason"),
                }
            elif action == "exit":
                exit_reasons[strategy] = str(payload.get("reason", ""))
            continue
        if kind == "order_created":
            orders.append({"timestamp": timestamp, **payload})
            continue
        if kind == "order_rejected":
            rejections.append({"timestamp": timestamp, **payload})
            continue
        if kind == "risk_decision" and not payload.get("approved", False):
            risk_violations.append({"timestamp": timestamp, **payload})
            continue
        if kind == "error":
            risk_violations.append(
                {
                    "timestamp": timestamp,
                    "reason": payload.get("error", payload.get("message")),
                    **payload,
                }
            )
            continue
        if kind == "fill":
            strategy = payload.get("strategy_name", "")
            row = {
                "timestamp": timestamp,
                **payload,
                "market_open": current_market.get("open"),
                "hmm_state": current_market.get("hmm_state"),
                "hmm_window": current_market.get("hmm_window"),
            }
            fills.append(row)
            if payload.get("signal") in {"long", "short"}:
                entry_fills.setdefault(strategy, []).append(row)
            elif strategy in active_trades:
                active_trades[strategy]["commission"] += (
                    int(payload.get("quantity", 0))
                    * engine.config.commission_per_contract
                )
            continue
        if kind == "position_opened":
            strategy = payload["strategy_name"]
            strategy_entry_fills = entry_fills.pop(strategy, [])
            entry_fill = strategy_entry_fills[-1] if strategy_entry_fills else {}
            active_trades[strategy] = {
                **payload,
                **pending_signal.pop(strategy, {}),
                "entry_timestamp": timestamp,
                "entry_execution_timestamp": entry_fill.get("timestamp", timestamp),
                "hmm_state": entry_fill.get("hmm_state"),
                "hmm_window": entry_fill.get("hmm_window"),
                "commission": sum(
                    int(fill.get("quantity", 0))
                    * engine.config.commission_per_contract
                    for fill in strategy_entry_fills
                ),
            }
            continue
        if kind == "position_closed":
            strategy = payload["strategy_name"]
            opened = active_trades.pop(strategy, {})
            quantity = int(payload.get("quantity", opened.get("quantity", 0)))
            entry_price = float(payload["entry_price"])
            exit_price = float(payload["exit_price"])
            side = str(payload.get("side", opened.get("side", "")))
            signed_points = (
                exit_price - entry_price
                if side.lower() == "long"
                else entry_price - exit_price
            )
            gross_pnl = signed_points * quantity * engine.config.point_value
            commission = float(opened.get("commission", 0.0))
            trades.append(
                {
                    "strategy_name": strategy,
                    "direction": side,
                    "signal_timestamp": opened.get("signal_timestamp"),
                    "signal": opened.get("signal"),
                    "entry_timestamp": opened.get("entry_timestamp"),
                    "entry_execution_timestamp": opened.get(
                        "entry_execution_timestamp"
                    ),
                    "entry_price": entry_price,
                    "exit_timestamp": timestamp,
                    "exit_execution_timestamp": timestamp,
                    "exit_price": exit_price,
                    "exit_reason": exit_reasons.pop(strategy, None),
                    "quantity": quantity,
                    "hmm_state": opened.get("hmm_state"),
                    "hmm_window": opened.get("hmm_window"),
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "net_pnl": gross_pnl - commission,
                }
            )

    daily_equity = pd.DataFrame(runner.daily_equity)
    daily_equity.to_csv(output_dir / "daily_equity.csv", index=False)
    pd.DataFrame(trades).to_csv(output_dir / "trade_ledger.csv", index=False)
    candidate_ledger = build_candidate_ledger(_event_records(event_path))
    candidate_ledger.to_csv(output_dir / "candidate_ledger.csv", index=False)
    pd.DataFrame(fills).to_csv(output_dir / "fills.csv", index=False)
    execution_costs = [
        {
            "timestamp": fill.get("timestamp"),
            "strategy_name": fill.get("strategy_name"),
            "fill_id": fill.get("fill_id"),
            "quantity": fill.get("quantity"),
            "configured_price_offset_points": engine.config.price_offset,
            "modeled_price_offset_cost": (
                abs(engine.config.price_offset)
                * int(fill.get("quantity", 0))
                * engine.config.point_value
            ),
            "commission": (
                int(fill.get("quantity", 0))
                * engine.config.commission_per_contract
            ),
        }
        for fill in fills
    ]
    pd.DataFrame(execution_costs).to_csv(
        output_dir / "execution_costs.csv", index=False
    )
    pd.DataFrame(orders).to_csv(output_dir / "orders.csv", index=False)
    pd.DataFrame(signals).to_csv(output_dir / "signals.csv", index=False)
    pd.DataFrame(rejections).to_csv(output_dir / "rejected_orders.csv", index=False)
    pd.DataFrame(risk_violations).to_csv(
        output_dir / "risk_violations.csv", index=False
    )
    attribution = (
        pd.DataFrame(trades)
        .groupby("strategy_name", dropna=False)
        .agg(
            trades=("strategy_name", "size"),
            gross_pnl=("gross_pnl", "sum"),
            commissions=("commission", "sum"),
            net_pnl=("net_pnl", "sum"),
            wins=("net_pnl", lambda values: int((values > 0).sum())),
        )
        .reset_index()
        if trades
        else pd.DataFrame(
            columns=[
                "strategy_name",
                "trades",
                "gross_pnl",
                "commissions",
                "net_pnl",
                "wins",
            ]
        )
    )
    attribution.to_csv(output_dir / "strategy_attribution.csv", index=False)
    context = runner.context_adapter.context
    fitted_s2_windows = sorted(context.s2_models)
    summary = {
        **runner.stats.as_dict(),
        "strategies": [strategy.name for strategy in engine.strategies],
        "hmm_mode": "causal_online",
        "hmm_fitted": bool(context._hmm is not None),
        "s2r_fitted_windows": fitted_s2_windows,
        "orders": len(orders),
        "fills": len(fills),
        "closed_trades": len(trades),
        "candidate_outcome_counts": {
            str(outcome): int(count)
            for outcome, count in candidate_ledger["terminal_outcome"]
            .value_counts()
            .items()
        },
        "open_positions": len(engine.execution.get_positions()),
        "gross_realized_pnl": engine.realized_pnl,
        "commissions": engine.commissions,
        "net_realized_pnl": engine.realized_pnl - engine.commissions,
        "ending_equity": engine.account_equity,
        "price_offset_points": engine.config.price_offset,
        "modeled_price_offset_cost": sum(
            row["modeled_price_offset_cost"] for row in execution_costs
        ),
        "commission_per_contract": engine.config.commission_per_contract,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def _smoke_replay(
    raw: pd.DataFrame,
    *,
    smoke_date: str,
    engine: PaperTradingEngine,
    context_adapter: PaperMarketContextAdapter,
    output_dir: Path,
) -> AutonomousPaperRunner:
    day = pd.Timestamp(smoke_date)
    local_start = day.tz_localize("America/New_York") + pd.Timedelta(
        hours=9, minutes=30
    )
    local_end = day.tz_localize("America/New_York") + pd.Timedelta(
        hours=15, minutes=59
    )
    start = local_start.tz_convert("UTC")
    end = local_end.tz_convert("UTC")

    history = raw.loc[raw["timestamp"] < start]
    if history.empty:
        raise ValueError("Smoke session has no prior canonical warmup history.")
    for _, row in history.iterrows():
        context_adapter.update(row.to_dict())

    session = raw.loc[
        (raw["timestamp"] >= start) & (raw["timestamp"] <= end)
    ].copy()
    if session.empty:
        raise ValueError(f"No canonical MNQ bars found for smoke date {smoke_date}.")
    runner = AutonomousPaperRunner(
        paper_engine=engine,
        context_adapter=context_adapter,
        config=AutonomousRunConfig(
            output_dir=output_dir,
            start_timestamp=start,
            end_timestamp=end,
            progress_every_bars=100,
        ),
    )
    runner.run(session)
    if context_adapter.context.bars_seen <= len(history):
        raise RuntimeError("Smoke replay did not advance the real market context.")
    if context_adapter.context._hmm is None:
        raise RuntimeError("Smoke replay did not initialize the causal online HMM.")
    if engine.execution.get_positions():
        raise RuntimeError("Smoke replay ended with open positions.")
    return runner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal autonomous MNQ paper engine."
    )
    parser.add_argument("--start", help="UTC-aware start timestamp or ISO date.")
    parser.add_argument("--end", help="UTC-aware end timestamp or ISO date.")
    parser.add_argument(
        "--smoke-date",
        help="Run one real New York session after causal history warmup.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--initial-equity", type=float, default=50_000.0)
    parser.add_argument("--commission-per-contract", type=float, default=0.0)
    parser.add_argument("--price-offset", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output_dir = args.output_dir / (
        f"smoke-{args.smoke_date}-{run_id}" if args.smoke_date else f"run-{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    raw = load_canonical_raw_mnq()
    engine, context_adapter = build_real_paper_engine(
        output_dir,
        initial_equity=args.initial_equity,
        commission_per_contract=args.commission_per_contract,
        price_offset=args.price_offset,
    )
    engine.connect()
    try:
        if args.smoke_date:
            runner = _smoke_replay(
                raw,
                smoke_date=args.smoke_date,
                engine=engine,
                context_adapter=context_adapter,
                output_dir=output_dir,
            )
        else:
            config = AutonomousRunConfig(
                output_dir=output_dir,
                start_timestamp=(
                    pd.Timestamp(args.start) if args.start is not None else None
                ),
                end_timestamp=(
                    pd.Timestamp(args.end) if args.end is not None else None
                ),
            )
            runner = AutonomousPaperRunner(
                paper_engine=engine,
                context_adapter=context_adapter,
                config=config,
            )
            runner.run(raw)
        write_run_artifacts(
            engine=engine,
            runner=runner,
            output_dir=output_dir,
        )
    finally:
        engine.disconnect()
    print(f"Autonomous paper outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
