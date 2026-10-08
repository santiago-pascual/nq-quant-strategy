from datetime import date, datetime, timezone
import json
import sqlite3
import pytest

from src.paper.analytics import PaperAnalyticsReader
from src.paper.analytics_db import PaperAnalyticsStore
from src.paper.costs import PaperCostPolicy, calculate_trade_economics
from src.paper.logger import PaperEvent, PaperEventType


TOPSTEP_COSTS = "src/paper/config/topstepx_mnq_fees_2026-07.json"


def _event(sequence, kind, minute, payload):
    return PaperEvent(
        event_id=f"evt-{sequence}", sequence=sequence, event_type=kind,
        timestamp=datetime(2024, 1, 2, 14, 30 + minute, tzinfo=timezone.utc),
        payload=payload,
    )


def test_topstep_cost_policy_long_short_multiple_contract_winner_loser():
    policy = PaperCostPolicy.from_json(TOPSTEP_COSTS)
    assert policy.round_turn_per_contract == 1.22
    long = calculate_trade_economics(
        direction="long", entry_price=100, exit_price=102, quantity=2,
        point_value=2, initial_risk_usd=4, policy=policy,
    )
    assert long["gross_pnl"] == 8
    assert long["commission"] == 1.0
    assert long["exchange_fees"] == 1.4
    assert long["regulatory_fees"] == 0.04
    assert long["total_costs"] == 2.44
    assert long["net_pnl"] == pytest.approx(5.56)
    assert long["realized_r"] == pytest.approx(1.39)
    short = calculate_trade_economics(
        direction="short", entry_price=101, exit_price=102, quantity=3,
        point_value=2, initial_risk_usd=6, policy=policy,
    )
    assert short["gross_pnl"] == -6
    assert short["total_costs"] == pytest.approx(3.66)
    assert short["net_pnl"] == pytest.approx(-9.66)
    assert short["realized_r"] == pytest.approx(-1.61)
    assert policy.artificial_slippage_ticks == 0


def test_sqlite_migration_idempotency_trade_recalculation_and_daily_report(tmp_path):
    policy = PaperCostPolicy.from_json(TOPSTEP_COSTS)
    store = PaperAnalyticsStore(tmp_path / "paper.sqlite3", point_value=2, cost_policy=policy)
    store.record_configuration(config_kind="fees", version=policy.profile_id,
                               identity=policy.identity, source=policy.__dict__)
    events = [
        _event(1, PaperEventType.MARKET_DATA, 0, {"symbol":"MNQ","provider":"fixture","timestamp":"2024-01-02T14:30:00+00:00","open":100,"high":101,"low":99,"close":100,"volume":10,"hmm_state":1,"hmm_posterior":[.1,.8,.1]}),
        _event(2, PaperEventType.RISK_REQUEST, 0, {"strategy_name":"MRL1","entry_price":100,"stop_price":99,"risk_per_contract":2}),
        _event(3, PaperEventType.CANDIDATE_CREATED, 0, {"strategy_name":"MRL1","signal":"long"}),
        _event(4, PaperEventType.RISK_DECISION, 0, {"strategy_name":"MRL1","approved":True,"quantity":2,"risk_per_contract":2,"total_risk":4}),
        _event(5, PaperEventType.ORDER_CREATED, 0, {"strategy_name":"MRL1","order_id":"ord-1","quantity":2,"price":100}),
        _event(6, PaperEventType.FILL, 0, {"strategy_name":"MRL1","fill_id":"fill-in","broker_order_id":"ord-1","quantity":2,"price":100,"signal":"long"}),
        _event(7, PaperEventType.POSITION_OPENED, 0, {"strategy_name":"MRL1","quantity":2,"entry_price":100,"side":"long"}),
        _event(8, PaperEventType.MARKET_DATA, 1, {"symbol":"MNQ","timestamp":"2024-01-02T14:31:00+00:00","open":100,"high":103,"low":99,"close":102,"volume":11}),
        _event(9, PaperEventType.STRATEGY_DECISION, 2, {"strategy_name":"MRL1","action":"exit","reason":"target"}),
        _event(10, PaperEventType.FILL, 2, {"strategy_name":"MRL1","fill_id":"fill-out","broker_order_id":"ord-2","quantity":2,"price":102,"signal":"flat"}),
        _event(11, PaperEventType.POSITION_CLOSED, 2, {"strategy_name":"MRL1","quantity":2,"entry_price":100,"exit_price":102,"side":"long","initial_risk_usd":4,"gross_pnl":8,"commission":1,"exchange_fees":1.4,"regulatory_fees":.04,"total_costs":2.44,"net_pnl":5.56,"realized_r":1.39,"account_balance_after":50005.56}),
        _event(12, PaperEventType.HMM_STATE, 3, {"stream":"MR","raw_state":1,"posterior":[.1,.8,.1],"model_version":"mr-1","model_hash":"abc"}),
        _event(13, PaperEventType.HMM_STATE, 4, {"stream":"MR","raw_state":2,"posterior":[.1,.1,.8],"model_version":"mr-1","model_hash":"abc"}),
    ]
    for event in events:
        assert store.ingest_event(event)
    assert not store.ingest_event(events[0])
    assert store.integrity_check() == "ok"
    assert store._db.execute("PRAGMA user_version").fetchone()[0] == 1
    trade = store.closed_trades()[0]
    assert trade["gross_pnl"] == 8
    assert trade["total_costs"] == 2.44
    assert trade["net_pnl"] == 5.56
    assert trade["realized_r"] == 1.39
    assert trade["mae_points"] == 1
    assert trade["mfe_points"] == 3
    assert trade["hmm_state"] == 1
    assert store._db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(events)
    assert store._db.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 2
    report = store.write_daily_report(date(2024, 1, 2), tmp_path)
    assert report["portfolio"]["trades"] == 1
    assert report["strategies"]["MRL1"]["net_pnl"] == 5.56
    assert report["strategies"]["MRS2"]["trades"] == 0
    assert report["strategies"]["MRS2"]["profit_factor"] is None
    assert report["hmm"]["streams"]["MR"]["state_occupancy"] == {"1": 1, "2": 1}
    assert report["hmm"]["streams"]["MR"]["transitions"] == {"1->2": 1}
    assert (tmp_path / "daily_reports" / "2024-01-02.json").exists()
    assert (tmp_path / "daily_reports" / "2024-01-02.md").exists()
    store.close()
    reader = PaperAnalyticsReader(str(tmp_path / "paper.sqlite3"))
    assert reader.strategy_statistics("MRL1")["MRL1"]["trades"] == 1
    assert reader.portfolio_statistics()["trades"] == 1
    assert reader.performance_series("day")["2024-01-02"]["trades"] == 1
    assert reader.performance_series("week")["2024-W01"]["trades"] == 1
    assert reader.performance_series("month")["2024-01"]["trades"] == 1
    assert reader.trade_detail(trade["trade_id"])["net_pnl"] == 5.56
    assert reader.positions() == []
    reader.close()


def test_sqlite_backup_is_consistent_and_independent(tmp_path):
    store = PaperAnalyticsStore(tmp_path / "paper.sqlite3")
    backup = store.backup_to(tmp_path / "backup" / "paper.sqlite3")
    assert backup.exists()
    assert store.integrity_check() == "ok"
    copied = PaperAnalyticsStore(backup)
    assert copied.integrity_check() == "ok"
    assert copied._db.execute("PRAGMA user_version").fetchone()[0] == 1
    copied.close()
    store.close()


def test_database_restart_replay_is_idempotent_and_corruption_fails_loudly(tmp_path):
    path = tmp_path / "paper.sqlite3"
    event = _event(1, PaperEventType.SYSTEM_WARNING, 0, {"message": "fixture"})
    first = PaperAnalyticsStore(path)
    assert first.ingest_event(event)
    first.close()
    restarted = PaperAnalyticsStore(path)
    assert not restarted.ingest_event(event)
    assert restarted._db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    restarted.close()

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        PaperAnalyticsStore(corrupt)
