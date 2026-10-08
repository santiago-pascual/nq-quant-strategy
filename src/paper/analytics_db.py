"""Durable, local analytics facts for realtime Paper (SQLite/WAL)."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import sqlite3
from threading import RLock
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from src.paper.logger import PaperEvent, PaperEventType


SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=True)


class PaperAnalyticsStore:
    """Append-only event mirror plus normalized Paper analytics facts."""

    def __init__(self, path: str | Path, *, point_value: float = 2.0,
                 cost_policy: Any | None = None, calendar: Any | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.point_value = float(point_value)
        self.cost_policy = cost_policy
        self.calendar = calendar
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _migrate(self) -> None:
        version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"analytics database schema {version} is newer than supported {SCHEMA_VERSION}")
        if version == 0:
            with self._db:
                self._db.executescript("""
                CREATE TABLE events(
                    event_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL UNIQUE,
                    event_type TEXT NOT NULL, timestamp_utc TEXT NOT NULL,
                    strategy TEXT, severity TEXT, payload_json TEXT NOT NULL
                );
                CREATE INDEX idx_events_time_type ON events(timestamp_utc,event_type);
                CREATE INDEX idx_events_strategy_time ON events(strategy,timestamp_utc);
                CREATE TABLE market_bars(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL, contract TEXT, source TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, bid REAL, ask REAL, reference_price REAL,
                    feed_timestamp_utc TEXT, feature_json TEXT NOT NULL, final_rth INTEGER
                );
                CREATE INDEX idx_market_bars_time ON market_bars(timestamp_utc);
                CREATE TABLE evaluations(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    strategy TEXT, action TEXT, signal TEXT, reason TEXT, features_json TEXT NOT NULL
                );
                CREATE TABLE candidates(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    strategy TEXT, status TEXT NOT NULL, reason TEXT, payload_json TEXT NOT NULL
                );
                CREATE INDEX idx_candidates_strategy_time ON candidates(strategy,timestamp_utc);
                CREATE TABLE risk_decisions(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    strategy TEXT, approved INTEGER, reason TEXT, quantity INTEGER, risk_per_contract REAL,
                    total_risk REAL, payload_json TEXT NOT NULL
                );
                CREATE TABLE orders(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), order_id TEXT,
                    timestamp_utc TEXT NOT NULL, strategy TEXT, status TEXT, quantity INTEGER,
                    reference_price REAL, payload_json TEXT NOT NULL
                );
                CREATE INDEX idx_orders_order_id ON orders(order_id);
                CREATE TABLE fills(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), fill_id TEXT UNIQUE,
                    order_id TEXT, timestamp_utc TEXT NOT NULL, strategy TEXT, contract TEXT,
                    quantity INTEGER, price REAL, side TEXT, commission REAL NOT NULL DEFAULT 0,
                    exchange_fee REAL NOT NULL DEFAULT 0, regulatory_fee REAL NOT NULL DEFAULT 0,
                    artificial_slippage REAL NOT NULL DEFAULT 0, observed_spread REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE positions(
                    position_event_id TEXT PRIMARY KEY REFERENCES events(event_id), trade_id TEXT,
                    strategy TEXT, timestamp_utc TEXT NOT NULL, status TEXT NOT NULL,
                    side TEXT, quantity INTEGER, entry_price REAL, exit_price REAL, payload_json TEXT NOT NULL
                );
                CREATE TABLE trades(
                    trade_id TEXT PRIMARY KEY, opening_event_id TEXT UNIQUE REFERENCES events(event_id),
                    closing_event_id TEXT UNIQUE REFERENCES events(event_id), strategy TEXT NOT NULL,
                    signal_id TEXT, contract TEXT, direction TEXT, quantity INTEGER,
                    signal_timestamp_utc TEXT, order_timestamp_utc TEXT, entry_timestamp_utc TEXT NOT NULL,
                    exit_timestamp_utc TEXT, entry_reference_price REAL, entry_fill_price REAL,
                    exit_reference_price REAL, exit_fill_price REAL, stop_price REAL, target_price REAL,
                    exit_reason TEXT, initial_risk_usd REAL, initial_risk_per_contract REAL,
                    gross_pnl REAL, commission REAL, exchange_fees REAL, regulatory_fees REAL,
                    total_costs REAL, net_pnl REAL, realized_r REAL, duration_seconds REAL,
                    mae_points REAL, mfe_points REAL, entry_features_json TEXT,
                    hmm_state INTEGER, hmm_posterior_json TEXT, model_version TEXT,
                    account_balance_after REAL, drawdown_after REAL, status TEXT NOT NULL,
                    close_payload_json TEXT
                );
                CREATE INDEX idx_trades_strategy_entry ON trades(strategy,entry_timestamp_utc);
                CREATE INDEX idx_trades_exit ON trades(exit_timestamp_utc);
                CREATE TABLE hmm_observations(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    stream TEXT, raw_state INTEGER, posterior_json TEXT, model_version TEXT, model_hash TEXT
                );
                CREATE TABLE model_events(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    stream TEXT, event_type TEXT NOT NULL, model_version TEXT, model_hash TEXT,
                    training_start TEXT, training_end TEXT, fit_duration REAL, payload_json TEXT NOT NULL
                );
                CREATE TABLE sessions(
                    trading_date TEXT PRIMARY KEY, session_type TEXT, rth_start TEXT, rth_end TEXT,
                    final_rth_bar TEXT, globex_open TEXT, globex_close TEXT,
                    calendar_source TEXT, calendar_version TEXT, calendar_identity TEXT
                );
                CREATE TABLE account_snapshots(
                    snapshot_id TEXT PRIMARY KEY, timestamp_utc TEXT NOT NULL, balance REAL NOT NULL,
                    equity REAL NOT NULL, realized_pnl REAL NOT NULL, unrealized_pnl REAL,
                    cumulative_net_r REAL, open_risk REAL, gross_exposure REAL,
                    drawdown REAL, open_positions_json TEXT NOT NULL
                );
                CREATE INDEX idx_account_time ON account_snapshots(timestamp_utc);
                CREATE TABLE configuration_versions(
                    config_id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
                    config_kind TEXT NOT NULL, version TEXT NOT NULL, identity TEXT NOT NULL,
                    source_json TEXT NOT NULL
                );
                CREATE TABLE checkpoint_events(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    action TEXT, checkpoint_path TEXT, last_processed_bar TEXT, payload_json TEXT NOT NULL
                );
                CREATE TABLE shadow_comparisons(
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id), timestamp_utc TEXT NOT NULL,
                    status TEXT, expected_count INTEGER, actual_count INTEGER, discrepancy_count INTEGER,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE daily_reports(
                    report_date TEXT PRIMARY KEY, generated_at_utc TEXT NOT NULL,
                    json_path TEXT NOT NULL, markdown_path TEXT NOT NULL, report_json TEXT NOT NULL
                );
                PRAGMA user_version=1;
                """)

    @staticmethod
    def _strategy(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("strategy_name") or payload.get("strategy")
        return str(value) if value is not None else None

    def ingest_event(self, event: PaperEvent) -> bool:
        """Insert one JSONL event and normalized fact rows in a single SQLite transaction."""
        payload = dict(event.payload)
        strategy = self._strategy(payload)
        stamp = event.timestamp.astimezone(timezone.utc).isoformat()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",
                (event.event_id, event.sequence, event.event_type.value, stamp, strategy,
                 payload.get("severity"), _json(payload)),
            )
            if cur.rowcount == 0:
                return False
            kind = event.event_type
            if kind is PaperEventType.MARKET_DATA:
                self._insert_market_bar(event.event_id, stamp, payload)
                self._record_session(stamp)
            elif kind in {PaperEventType.STRATEGY_DECISION, PaperEventType.CANDIDATE_EVALUATION}:
                self._db.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?)", (
                    event.event_id, stamp, strategy, payload.get("action"), payload.get("signal"),
                    payload.get("reason"), _json(payload.get("features", payload)),
                ))
            elif kind in {PaperEventType.CANDIDATE_CREATED, PaperEventType.CANDIDATE_REJECTED}:
                if kind is PaperEventType.CANDIDATE_CREATED:
                    self._db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?)", (
                        event.event_id, stamp, strategy, "CREATED", payload.get("reason"), _json(payload),
                    ))
                else:
                    row = self._db.execute("SELECT event_id FROM candidates WHERE strategy=? AND timestamp_utc=? AND status='CREATED' ORDER BY rowid DESC LIMIT 1", (strategy,stamp)).fetchone()
                    if row:
                        self._db.execute("UPDATE candidates SET status='REJECTED',reason=?,payload_json=? WHERE event_id=?",
                                         (payload.get("reason"),_json(payload),row["event_id"]))
                    else:
                        self._db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?)", (
                            event.event_id,stamp,strategy,"REJECTED",payload.get("reason"),_json(payload)))
            elif kind is PaperEventType.RISK_DECISION:
                self._db.execute("INSERT INTO risk_decisions VALUES(?,?,?,?,?,?,?,?,?)", (
                    event.event_id, stamp, strategy, int(bool(payload.get("approved"))),
                    payload.get("reason"), payload.get("quantity", payload.get("final_quantity")),
                    payload.get("risk_per_contract"), payload.get("total_risk"), _json(payload),
                ))
            elif kind in {PaperEventType.ORDER_CREATED, PaperEventType.ORDER_SUBMITTED,
                          PaperEventType.ORDER_FILLED, PaperEventType.ORDER_CANCELLED,
                          PaperEventType.ORDER_REJECTED, PaperEventType.PAPER_ORDER_CREATED,
                          PaperEventType.PAPER_ORDER_FILLED}:
                self._db.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?)", (
                    event.event_id, payload.get("broker_order_id", payload.get("order_id")),
                    stamp, strategy, kind.value, payload.get("quantity"),
                    payload.get("price", payload.get("reference_price")), _json(payload),
                ))
                if kind is PaperEventType.ORDER_CREATED:
                    self._db.execute("UPDATE candidates SET status='ACCEPTED' WHERE strategy=? AND timestamp_utc=? AND status='CREATED'",
                                     (strategy,stamp))
            elif kind is PaperEventType.FILL:
                self._insert_fill(event.event_id, stamp, payload)
            elif kind is PaperEventType.POSITION_OPENED:
                self._open_trade(event.event_id, stamp, strategy, payload)
            elif kind is PaperEventType.POSITION_CLOSED:
                self._close_trade(event.event_id, stamp, strategy, payload)
            elif kind is PaperEventType.HMM_STATE:
                self._db.execute("INSERT INTO hmm_observations VALUES(?,?,?,?,?,?,?)", (
                    event.event_id, stamp, payload.get("stream"), payload.get("raw_state"),
                    _json(payload.get("posterior")), payload.get("model_version"), payload.get("model_hash"),
                ))
            elif kind in {PaperEventType.HMM_REFIT_STARTED, PaperEventType.HMM_REFIT_COMPLETED,
                          PaperEventType.HMM_REFIT_FAILED}:
                self._db.execute("INSERT INTO model_events VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    event.event_id, stamp, payload.get("stream"), kind.value,
                    str(payload.get("model_version", "")), payload.get("model_hash"),
                    payload.get("training_start"), payload.get("training_last_timestamp"),
                    payload.get("wall_duration_seconds"), _json(payload),
                ))
            elif kind in {PaperEventType.CHECKPOINT_SAVED, PaperEventType.CHECKPOINT_RESTORED,
                          PaperEventType.CHECKPOINT_FAILED, PaperEventType.SYSTEM_RECOVERED}:
                self._db.execute("INSERT INTO checkpoint_events VALUES(?,?,?,?,?,?)", (
                    event.event_id, stamp, kind.value, payload.get("checkpoint_path"),
                    payload.get("last_processed_bar"), _json(payload),
                ))
            elif kind is PaperEventType.PARITY_WARNING:
                self._db.execute("INSERT INTO shadow_comparisons VALUES(?,?,?,?,?,?,?)", (
                    event.event_id, stamp, payload.get("status"), payload.get("expected_count"),
                    payload.get("actual_count"), payload.get("discrepancy_count"), _json(payload),
                ))
        return True

    def _insert_market_bar(self, event_id: str, stamp: str, p: Mapping[str, Any]) -> None:
        feature_keys = ("log_return", "realized_vol_5", "realized_vol_15", "realized_vol_30",
                        "realized_vol_60", "variance_ratio_5_30", "variance_ratio_5_60",
                        "hmm_state", "hmm_posterior", "hmm_model_version", "s2r_hmm_state",
                        "s2r_hmm_posterior", "s2r_model_version", "signal_quality",
                        "volatility_percentile", "zscore")
        features = {key: value for key, value in p.items() if key not in {
            "open", "high", "low", "close", "volume", "provider", "source_id",
        }}
        ref = p.get("reference_price", p.get("mid", p.get("close")))
        self._db.execute("INSERT INTO market_bars VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            event_id, stamp, p.get("symbol", "MNQ"), p.get("contract_symbol"),
            p.get("provider", p.get("source")), p.get("open"), p.get("high"), p.get("low"),
            p.get("close"), p.get("volume"), p.get("bid"), p.get("ask"), ref,
            p.get("provider_timestamp"), _json(features), int(bool(p.get("is_final_rth_bar", False))),
        ))

    def _record_session(self, stamp: str) -> None:
        if self.calendar is None:
            return
        from pandas import Timestamp
        session = self.calendar.session_for_timestamp(Timestamp(stamp))
        trading_date = session.trading_date.isoformat()
        self._db.execute("INSERT OR IGNORE INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?)", (
            trading_date, session.session_type,
            session.rth_start.isoformat() if session.rth_start else None,
            session.rth_end.isoformat() if session.rth_end else None,
            session.final_rth_bar.isoformat() if session.final_rth_bar else None,
            session.globex_open.isoformat() if session.globex_open else None,
            session.globex_close.isoformat() if session.globex_close else None,
            session.calendar_source, session.calendar_version, self.calendar.snapshot.identity,
        ))

    def _latest_event_payload(self, event_type: str, strategy: str, stamp: str) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT payload_json FROM events WHERE event_type=? AND strategy=? AND timestamp_utc<=? ORDER BY sequence DESC LIMIT 1",
            (event_type, strategy, stamp),
        ).fetchone()
        return json.loads(row[0]) if row else {}

    def _open_trade(self, event_id: str, stamp: str, strategy: str | None, p: Mapping[str, Any]) -> None:
        if not strategy:
            return
        trade_id = "paper-" + event_id
        risk = self._latest_event_payload(PaperEventType.RISK_REQUEST.value, strategy, stamp)
        market = self._db.execute("SELECT * FROM market_bars WHERE timestamp_utc<=? ORDER BY timestamp_utc DESC LIMIT 1", (stamp,)).fetchone()
        features = json.loads(market["feature_json"]) if market else {}
        qty = int(p.get("quantity", 0))
        risk_per = risk.get("risk_per_contract")
        position_payload = {**dict(p), "trade_id": trade_id}
        self._db.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?,?,?,?)", (
            event_id, trade_id, strategy, stamp, "OPEN", p.get("side"), qty,
            p.get("entry_price"), None, _json(position_payload),
        ))
        self._db.execute("""INSERT INTO trades(
            trade_id,opening_event_id,strategy,signal_id,contract,direction,quantity,
            signal_timestamp_utc,order_timestamp_utc,entry_timestamp_utc,
            entry_reference_price,entry_fill_price,stop_price,initial_risk_usd,
            initial_risk_per_contract,entry_features_json,hmm_state,hmm_posterior_json,
            model_version,status)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            trade_id,event_id,strategy,risk.get("signal_id"),p.get("contract_symbol"),p.get("side"),qty,
            risk.get("signal_timestamp"),risk.get("order_timestamp"),stamp,
            risk.get("entry_price"),p.get("entry_price"),risk.get("stop_price"),
            float(risk_per)*qty if risk_per is not None else None,risk_per,_json(features),
            features.get("hmm_state"),_json(features.get("hmm_posterior")),features.get("hmm_model_version"),"OPEN",
        ))

    def _close_trade(self, event_id: str, stamp: str, strategy: str | None, p: Mapping[str, Any]) -> None:
        if not strategy:
            return
        opened = self._db.execute("SELECT * FROM trades WHERE strategy=? AND status='OPEN' ORDER BY entry_timestamp_utc DESC LIMIT 1", (strategy,)).fetchone()
        if opened is None:
            return
        from src.paper.costs import calculate_trade_economics
        qty = int(p.get("quantity", opened["quantity"]))
        economics: dict[str, Any]
        if self.cost_policy is not None:
            economics = calculate_trade_economics(
                direction=str(p.get("side", opened["direction"])),
                entry_price=float(p.get("entry_price", opened["entry_fill_price"])),
                exit_price=float(p["exit_price"]), quantity=qty, point_value=self.point_value,
                initial_risk_usd=opened["initial_risk_usd"], policy=self.cost_policy,
            )
        else:
            side = str(p.get("side", opened["direction"])).lower()
            entry, exit_price = float(p.get("entry_price", opened["entry_fill_price"])), float(p["exit_price"])
            gross = (exit_price - entry if side in {"long", "strategysignal.long"} else entry - exit_price) * qty * self.point_value
            risk = opened["initial_risk_usd"]
            economics = {"gross_pnl": gross,"commission":None,"exchange_fees":None,
                         "regulatory_fees":None,"total_costs":None,"net_pnl":None,
                         "realized_r": gross/float(risk) if risk else None}
        for field in ("gross_pnl", "commission", "exchange_fees", "regulatory_fees", "total_costs", "net_pnl", "realized_r"):
            if p.get(field) is not None:
                economics[field] = p[field]
        risk_usd = p.get("initial_risk_usd", opened["initial_risk_usd"])
        if economics.get("realized_r") is None and risk_usd and economics.get("net_pnl") is not None:
            economics["realized_r"] = float(economics["net_pnl"]) / float(risk_usd)
        feature_rows = self._db.execute(
            "SELECT high,low FROM market_bars WHERE timestamp_utc>? AND timestamp_utc<=? ORDER BY timestamp_utc",
            (opened["entry_timestamp_utc"], stamp),
        ).fetchall()
        mae = mfe = None
        if feature_rows:
            entry = float(p.get("entry_price", opened["entry_fill_price"]))
            side = str(p.get("side", opened["direction"])).lower()
            highs = [float(r["high"]) for r in feature_rows if r["high"] is not None]
            lows = [float(r["low"]) for r in feature_rows if r["low"] is not None]
            if highs and lows:
                if side in {"long", "strategysignal.long"}:
                    mae, mfe = max(0.0, entry-min(lows)), max(0.0, max(highs)-entry)
                else:
                    mae, mfe = max(0.0, max(highs)-entry), max(0.0, entry-min(lows))
        exit_decision = self._latest_event_payload(PaperEventType.STRATEGY_DECISION.value, strategy, stamp)
        duration = (datetime.fromisoformat(stamp)-datetime.fromisoformat(opened["entry_timestamp_utc"])).total_seconds()
        account = self._db.execute("SELECT equity,drawdown FROM account_snapshots ORDER BY timestamp_utc DESC LIMIT 1").fetchone()
        self._db.execute("UPDATE trades SET closing_event_id=?,exit_timestamp_utc=?,exit_fill_price=?,exit_reason=?,gross_pnl=?,commission=?,exchange_fees=?,regulatory_fees=?,total_costs=?,net_pnl=?,realized_r=?,duration_seconds=?,mae_points=?,mfe_points=?,account_balance_after=?,drawdown_after=?,status='CLOSED',close_payload_json=? WHERE trade_id=?", (
            event_id,stamp,p.get("exit_price"),exit_decision.get("reason"),economics["gross_pnl"],
            economics["commission"],economics["exchange_fees"],economics["regulatory_fees"],
            economics["total_costs"],economics["net_pnl"],economics["realized_r"],duration,
            mae,mfe,p.get("account_balance_after", account["equity"] if account else None),account["drawdown"] if account else None,
            _json(p),opened["trade_id"],
        ))
        self._db.execute("UPDATE positions SET status='CLOSED',timestamp_utc=?,exit_price=?,payload_json=? WHERE trade_id=? AND status='OPEN'", (
            stamp,p.get("exit_price"),_json(p),opened["trade_id"],
        ))

    def _insert_fill(self, event_id: str, stamp: str, p: Mapping[str, Any]) -> None:
        qty = int(p.get("quantity", 0))
        policy = self.cost_policy
        commission = float(policy.commission_per_contract_side)*qty if policy else 0.0
        exchange = float(policy.exchange_fee_per_contract_side)*qty if policy else 0.0
        regulatory = float(policy.regulatory_fee_per_contract_side)*qty if policy else 0.0
        market = self._db.execute("SELECT bid,ask FROM market_bars WHERE timestamp_utc<=? ORDER BY timestamp_utc DESC LIMIT 1", (stamp,)).fetchone()
        spread = (market["ask"]-market["bid"]) if market and market["ask"] is not None and market["bid"] is not None else None
        self._db.execute("INSERT OR IGNORE INTO fills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            event_id,p.get("fill_id"),p.get("broker_order_id"),stamp,self._strategy(p),
            p.get("contract_symbol"),qty,p.get("price"),p.get("signal"),
            p.get("commission",commission),p.get("exchange_fee",exchange),
            p.get("regulatory_fee",regulatory),p.get("artificial_slippage",0.0),spread,_json(p),
        ))

    def record_account_snapshot(self, *, timestamp: datetime, balance: float, equity: float,
                                realized_pnl: float, unrealized_pnl: float | None,
                                cumulative_net_r: float | None, open_risk: float,
                                gross_exposure: float, drawdown: float,
                                open_positions: list[Mapping[str, Any]]) -> None:
        key = timestamp.astimezone(timezone.utc).isoformat()
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO account_snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                key,key,balance,equity,realized_pnl,unrealized_pnl,cumulative_net_r,open_risk,
                gross_exposure,drawdown,_json(open_positions),
            ))

    def record_configuration(self, *, config_kind: str, version: str, identity: str,
                             source: Mapping[str, Any]) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO configuration_versions VALUES(?,?,?,?,?,?)", (
                identity,datetime.now(timezone.utc).isoformat(),config_kind,version,identity,_json(source),
            ))

    def update_closed_trade_account(self, *, timestamp: datetime, balance: float,
                                    equity: float, drawdown: float) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE trades SET account_balance_after=?,drawdown_after=? WHERE exit_timestamp_utc=?",
                             (balance,drawdown,timestamp.astimezone(timezone.utc).isoformat()))

    def backfill_jsonl(self, events: list[PaperEvent]) -> int:
        return sum(bool(self.ingest_event(event)) for event in events)

    def integrity_check(self) -> str:
        return str(self._db.execute("PRAGMA integrity_check").fetchone()[0])

    def backup_to(self, destination: str | Path) -> Path:
        """Write a consistent SQLite snapshot without copying a live WAL file."""
        target = Path(destination).resolve()
        if target == self.path.resolve():
            raise ValueError("database backup destination must differ from the live database")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with self._lock:
                backup_db = sqlite3.connect(temporary)
                try:
                    self._db.backup(backup_db)
                    result = str(backup_db.execute("PRAGMA integrity_check").fetchone()[0])
                    if result != "ok":
                        raise RuntimeError(f"database backup integrity check failed: {result}")
                finally:
                    backup_db.close()
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            return target
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def closed_trades(self, *, start: str | None = None, end: str | None = None,
                      strategy: str | None = None) -> list[dict[str, Any]]:
        clauses = ["status='CLOSED'"]
        params: list[Any] = []
        if start:
            clauses.append("exit_timestamp_utc>=?"); params.append(start)
        if end:
            clauses.append("exit_timestamp_utc<?"); params.append(end)
        if strategy:
            clauses.append("strategy=?"); params.append(strategy)
        rows = self._db.execute("SELECT * FROM trades WHERE " + " AND ".join(clauses) + " ORDER BY exit_timestamp_utc", params)
        return [dict(r) for r in rows]

    def write_daily_report(self, day: date, output_dir: str | Path) -> dict[str, Any]:
        from src.paper.analytics import build_daily_report
        report = build_daily_report(self, day)
        target = Path(output_dir) / "daily_reports"
        target.mkdir(parents=True, exist_ok=True)
        json_path, md_path = target / f"{day.isoformat()}.json", target / f"{day.isoformat()}.md"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        md_path.write_text(_daily_markdown(report), encoding="utf-8")
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO daily_reports VALUES(?,?,?,?,?)", (
                day.isoformat(),datetime.now(timezone.utc).isoformat(),str(json_path),str(md_path),_json(report),
            ))
        return report

    def generate_daily_reports(self, output_dir: str | Path) -> list[dict[str, Any]]:
        from pandas import Timestamp
        stamps = [Timestamp(row[0]) for row in self._db.execute("SELECT DISTINCT timestamp_utc FROM market_bars")]
        days = sorted({stamp.tz_convert("America/New_York").date() for stamp in stamps})
        return [self.write_daily_report(day, output_dir) for day in days]


def _daily_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# Paper daily report — {report['date']}", "", f"Technical: **{report['status']['technical']}** | Operational: **{report['status']['operational']}** | Statistical: **{report['status']['statistical']}**", "", "## Portfolio", "", f"- Closed trades: {report['portfolio']['trades']}", f"- Gross P&L: {report['portfolio']['gross_pnl']}", f"- Costs: {report['portfolio']['total_costs']}", f"- Net P&L: {report['portfolio']['net_pnl']}", "", "## Strategies", "", "| Strategy | Trades | Win rate | PF | Net P&L | Expectancy R |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["strategies"].items():
        lines.append(f"| {name} | {row['trades']} | {row['win_rate']} | {row['profit_factor']} | {row['net_pnl']} | {row['expectancy_r']} |")
    return "\n".join(lines) + "\n"
