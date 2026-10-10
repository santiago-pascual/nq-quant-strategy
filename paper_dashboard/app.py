"""Local-first, read-only monitoring UI for MNQ Paper Trading."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
import logging
import os
from pathlib import Path
import secrets
import sys
from threading import Thread
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_dashboard.run_selector import RunOption, default_run, discover_runs
from paper_dashboard.display_values import alert_table_rows, market_display_state, account_max_drawdown
from paper_dashboard.research_analytics import load_validated_research
from paper_dashboard.analytics_views import render_analytics
from paper_dashboard.closed_outcome_metrics import closed_trade_drawdown

STRATEGIES = ("MRL1", "MRS2", "S2R", "ORB")
PAGES = ("Command Center", "Performance", "Analytics", "Strategies", "Trade Explorer", "Positions & Orders", "Risk Monitor", "System Observatory")
BG = "#0b0f14"
SURFACE = "#111820"
SURFACE_ALT = "#151e27"
LINE = "#27333e"
TEXT = "#e6ebef"
MUTED = "#93a0ad"
GREEN = "#67c6a3"
RED = "#d77f87"
AMBER = "#d4b574"


def _configure_dashboard_logging() -> logging.Logger:
    logger = logging.getLogger("mnq_paper_dashboard")
    if logger.handlers:
        return logger
    log_dir = Path(os.environ.get(
        "PAPER_DASHBOARD_LOG_DIR",
        str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MNQPaperDashboard" / "logs"),
    ))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(log_dir / "dashboard.log", maxBytes=2_000_000,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


LOGGER = _configure_dashboard_logging()

st.set_page_config(page_title="MNQ | Paper Operations", page_icon=None, layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<style>
:root {{ color-scheme: dark; --mnq-bg:{BG}; --mnq-surface:{SURFACE}; --mnq-line:{LINE}; }}
.stApp {{ background:var(--mnq-bg); color:{TEXT}; }}
[data-testid="stHeader"] {{ background:rgba(11,15,20,.94); border-bottom:1px solid #1c2731; }}
[data-testid="stSidebar"] {{ background:#0e141b; border-right:1px solid #202a34; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:{MUTED}; }}
.block-container {{ max-width:1640px; padding-top:1.5rem; padding-bottom:3rem; }}
h1,h2,h3 {{ letter-spacing:-.025em; }}
h1 {{ font-size:1.7rem!important; font-weight:620!important; }}
h2 {{ font-size:1.25rem!important; font-weight:600!important; }}
h3 {{ font-size:1.02rem!important; font-weight:580!important; }}
.mnq-eyebrow {{ color:{MUTED}; font:600 .69rem/1.2 ui-monospace,Consolas,monospace; letter-spacing:.13em; text-transform:uppercase; margin:0 0 .4rem; }}
.mnq-caption {{ color:{MUTED}; font-size:.82rem; }}
.mnq-empty {{ background:{SURFACE}; border:1px solid {LINE}; border-left:2px solid #435665; border-radius:7px; padding:11px 13px; color:{MUTED}; font-size:.84rem; }}
.mnq-metric-value {{ color:{TEXT}; font:500 1.32rem/1.25 ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums lining-nums; font-feature-settings:"tnum"; white-space:normal; overflow-wrap:anywhere; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:7px; overflow:hidden; }}
[data-testid="stAlert"] {{ border-radius:7px; }}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-color:{LINE}; border-radius:8px; }}
div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label, div[data-testid="stTextInput"] label {{ color:{MUTED}; font-size:.78rem; }}
.stPlotlyChart {{ border:1px solid {LINE}; border-radius:8px; overflow:hidden; background:{SURFACE}; }}
hr {{ border-color:#202a34; }}
@media(max-width:900px) {{
  .block-container {{ padding:1rem .7rem 2rem; }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap!important; gap:.55rem!important; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ min-width:min(100%, 200px)!important; flex:1 1 45%!important; }}
  .mnq-metric-value {{ font-size:1.12rem; }}
}}
</style>
""", unsafe_allow_html=True)


def _output_dir() -> Path:
    return Path(os.environ.get("PAPER_MONITOR_OUTPUT_DIR", "results/paper/realtime")).resolve()


@st.cache_data(ttl=300, show_spinner=False)
def _validated_research(root: str) -> tuple[dict[str, Any] | None, str | None]:
    """Cache the immutable, hash-verified Research benchmark for five minutes."""
    try:
        return load_validated_research(root), None
    except Exception as exc:
        LOGGER.exception("validated research analytics unavailable")
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def _start_api(output_dir: str, port: int) -> tuple[Any, str, int]:
    from src.paper.monitoring_api import create_monitoring_server

    token = secrets.token_urlsafe(36)
    try:
        server = create_monitoring_server(output_dir, token=token, host="127.0.0.1", port=port)
    except OSError:
        if not port:
            raise
        # A prior app session may still own its API port. Isolate each selected
        # run on its own ephemeral loopback-only port instead of failing the UI.
        server = create_monitoring_server(output_dir, token=token, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .5},
                    name="paper-readonly-api", daemon=True)
    thread.start()
    return server, token, int(server.server_port)


def _api_json(api_port: int, token: str, path: str) -> Any:
    request = Request(f"http://127.0.0.1:{api_port}{path}",
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=2.0) as response:
            body = response.read(2_000_000)
    except HTTPError as exc:
        try:
            error = json.loads(exc.read(64_000)).get("data", {}).get("error")
        except (ValueError, json.JSONDecodeError):
            error = None
        if error == "analytics_database_unavailable":
            return {"_unavailable": "Analytics database has not been persisted yet."}
        return {"_error": f"Monitoring API request failed (HTTP {exc.code})."}
    except (URLError, TimeoutError, OSError) as exc:
        return {"_error": f"Monitoring API unavailable ({type(exc).__name__})"}
    try:
        return json.loads(body).get("data", {})
    except (ValueError, json.JSONDecodeError):
        return {"_error": "Invalid JSON from monitoring API"}


def _money(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        number = float(value)
        if number < 0:
            return f"-${abs(number):,.2f}"
        return f"+${number:,.2f}" if signed and number > 0 else f"${number:,.2f}"
    except (TypeError, ValueError):
        return "Unavailable"


def _number(value: Any, places: int = 0) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return "Unavailable"


def _percent(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def _stamp(value: Any, local: bool = False) -> str:
    if not value:
        return "Unavailable"
    try:
        from datetime import datetime, timezone
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return "Unavailable"
        parsed = parsed.astimezone() if local else parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (TypeError, ValueError):
        return str(value)


def _stamp_dual(value: Any) -> str:
    """Show market instants in UTC and the user's Argentina clock."""
    if not value:
        return "Unavailable"
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return "Unavailable"
        utc = parsed.astimezone(timezone.utc)
        local = utc.astimezone(ZoneInfo("America/Argentina/Buenos_Aires"))
        return f"{utc:%Y-%m-%d %H:%M} UTC · {local:%Y-%m-%d %H:%M} ART"
    except (TypeError, ValueError, KeyError):
        return "Unavailable"


def _status_line(run: dict[str, Any]) -> None:
    kind = run.get("kind") or "UNAVAILABLE"
    status = run.get("status") or "UNAVAILABLE"
    label = ("Historical replay · simulated results" if kind == "HISTORICAL_REPLAY"
             else "Delayed Paper · simulated fills" if kind == "DELAYED_PAPER"
             else "Realtime Paper · simulated fills" if kind == "REALTIME_PAPER" else str(kind))
    color = GREEN if status in {"RUNNING", "COMPLETED", "CAUGHT_UP"} else RED if status in {"FAILED", "DEGRADED", "ERROR"} else AMBER
    st.markdown(f"<span style='color:{MUTED};font-size:.83rem'>{escape(label)}</span> &nbsp; <span style='color:{color};font:600 .76rem ui-monospace,Consolas,monospace'>{escape(status.replace('_',' '))}</span>", unsafe_allow_html=True)


def _chart_layout(fig: go.Figure, *, height: int = 330, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"color": MUTED, "family": "Inter, Segoe UI, sans-serif", "size": 11},
        margin={"l": 50, "r": 18, "t": 24, "b": 38}, height=height,
        hovermode="x unified", showlegend=False,
        xaxis={"showgrid": False, "linecolor": LINE, "zeroline": False, "tickfont": {"color": MUTED}},
        yaxis={"showgrid": True, "gridcolor": "#222d37", "zerolinecolor": LINE,
              "title": y_title, "tickfont": {"color": MUTED}, "automargin": True},
    )
    return fig


def _line_chart(points: list[dict[str, Any]], xkey: str, ykey: str, *, color: str = GREEN,
                fill: bool = False, height: int = 330, y_title: str | None = None) -> go.Figure | None:
    rows = []
    for row in points:
        try:
            if row.get(xkey) is not None and row.get(ykey) is not None:
                rows.append((row[xkey], float(row[ykey])))
        except (TypeError, ValueError):
            continue
    if len(rows) < 2:
        return None
    fig = go.Figure(go.Scatter(
        x=[r[0] for r in rows], y=[r[1] for r in rows], mode="lines",
        line={"color": color, "width": 1.8}, fill="tozeroy" if fill else None,
        fillcolor="rgba(103,198,163,.08)" if color == GREEN else "rgba(215,127,135,.08)",
        hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
    ))
    return _chart_layout(fig, height=height, y_title=y_title)


def _metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for col, (label, value, delta) in zip(columns, items):
        with col.container(border=True):
            st.caption(label)
            from html import escape
            st.markdown(f"<div class='mnq-metric-value'>{escape(str(value))}</div>", unsafe_allow_html=True)
            if delta:
                st.caption(delta)


def _table(rows: Any, *, empty: str, columns: list[str] | None = None, height: int = 360) -> None:
    if not rows:
        _empty(empty)
    else:
        data = [{key: row.get(key) for key in columns} for row in rows] if columns else rows
        st.dataframe(data, width="stretch", hide_index=True, height=height)


def _empty(message: str) -> None:
    from html import escape
    st.markdown(f"<div class='mnq-empty'>{escape(message)}</div>", unsafe_allow_html=True)


def _scope_caption(run: dict[str, Any]) -> None:
    scope = run.get("scope") or {}
    if scope:
        st.caption(f"Scope · {_stamp(scope.get('start_utc_inclusive'))} → {_stamp(scope.get('replay_end_utc_exclusive'))} (exclusive)")
    else:
        st.caption("No replay scope is persisted. Historical or realtime classification follows the run manifest and status files.")


def _command_center(data: dict[str, Any]) -> None:
    run = data.get("run") or {}
    status = data.get("status") or {}
    account = data.get("account") or {}
    portfolio = data.get("portfolio") or {}
    _status_line(run)
    _scope_caption(run)
    if run.get("kind") == "DELAYED_PAPER":
        engine = status.get("system") or {}
        health = engine.get("feed_health") or {}
        market_state = market_display_state(data.get('alerts'))
        st.caption(f"Reviewed CME session · {market_state} · Acquisition mode · {health.get('mode') or 'Unavailable'}")
        _metric_row([
            ("Paper bars committed", _number(engine.get("bars_processed")), None),
            ("Last committed bar · UTC", _stamp(engine.get("last_bar")), None),
            ("Provider frontier · UTC", _stamp(health.get("latest_available_bar")), None),
            ("Recovery backlog", _number(health.get("backlog_bars") if health.get("backlog_bars") is not None else engine.get("queue_depth")),
             str(health.get("mode") or "Feed state unavailable")),
        ])
        st.caption(f"Market clock · last Paper commit {_stamp_dual(engine.get('last_bar'))} · provider frontier {_stamp_dual(health.get('latest_available_bar'))}")
        if health.get("last_error"):
            st.error(f"Feed recovery error: {health['last_error']}")
        elif market_state in {'MARKET_CLOSED', 'MARKET_BREAK'} and health.get('backlog_bars') == 0:
            st.info("Verified exchange closure · no processing backlog. New-bar validation resumes when the reviewed session opens.")
        elif health.get("mode") in {"RECOVERING", "DISCONNECTED", "BLOCKED"}:
            st.warning(f"Delayed feed status: {health.get('mode')}; backlog and provider timestamps are reported from persisted feed health.")
        activity = status.get("strategies") or {}
        activity_rows = [{
            "Strategy": name,
            "Candidates": (activity.get(name) or {}).get("candidates"),
            "Accepted": (activity.get(name) or {}).get("accepted"),
            "Rejected": (activity.get(name) or {}).get("rejected"),
            "Closed trades": (activity.get(name) or {}).get("closed_trades"),
            "Open position": (activity.get(name) or {}).get("has_open_position"),
        } for name in STRATEGIES]
        st.markdown("#### Strategy activity · persisted engine counters")
        _table(activity_rows, empty="Strategy activity is unavailable while this run initializes.", height=220)
    _metric_row([
        ("Account equity", _money(account.get("equity")), None),
        ("Balance", _money(account.get("balance")), None),
        ("Realized net P&L", _money(account.get("realized_pnl")), None),
        ("Unrealized P&L", _money(account.get("unrealized_pnl")), None),
    ])
    _metric_row([
        ("Gross P&L · closed trades", _money(portfolio.get("gross_pnl"), signed=True), None),
        ("Net P&L · closed trades", _money(portfolio.get("net_pnl"), signed=True), None),
        ("Total costs", _money(portfolio.get("total_costs")), None),
        ("Commissions", _money(portfolio.get("commissions")), None),
    ])
    _metric_row([
        ("Current drawdown", _money(account.get("drawdown")), None),
        ("Maximum account drawdown", _money(account_max_drawdown(data)), "Persisted account high-water mark"),
        ("Account floor", _money((data.get("risk") or {}).get("account_floor")), None),
        ("Distance to floor", _money(_distance_to_floor(account, data.get("risk") or {})), None),
    ])
    left, right = st.columns([1.65, 1])
    with left:
        st.markdown("#### Account path")
        curve = data.get("equity_curve") or []
        equity = _line_chart(curve, "timestamp_utc", "equity", color=GREEN, fill=True, y_title="Account equity · USD")
        if equity:
            st.plotly_chart(equity, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
        else:
            _empty("Account snapshots have not been persisted. Equity history is unavailable.")
    with right:
        st.markdown("#### Run status")
        st.metric("Processed bars", _number(run.get("processed_bars")))
        st.caption(f"Last processed bar · {_stamp(run.get('last_processed_bar'))}")
        system = (data.get("system") or {})
        latest = system.get("latest_event") or {}
        st.caption(f"Last persisted event · {latest.get('event_type') or 'Unavailable'}")
        if run.get("progress_pct") is None:
            st.caption(run.get("progress_note") or "Replay percentage unavailable: total bars are not persisted.")
        else:
            st.progress(float(run["progress_pct"]) / 100, text=f"{float(run['progress_pct']):.1f}%")
        if run.get("kind") == "HISTORICAL_REPLAY":
            _empty("Historical replay data · this run is not live market trading.")
        freshness = _data_freshness(data)
        st.caption(f"Market data · {freshness}")
    st.markdown("#### Portfolio drawdown")
    dd = _line_chart(curve, "timestamp_utc", "drawdown", color=RED, fill=True, y_title="Drawdown · USD")
    if dd:
        st.plotly_chart(dd, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
    else:
        _empty("Drawdown history is unavailable until account snapshots exist.")
    st.markdown("#### Strategy contribution")
    _strategy_summary(data.get("strategies") or {})
    sample = data.get("analytics_sample") or {}
    if sample.get("truncated"):
        st.warning(f"Portfolio trade metrics use the latest {sample.get('included_closed_trades'):,} of {sample.get('total_closed_trades'):,} closed trades.")
    elif sample.get("included_closed_trades") is not None:
        st.caption(f"Trade-based metrics use {sample.get('included_closed_trades'):,} persisted closed trades; account values come from the latest account snapshot.")
    st.markdown("#### Recent closed trades")
    closed = [t for t in data.get("trades") or [] if t.get("status") == "CLOSED"]
    _trade_table(closed[:12])


def _data_freshness(data: dict[str, Any]) -> str:
    run = data.get("run") or {}
    stamp = run.get("last_processed_bar")
    if not stamp:
        feed = (data.get("system") or {}).get("feed") or {}
        bar = feed.get("last_bar") or {}
        stamp = bar.get("timestamp_utc")
    if not stamp:
        return "Unavailable"
    if run.get("kind") == "HISTORICAL_REPLAY":
        return f"Historical timestamp {_stamp(stamp)}"
    try:
        from datetime import datetime, timezone
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds()
        return f"{max(0,age)/60:.1f} min old" if age >= 0 else "Timestamp is ahead of local clock"
    except (TypeError, ValueError):
        return "Unavailable"


def _distance_to_floor(account: dict[str, Any], risk: dict[str, Any]) -> float | None:
    floor = risk.get("account_floor", account.get("account_floor"))
    equity = account.get("equity", account.get("balance"))
    try:
        return float(equity) - float(floor) if equity is not None and floor is not None else None
    except (TypeError, ValueError):
        return None


def _strategy_summary(stats: dict[str, Any]) -> None:
    rows = []
    for name in STRATEGIES:
        s = stats.get(name) or {}
        rows.append({"Strategy": name, "Closed": s.get("trades"), "Gross P&L": s.get("gross_pnl"),
                     "Net P&L": s.get("net_pnl"), "Costs": s.get("total_costs"),
                     "Win rate": _percent(s.get("win_rate")), "Profit factor": _number(s.get("profit_factor"), 2),
                     "Expectancy (R)": _number(s.get("expectancy_r"), 3)})
    _table(rows, empty="Strategy statistics unavailable until closed trades are persisted.", height=220)


def _trade_table(rows: list[dict[str, Any]]) -> None:
    columns = ["trade_id", "strategy", "direction", "quantity", "entry_timestamp_utc", "entry_fill_price",
               "exit_timestamp_utc", "exit_fill_price", "gross_pnl", "total_costs", "net_pnl", "realized_r", "exit_reason", "status"]
    _table(rows, empty="No persisted trades match this view.", columns=columns)


def _date_filters(prefix: str = "analysis", date_basis: str = "Exit date") -> tuple[str | None, str | None, str | None]:
    c1, c2, c3 = st.columns([1, 1, 1])
    strategy = c1.selectbox("Strategy", ["All", *STRATEGIES], key=f"{prefix}_strategy")
    start = c2.date_input(f"{date_basis} from (UTC)", value=None, key=f"{prefix}_start")
    end = c3.date_input(f"{date_basis} through (UTC)", value=None, key=f"{prefix}_end")
    if start and end and start > end:
        st.error("Start date must be on or before end date.")
        return "INVALID", "INVALID", "INVALID"
    return (None if strategy == "All" else strategy,
            start.isoformat() if start else None,
            end.isoformat() if end else None)


def _query_path(route: str, params: dict[str, Any]) -> str:
    present = {key: value for key, value in params.items() if value not in (None, "")}
    return route + ("?" + urlencode(present) if present else "")


def _performance(data: dict[str, Any], api_port: int, token: str) -> None:
    st.markdown('<div class="mnq-eyebrow">CLOSED-TRADE ANALYSIS</div>', unsafe_allow_html=True)
    strategy, start, end = _date_filters("performance")
    query = {"strategy": strategy, "start": start, "end": end, "limit": 5000}
    analytics = _api_json(api_port, token, _query_path("/v1/analytics", query))
    if analytics.get("_unavailable"):
        _empty(analytics["_unavailable"])
        return
    if analytics.get("_error"):
        st.warning(analytics["_error"])
        return
    metrics = analytics.get("metrics") or {}
    sample = analytics.get("sample") or {}
    _metric_row([
        ("Closed trades", _number(sample.get("included_closed_trades")), f"Total: {_number(sample.get('total_closed_trades'))}"),
        ("Net P&L", _money(metrics.get("net_pnl"), signed=True), None),
        ("Expectancy / trade", _money(metrics.get("expectancy_usd"), signed=True), None),
        ("Expectancy (R)", _number(metrics.get("expectancy_r"), 3), None),
    ])
    _metric_row([
        ("Profit factor", _number(metrics.get("profit_factor"), 2), None),
        ("Win rate", _percent(metrics.get("win_rate")), None),
        ("Gross P&L", _money(metrics.get("gross_pnl"), signed=True), None),
        ("Costs", _money(metrics.get("total_costs")), None),
    ])
    _metric_row([
        ("Commissions", _money(metrics.get("commissions")), None),
        ("Exchange fees", _money(metrics.get("exchange_fees")), None),
        ("Regulatory fees", _money(metrics.get("regulatory_fees")), None),
        ("Maximum closed-outcome DD", _money(closed_trade_drawdown(analytics.get("trade_outcomes") or []).get("maximum_usd")), "Net outcomes; zero baseline; excludes intratrade equity"),
    ])
    if sample.get("truncated"):
        st.warning(f"Analytics are based on the most recent {sample.get('included_closed_trades'):,} of {sample.get('total_closed_trades'):,} matching closed trades. Increase the API sample cap only after reviewing query cost.")
    else:
        st.caption(sample.get("basis", "Analytics use persisted closed-trade records."))
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Cumulative net P&L")
        fig = _line_chart(analytics.get("cumulative_net_pnl") or [], "timestamp_utc", "cumulative_net_pnl", color=GREEN, fill=True, y_title="Net P&L · USD")
        if fig:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            _empty("At least two closed trades with P&L are needed for a cumulative series.")
    with right:
        st.markdown("#### Daily net performance · ET")
        daily = analytics.get("daily") or []
        if daily:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[r["date_et"] for r in daily], y=[r["gross_pnl"] for r in daily], name="Gross", marker_color="#6d8798"))
            fig.add_trace(go.Bar(x=[r["date_et"] for r in daily], y=[r["net_pnl"] for r in daily], name="Net", marker_color=[GREEN if r["net_pnl"] >= 0 else RED for r in daily]))
            fig.update_layout(barmode="group", showlegend=True, legend={"orientation":"h","y":1.08})
            st.plotly_chart(_chart_layout(fig, height=330, y_title="P&L · USD"), width="stretch", config={"displayModeBar": False})
        else:
            _empty("Daily results unavailable until matching trades close.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Net P&L distribution")
        pnl = analytics.get("pnl_distribution") or []
        values = [float(r["net_pnl"]) for r in pnl if r.get("net_pnl") is not None]
        if values:
            fig = go.Figure(go.Histogram(x=values, nbinsx=36, marker_color="#7592a7", opacity=.9,
                                         hovertemplate="Net P&L $%{x:,.2f}<br>Count %{y}<extra></extra>"))
            fig.add_vline(x=0, line_color=LINE, line_dash="dash")
            st.plotly_chart(_chart_layout(fig, height=290, y_title="Closed trades"), width="stretch", config={"displayModeBar": False})
        else:
            _empty("P&L distribution unavailable.")
    with right:
        st.markdown("#### Duration distribution")
        durations = [float(r["duration_seconds"])/60 for r in pnl if r.get("duration_seconds") is not None]
        if durations:
            fig = go.Figure(go.Histogram(x=durations, nbinsx=32, marker_color="#7592a7",
                                         hovertemplate="Duration %{x:.1f} min<br>Count %{y}<extra></extra>"))
            st.plotly_chart(_chart_layout(fig, height=290, y_title="Closed trades"), width="stretch", config={"displayModeBar": False})
        else:
            _empty("Duration data unavailable.")
    st.markdown("#### Strategy contribution")
    contribution = analytics.get("strategy_contribution") or []
    _table(contribution, empty="No strategy contribution data for these filters.", height=230)
    with st.expander("Metric definitions"):
        for label, description in (analytics.get("assumptions") or {}).items():
            st.markdown(f"**{label.replace('_',' ').title()}** · {description}")
        st.caption("Maximum drawdown in the stored trade metrics is calculated over cumulative net P&L for the included sample. Account drawdown is separately sourced from persisted account snapshots.")


def _strategies(data: dict[str, Any], api_port: int, token: str) -> None:
    strategy, start, end = _date_filters("strategy", "Exit date")
    analytics = _api_json(api_port, token, _query_path("/v1/analytics", {"strategy": strategy, "start": start, "end": end, "limit": 5000}))
    if analytics.get("_unavailable"):
        _empty(analytics["_unavailable"])
        return
    if analytics.get("_error"):
        st.warning(analytics["_error"])
        return
    _strategy_summary(analytics.get("strategies") or {})
    sample = analytics.get("sample") or {}
    st.caption(f"Closed-trade sample: {_number(sample.get('included_closed_trades'))}; truncated={sample.get('truncated', False)}. Activity and eligibility are shown only where persisted by engine status.")
    activity = (data.get("status") or {}).get("strategies") or {}
    activity_rows = []
    for name in STRATEGIES:
        row = activity.get(name) or {}
        activity_rows.append({"Strategy": name, "Candidates today": row.get("candidates_today"),
                              "Accepted today": row.get("accepted_today"), "Rejected today": row.get("rejected_today"),
                              "Open position": row.get("has_open_position")})
    st.markdown("#### Persisted activity")
    _table(activity_rows, empty="Strategy activity is unavailable until engine status is persisted.", height=220)
    st.markdown("#### Raw HMM regime timeline")
    bars = _api_json(api_port, token, "/v1/bars?limit=1200")
    if isinstance(bars, list) and bars:
        rows = []
        for bar in bars:
            features = bar.get("features") or {}
            if features.get("hmm_state") is not None or features.get("s2r_hmm_state") is not None:
                rows.append({"timestamp_utc": bar.get("timestamp_utc"),
                             "MR state": features.get("hmm_state"),
                             "S2R state": features.get("s2r_hmm_state")})
        if rows:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[r["timestamp_utc"] for r in rows], y=[r["MR state"] for r in rows],
                                     name="MR stream raw state", mode="lines", line={"shape":"hv","color":"#7f9fb4","width":1.5}))
            fig.add_trace(go.Scatter(x=[r["timestamp_utc"] for r in rows], y=[r["S2R state"] for r in rows],
                                     name="S2R raw state", mode="lines", line={"shape":"hv","color":AMBER,"width":1.4}))
            fig.update_layout(showlegend=True, legend={"orientation":"h","y":1.08}, yaxis={"dtick":1,"range":[-.5,2.5],"title":"Raw component ID"})
            st.plotly_chart(_chart_layout(fig, height=300, y_title="Raw component ID"), width="stretch", config={"displayModeBar": False})
            st.caption("Raw HMM component IDs are displayed as persisted. No state semantics are inferred or aligned.")
        else:
            _empty("No HMM state observations are present in the recent persisted bars.")
    else:
        _empty("HMM timeline unavailable until market-bar features are persisted.")


def _trades(api_port: int, token: str) -> None:
    st.markdown('<div class="mnq-eyebrow">PERSISTED PAPER LEDGER</div>', unsafe_allow_html=True)
    strategy, start, end = _date_filters("trades", "Entry date")
    query_text = st.text_input("Search visible trade fields", placeholder="Strategy, side, timestamp, exit reason, trade ID…", key="trade_search")
    rows = _api_json(api_port, token, _query_path("/v1/trades", {"strategy": strategy, "start": start, "end": end, "limit": 1000}))
    if isinstance(rows, dict) and rows.get("_error"):
        st.warning(rows["_error"])
        return
    if isinstance(rows, dict) and rows.get("_unavailable"):
        _empty(rows["_unavailable"])
        return
    rows = rows if isinstance(rows, list) else []
    filtered = [r for r in rows if not query_text or query_text.casefold() in json.dumps(r, default=str).casefold()]
    st.caption(f"Showing {len(filtered):,} of at most 1,000 matching trades, sorted by entry timestamp. Filters use UTC entry dates.")
    _trade_table(filtered)
    if filtered:
        choices = {f"{r.get('entry_timestamp_utc')} · {r.get('strategy')} · {r.get('direction')} · {r.get('trade_id')}": r for r in filtered}
        label = st.selectbox("Inspect trade", list(choices), key="trade_detail_choice")
        trade = choices[label]
        _metric_row([
            ("Gross P&L", _money(trade.get("gross_pnl"), signed=True), None),
            ("Net P&L", _money(trade.get("net_pnl"), signed=True), None),
            ("Costs", _money(trade.get("total_costs")), None),
            ("Realized R", _number(trade.get("realized_r"), 3), None),
        ])
        st.json({k:v for k,v in trade.items() if k not in {"entry_features_json", "close_payload_json", "hmm_posterior_json"}}, expanded=False)
        bars = _api_json(api_port, token, f"/v1/trade-bars/{trade.get('trade_id')}")
        if isinstance(bars, list) and bars:
            fig = go.Figure(go.Candlestick(x=[b["timestamp_utc"] for b in bars], open=[b["open"] for b in bars],
                                           high=[b["high"] for b in bars], low=[b["low"] for b in bars], close=[b["close"] for b in bars],
                                           increasing_line_color=GREEN, decreasing_line_color=RED, name="MNQ"))
            markers_x = [trade.get("entry_timestamp_utc"), trade.get("exit_timestamp_utc")]
            markers_y = [trade.get("entry_fill_price"), trade.get("exit_fill_price")]
            fig.add_trace(go.Scatter(x=markers_x, y=markers_y, mode="markers", marker={"size":10,"color":[GREEN,RED],"symbol":["triangle-up","x"]}, name="Paper fills"))
            st.markdown("#### Price path around trade")
            st.plotly_chart(_chart_layout(fig, height=380, y_title="MNQ price"), width="stretch", config={"displayModeBar": False})
        else:
            st.caption("No persisted market bars are available around this trade; price-path chart unavailable.")


def _risk(data: dict[str, Any], api_port: int, token: str) -> None:
    account = data.get("account") or {}
    risk = data.get("risk") or {}
    status = data.get("status") or {}
    state = status.get("system") or {}
    floor = risk.get("account_floor", account.get("account_floor"))
    current_balance = account.get("equity", account.get("balance"))
    buffer = float(current_balance) - float(floor) if current_balance is not None and floor is not None else None
    _metric_row([
        ("Current drawdown", _money(account.get("drawdown")), None),
        ("Maximum account drawdown", _money(account_max_drawdown(data)), "Persisted account high-water mark"),
        ("Open risk", _money(risk.get("open_risk")), None),
        ("Gross exposure", _money(risk.get("gross_exposure")), None),
    ])
    _metric_row([
        ("Account floor", _money(floor), None),
        ("Distance to floor", _money(buffer), None),
        ("Daily realized P&L", _money((status.get("portfolio") or {}).get("daily_pnl")), None),
        ("Account state", str(state.get("account_state") or account.get("status") or "Unavailable"), None),
    ])
    if floor is None:
        _empty("Account floor, failure threshold and limit utilization are unavailable because those values are not persisted in the current monitoring contract.")
    curve = data.get("equity_curve") or []
    fig = _line_chart(curve, "timestamp_utc", "drawdown", color=RED, fill=True, y_title="Drawdown · USD")
    if fig:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        _empty("Risk history unavailable until account snapshots are persisted.")
    _alert_panel(data.get("alerts") or (_api_json(api_port, token, "/v1/alerts") if token else {}))
    st.markdown("#### Open simulated positions")
    _table(data.get("positions") or [], empty="No persisted open Paper positions.")
    st.markdown("#### Verified risk warnings")
    warnings = (data.get("system") or {}).get("warnings") or []
    if warnings:
        for warning in warnings:
            st.warning(str(warning))
    else:
        st.caption("No persisted warnings. No proximity alert is calculated without a persisted limit and current utilization.")


def _execution(data: dict[str, Any]) -> None:
    orders = data.get("orders") or []
    fills = data.get("fills") or []
    positions = data.get("positions") or []
    _metric_row([
        ("Open positions", _number(len(positions)) if data.get("available") else "Unavailable", None),
        ("Recent order records", _number(len(orders)) if data.get("available") else "Unavailable", "Latest persisted sample" if data.get("available") else None),
        ("Recent fills", _number(len(fills)) if data.get("available") else "Unavailable", "Latest persisted sample" if data.get("available") else None),
        ("Run mode", _kind_label((data.get("run") or {}).get("kind")), None),
    ])
    st.markdown("#### Open simulated positions")
    _table(positions, empty="No persisted open positions.")
    st.markdown("#### Orders")
    _table(orders, empty="No persisted order records.")
    st.markdown("#### Fills")
    _table(fills, empty="No persisted fills.")
    if orders or fills:
        st.caption("Order/fill records are bounded recent samples. Reconciliation totals are not inferred from truncated samples.")
    else:
        _empty("Execution reconciliation is unavailable until both order and fill events have been persisted.")


def _alert_panel(alerts: Any) -> None:
    st.markdown("#### Monitoring alerts")
    if not isinstance(alerts, dict) or not alerts.get("available"):
        _empty("Alert monitor has not published a state snapshot. Monitoring capability is unavailable, not healthy.")
        if isinstance(alerts, dict) and alerts.get("capabilities"):
            st.json(alerts["capabilities"], expanded=False)
        return
    active = alerts.get("active") or []
    history = alerts.get("history") or []
    market = alerts.get("market_state") if isinstance(alerts.get("market_state"), dict) else {}
    feed = alerts.get("feed_assessment") if isinstance(alerts.get("feed_assessment"), dict) else {}
    st.markdown("#### CME session and feed health")
    st.caption(f"Market · {market.get('state', 'MARKET_UNKNOWN')} · {market.get('reason', 'calendar state unavailable')} · evaluated {_stamp(market.get('evaluated_at_utc'))}")
    st.caption(
        f"Provider · {feed.get('connection', 'UNKNOWN')} · "
        f"provider bar · {_stamp(feed.get('last_provider_bar_utc'))} · "
        f"Paper commit · {_stamp(feed.get('last_committed_bar_utc'))} · "
        f"expected delayed frontier · {_stamp(feed.get('expected_provider_frontier_utc'))} · "
        f"backlog · {_number(feed.get('backlog_bars'))} · "
        f"provider age · {_number(feed.get('provider_bar_age_seconds'))}s · "
        f"expected frontier age · {_number(feed.get('expected_frontier_lag_seconds'))}s"
    )
    if feed.get("connection_error_code_1100"):
        st.caption("IBKR API reported connectivity code 1100. This confirms a TWS/IB link loss; the code alone does not identify maintenance as the cause.")
    if feed.get("operator_reported_maintenance"):
        st.caption(f"Operator context · {feed['operator_reported_maintenance']}")
    if feed.get("last_error"):
        st.warning(f"Persisted provider error · {feed['last_error']}")
    services = alerts.get("services") if isinstance(alerts.get("services"), dict) else {}
    notifier = services.get("notification_service") or {}
    watchdog = services.get("process_watchdog") or {}
    st.caption(
        f"Notifier · {notifier.get('state', 'UNAVAILABLE')} · heartbeat age · "
        f"{_number(notifier.get('heartbeat_age_seconds'))}s · "
        f"watchdog · {watchdog.get('state', 'UNAVAILABLE')} · "
        f"process probe · {watchdog.get('process_probe', 'UNAVAILABLE')}"
    )
    if active:
        _table(alert_table_rows(active), empty="No active alerts.", height=260)
    else:
        st.caption("No active persisted alerts.")
    with st.expander("Recent alert transitions", expanded=False):
        _table(alert_table_rows(list(reversed(history[-100:]))), empty="No alert transition history.", height=300)
    st.caption(f"Last evaluation · {_stamp(alerts.get('last_evaluated_utc'))} · capabilities are per-field and may be unavailable.")
    if alerts.get("capabilities"):
        st.json(alerts["capabilities"], expanded=False)
    if alerts.get("thresholds"):
        with st.expander("Risk and monitoring thresholds", expanded=False):
            st.json(alerts["thresholds"], expanded=False)


def _system(data: dict[str, Any], api_port: int, token: str) -> None:
    with st.expander("Operational validation evidence", expanded=False):
        st.caption("Read-only bounded journal inspection. Market-session progression needs observations across time; a connected socket alone does not establish it.")
        if token and st.button("Inspect current operational evidence", key="operational_health_read"):
            evidence = _api_json(api_port, token, "/v1/operational-health")
            if evidence:
                st.json(evidence, expanded=False)
            else:
                st.info("Operational evidence is unavailable.")
    run = data.get("run") or {}
    status = data.get("status") or {}
    system = data.get("system") or {}
    feed = system.get("feed") or {}
    _status_line(run)
    _alert_panel(data.get("alerts") or (_api_json(api_port, token, "/v1/alerts") if token else {}))
    _metric_row([
        ("Mode", _kind_label(run.get("kind")), None),
        ("Engine state", str(run.get("status") or "Unavailable"), None),
        ("Processed bars", _number(run.get("processed_bars")), None),
        ("Last committed market bar", _stamp((status.get("system") or {}).get("last_bar") or run.get("last_processed_bar")), None),
    ])
    st.caption(f"Source run: {run.get('source_directory') or 'Unavailable'}")
    _scope_caption(run)
    if run.get("progress_pct") is None:
        _empty(run.get("progress_note") or "Run progress is unavailable; no reliable processed/eligible denominator is persisted.")
    hmm = system.get("hmm") or status.get("hmm") or {}
    st.markdown("#### Current HMM observation")
    if hmm:
        st.json(hmm, expanded=False)
    else:
        _empty("No HMM observation has been persisted.")
    st.markdown("#### Feed status")
    current_feed = (status.get("system") or {}).get("feed_health") or feed
    if current_feed:
        st.caption(f"Paper committed bar · {_stamp((status.get('system') or {}).get('last_bar'))} · provider frontier · {_stamp(current_feed.get('latest_available_bar'))} · backlog · {_number(current_feed.get('backlog_bars'))}")
        st.caption(
            f"IBKR · {current_feed.get('connection_state') or current_feed.get('mode') or 'UNAVAILABLE'} · "
            f"retry count · {current_feed.get('retry_count', 'unavailable')} · "
            f"next retry UTC · {_stamp(current_feed.get('next_retry_at_utc'))} · "
            f"last verified handshake UTC · {_stamp(current_feed.get('last_successful_handshake_utc'))}"
        )
        progress = current_feed.get("recovery_progress") or {}
        st.caption(
            f"Recovery · {'catch-up active' if progress.get('catchup_active') else 'catch-up idle/unavailable'} · "
            f"next backfill epoch · {progress.get('backfill_next_start_epoch_utc', 'unavailable')} · "
            f"target epoch · {progress.get('backfill_target_end_epoch_utc', 'unavailable')}"
        )
        if current_feed.get("last_error"):
            st.error(f"Latest feed error · {current_feed['last_error']}")
        st.json(current_feed, expanded=False)
    else:
        _empty("Feed state unavailable until market bars or feed events are persisted.")
    st.markdown("#### Recent refits and latest checkpoint")
    refits = _api_json(api_port, token, "/v1/refits")
    checkpoints = _api_json(api_port, token, "/v1/checkpoint")
    left, right = st.columns(2)
    with left:
        _table(refits if isinstance(refits, list) else [], empty="No model refit events persisted.", height=280)
    with right:
        _table([checkpoints] if isinstance(checkpoints, dict) and checkpoints else [], empty="No checkpoint event persisted.", height=280)
    st.markdown("#### Data and execution diagnostics")
    events = _api_json(api_port, token, "/v1/events")
    if isinstance(events, list):
        noteworthy = [e for e in events if any(token in str(e.get("event_type", "")).upper() for token in ("GAP", "DUPLICATE", "ERROR", "WARNING", "RECONCIL", "FEED"))]
        _table(noteworthy, empty="No recorded feed-gap, timestamp, execution or system diagnostic events in the returned event window.", height=300)
    else:
        _empty("System event history unavailable.")
    parity = _api_json(api_port, token, "/v1/parity")
    st.markdown("#### Shadow parity")
    _table(parity if isinstance(parity, list) else [], empty="No persisted shadow parity comparison.")
    st.markdown("#### Runtime warnings and errors")
    errors, warnings = system.get("errors") or [], system.get("warnings") or []
    if errors:
        for error in errors: st.error(str(error))
    if warnings:
        for warning in warnings: st.warning(str(warning))
    if not errors and not warnings:
        st.caption("No persisted error or warning records were returned.")


def _kind_label(kind: Any) -> str:
    return {
        "HISTORICAL_REPLAY": "Historical replay · simulated results",
        "DELAYED_PAPER": "Delayed Paper · simulated fills",
        "REALTIME_PAPER": "Realtime Paper · simulated fills",
    }.get(str(kind), "Unavailable")


def _effective_kind(option: RunOption, data: dict[str, Any]) -> str:
    if option.kind != "UNAVAILABLE":
        return option.kind
    run = data.get("run") or {}
    status = data.get("status") or {}
    source = (((status.get("system") or {}).get("feed_health") or {}).get("source")
              or status.get("source"))
    if status.get("mode") == "PAPER" and str(source or "").startswith("ibkr_delayed"):
        return "DELAYED_PAPER"
    return str(run.get("kind") or "UNAVAILABLE")


def _run_dashboard(page: str, option: RunOption) -> None:
    output = option.path
    api_port = int(os.environ.get("PAPER_MONITORING_API_PORT", "0"))
    try:
        _, token, port = _start_api(str(output), api_port)
        data = _api_json(port, token, "/v1/dashboard")
    except Exception as exc:
        LOGGER.exception("dashboard API failure page=%s run=%s", page, output.name)
        data = {"_error": f"Could not start local read-only API ({type(exc).__name__})"}
        token, port = "", api_port
    if data.get("_error"):
        st.error(data["_error"])
        st.caption("Local read-only monitoring is unavailable. Verify the selected Paper output directory and local port.")
        return
    if isinstance(data.get("run"), dict):
        data["run"]["kind"] = _effective_kind(option, data)
        data["run"]["source_directory"] = output.name
    # Read the sidecar snapshot directly as a compatibility fallback for an
    # already-running in-process API server created before /v1/alerts existed.
    alert_path = output.parent / ".notifications" / output.name / "alert_state.json"
    if alert_path.is_file():
        try:
            saved_alerts = json.loads(alert_path.read_text(encoding="utf-8"))
            public_alert = lambda row: {key: value for key, value in row.items() if not str(key).startswith("_")}
            runtime = alert_path.parent

            def sidecar_json(name: str) -> dict[str, Any]:
                try:
                    value = json.loads((runtime / name).read_text(encoding="utf-8"))
                    return value if isinstance(value, dict) else {}
                except (OSError, json.JSONDecodeError, TypeError):
                    return {}

            current_time = datetime.now(timezone.utc)

            def sidecar_age(value: Any) -> float | None:
                try:
                    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    return max(0.0, (current_time - stamp.astimezone(timezone.utc)).total_seconds())
                except (TypeError, ValueError):
                    return None

            notifier_state = sidecar_json("notifier_heartbeat.json")
            notifier_age = sidecar_age(notifier_state.get("timestamp_utc"))
            watchdog_state = sidecar_json("engine_watchdog_state.json")
            watchdog_age = sidecar_age(watchdog_state.get("last_probe_at_utc"))
            services = {
                "notification_service": {"state": "UNAVAILABLE" if notifier_age is None else ("HEALTHY" if notifier_age < 30 else "STALE"),
                    "pid": notifier_state.get("pid"), "heartbeat_utc": notifier_state.get("timestamp_utc"),
                    "heartbeat_age_seconds": notifier_age, "stale_after_seconds": 30},
                "process_watchdog": {"state": "UNAVAILABLE" if watchdog_age is None else ("HEALTHY" if watchdog_age < 45 else "STALE"),
                    "last_probe_utc": watchdog_state.get("last_probe_at_utc"), "last_probe_age_seconds": watchdog_age,
                    "process_probe": watchdog_state.get("last_probe_status", "UNAVAILABLE"), "probe_interval_seconds": 15},
            }
            data["alerts"] = {"available": True, "active": [public_alert(row) for row in (saved_alerts.get("active") or {}).values()],
                              "history": [public_alert(row) for row in (saved_alerts.get("history") or [])[-100:]],
                              "capabilities": saved_alerts.get("capabilities", {}),
                              "services": services,
                              "last_evaluated_utc": saved_alerts.get("last_evaluated_utc"),
                              "market_state": saved_alerts.get("market_state", {"state": "MARKET_UNKNOWN", "reason": "not_evaluated"}),
                              "feed_assessment": saved_alerts.get("feed_assessment", {"state": "UNAVAILABLE"}),
                              "thresholds": saved_alerts.get("thresholds", {})}
        except (OSError, json.JSONDecodeError, TypeError):
            data["alerts"] = {"available": False, "capabilities": {"state": "CORRUPT_OR_UNREADABLE"}}
    if not data.get("available"):
        _empty("No Paper analytics database is available yet. Persisted metrics will appear when the run writes them; no activity or P&L is estimated.")
    try:
        if page == "Command Center":
            _command_center(data)
        elif page == "Performance":
            _performance(data, port, token)
        elif page == "Analytics":
            research, research_error = _validated_research(str(ROOT))
            render_analytics(data, lambda route: _api_json(port, token, route), output,
                             research, research_error)
        elif page == "Strategies":
            _strategies(data, port, token)
        elif page == "Trade Explorer":
            _trades(port, token)
        elif page == "Positions & Orders":
            _execution(data)
        elif page == "Risk Monitor":
            _risk(data, port, token)
        else:
            _system(data, port, token)
    except Exception as exc:
        LOGGER.exception("dashboard render failure page=%s run=%s", page, output.name)
        st.error(f"This dashboard page could not render ({type(exc).__name__}). The traceback is saved to the local dashboard log.")
    st.markdown("---")
    st.caption("PAPER ONLY · authenticated loopback API · read-only interface · historical replay is not live market trading")


st.sidebar.markdown('<div class="mnq-eyebrow">MNQ QUANT SYSTEM</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Paper operations")
selected_page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
run_root = ROOT / "results" / "paper"
preferred_text = os.environ.get("PAPER_MONITOR_DEFAULT_RUN") or os.environ.get("PAPER_MONITOR_OUTPUT_DIR")
preferred_run = Path(preferred_text).resolve() if preferred_text else None
run_options = discover_runs(run_root, preferred=preferred_run)
if run_options:
    initial = default_run(run_options, preferred_run)
    option_by_path = {str(option.path): option for option in run_options}
    paths = list(option_by_path)
    choice = st.sidebar.selectbox(
        "Paper run", paths,
        index=paths.index(str(initial.path)) if initial else 0,
        format_func=lambda path: option_by_path[path].label,
        key="paper_run_selection",
    )
    selected_run = option_by_path[choice]
    st.sidebar.caption("Local monitoring · read-only · no order entry")
    st.sidebar.caption(f"Selected source · `{selected_run.path.name}`")
    st.sidebar.caption(f"Classification · {selected_run.kind.replace('_', ' ')}")
else:
    selected_run = None
    st.sidebar.error("No persisted Paper runs were found under results/paper.")

st.markdown('<div class="mnq-eyebrow">PAPER MONITORING / MNQ</div>', unsafe_allow_html=True)
st.title(selected_page)
st.caption("Persisted engine facts only. Unrecorded metrics remain unavailable.")


@st.fragment(run_every="30s")
def auto_refresh_dashboard(page: str) -> None:
    if selected_run is None:
        _empty("Create or select a persisted Paper run directory to monitor it.")
    else:
        _run_dashboard(page, selected_run)


auto_refresh_dashboard(selected_page)
