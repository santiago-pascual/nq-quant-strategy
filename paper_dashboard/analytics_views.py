"""Interactive Research-to-Paper analytics views (read-only)."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from paper_dashboard.forward_diagnostics import forward_comparison
from paper_dashboard.historical_extension import load_extension, source_timeline

from paper_dashboard.research_analytics import (
    STRATEGIES, cumulative_r_series, equity_drawdown_r_series,
    moving_block_expectancy_band, reconstruct_paper_outcomes,
    rolling_metrics, summarize_r,
)

BG, SURFACE, GRID = "#0b0f14", "#111820", "#27333e"
TEXT, MUTED, GREEN, RED, AMBER, BLUE = "#e6ebef", "#93a0ad", "#67c6a3", "#d77f87", "#d4b574", "#7f9fb4"
INITIAL_LIVE_DATE = "2026-06-19T23:59:59.999999+00:00"


@st.cache_data(ttl=300, show_spinner=False)
def _extension_data(root: str) -> dict:
    return load_extension(Path(root))


def _parse(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def _safe_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "delayed_paper_run.json").read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("mode") == "DELAYED_IBKR_PAPER" and value.get("paper_only") is True and value.get("orders_enabled") is False:
            return value
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        pass
    return {}


def _filter_rows(rows: list[dict[str, Any]], strategy: str | None,
                 start: date | None, end: date | None, time_key: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if strategy and row.get("strategy") != strategy:
            continue
        stamp = _parse(row.get(time_key))
        if start and (stamp is None or stamp.date() < start):
            continue
        if end and (stamp is None or stamp.date() > end):
            continue
        selected.append(row)
    return selected


def _metrics(values: list[float]) -> dict[str, Any]:
    return summarize_r(values)


def _paper_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r = _metrics([float(row["realized_r"]) for row in rows if row.get("realized_r") is not None])
    pnl = [float(row["net_pnl"]) for row in rows if row.get("net_pnl") is not None]
    gains = sum(x for x in pnl if x > 0)
    losses = abs(sum(x for x in pnl if x < 0))
    r.update({"net_pnl_usd": sum(pnl), "profit_factor_usd": gains / losses if losses else None,
              "closed_trades": len(pnl), "total_costs_usd": sum(float(row.get("total_costs") or 0) for row in rows)})
    return r


def _r_daily(rows: list[dict[str, Any]], *, strategy: str | None = None) -> list[dict[str, Any]]:
    groups: dict[str, float] = defaultdict(float)
    for row in rows:
        if strategy and row.get("strategy") != strategy:
            continue
        stamp = _parse(row.get("exit_timestamp_utc"))
        if stamp is None:
            continue
        try:
            value = float(row.get("realized_r"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            groups[stamp.date().isoformat()] += value
    result = []
    cumulative = 0.0
    peak = 0.0
    for day, value in sorted(groups.items()):
        cumulative += value
        peak = max(peak, cumulative)
        result.append({"timestamp_utc": f"{day}T00:00:00+00:00", "daily_r": value,
                       "cumulative_r": cumulative, "drawdown_r": cumulative - peak})
    return result


def _period_rows(daily: list[dict[str, Any]], period: str) -> list[dict[str, Any]]:
    grouped: dict[str, float] = defaultdict(float)
    for row in daily:
        stamp = _parse(row.get("timestamp_utc"))
        if stamp is None:
            continue
        if period == "month":
            key = stamp.strftime("%Y-%m")
        else:
            iso = stamp.isocalendar()
            key = f"{iso.year}-W{iso.week:02d}"
        grouped[key] += float(row.get("daily_r", 0))
    return [{"period": key, "r": value} for key, value in sorted(grouped.items())]


def _figure_layout(fig: go.Figure, title: str | None = None, *, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=title, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": MUTED, "family": "Inter, Segoe UI, sans-serif", "size": 11},
        margin={"l": 52, "r": 22, "t": 48 if title else 22, "b": 42}, height=height,
        hovermode="x unified", showlegend=legend,
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10}},
        xaxis={"showgrid": False, "linecolor": GRID, "tickfont": {"color": MUTED}, "rangeslider": {"visible": False}},
        yaxis={"showgrid": True, "gridcolor": "#222d37", "zerolinecolor": GRID,
               "tickfont": {"color": MUTED}, "automargin": True},
    )
    return fig


def _line(points: list[dict[str, Any]], x: str, y: str, name: str, color: str,
          *, dash: str = "solid", width: float = 2.0, connect: bool = False) -> go.Scatter:
    return go.Scatter(x=[row[x] for row in points], y=[row[y] for row in points],
                      mode="lines", name=name, connectgaps=connect,
                      line={"color": color, "width": width, "dash": dash},
                      hovertemplate="%{x}<br>%{y:,.3f}<extra>%{fullData.name}</extra>")


def _metric_cards(items: list[tuple[str, Any, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col.container(border=True):
            st.caption(label)
            col.markdown(f"<div style='font:500 1.25rem ui-monospace,Consolas,monospace;color:{TEXT};font-variant-numeric:tabular-nums'>{value}</div>", unsafe_allow_html=True)
            col.caption(note)


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "Unavailable"


def _unified_curve(research_daily: list[dict[str, Any]], paper_rows: list[dict[str, Any]],
                   activation: str | None, paper_truncated: bool, extension: dict | None = None) -> go.Figure:
    research_curve = equity_drawdown_r_series(research_daily)
    paper_curve = cumulative_r_series(paper_rows, value_key="realized_r",
                                      time_key="exit_timestamp_utc")
    fig = go.Figure()
    research_endpoint = 0.0
    if research_curve:
        fig.add_trace(_line(research_curve, "timestamp_utc", "cumulative_r", "Validated Research OOS · cumulative R", BLUE))
        research_endpoint = float(research_curve[-1]["cumulative_r"])
    if extension and extension.get('available'):
        extension_curve = cumulative_r_series(extension['trades'], value_key='realized_r', time_key='exit_timestamp_utc')
        fig.add_trace(_line(extension_curve, 'timestamp_utc', 'cumulative_r',
                            'Historical causal simulation extension · independent net R', AMBER))
    if paper_curve:
        if activation and not paper_truncated:
            paper_curve = [{"timestamp_utc": activation, "cumulative_r": 0.0}, *paper_curve]
        # The Paper trace is rebased to the Research endpoint only for visual
        # comparison in the shared R axis. No returns are inserted in the gap.
        paper_curve = [{**row, "cumulative_r": research_endpoint + float(row["cumulative_r"])}
                       for row in paper_curve]
        fig.add_trace(_line(paper_curve, "timestamp_utc", "cumulative_r", "Causal delayed Paper · cumulative net R", GREEN))
    else:
        st.caption("No persisted closed Paper trades with realized R yet; the Paper segment is currently unavailable.")
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Future LIVE · disabled",
                             line={"color": AMBER, "dash": "dash", "width": 2}, showlegend=True))
    research_end = "2026-06-19T23:59:59.999999+00:00"
    if activation:
        fig.add_vline(x=research_end, line_color=BLUE, line_dash="dot", annotation_text="Research OOS end",
                      annotation_position="top left")
        fig.add_vline(x=activation, line_color=GREEN, line_dash="dot", annotation_text="Paper activation",
                      annotation_position="top right")
        fig.add_vrect(x0="2026-06-20T00:00:00+00:00", x1=activation,
                      fillcolor="rgba(147,160,173,.10)", line_width=0,
                      annotation_text="Separate historical simulation" if extension and extension.get('available') else "Historical bars available · simulation pending",
                      annotation_position="top")
    fig.update_layout(yaxis_title="Cumulative risk units · R", legend={"orientation":"h","y":1.18,"x":0})
    return _figure_layout(fig, "Research-to-Paper · Paper rebased to Research endpoint for display", height=430)


def _overview(research: dict[str, Any], research_rows: list[dict[str, Any]], paper_rows: list[dict[str, Any]],
              paper_metrics: dict[str, Any], data: dict[str, Any], activation: str | None,
              paper_truncated: bool, extension: dict | None = None) -> None:
    report = research["report"]
    expected = report["portfolio_metrics"]
    status = (data.get("status") or {}).get("system") or {}
    account = data.get("account") or {}
    _metric_cards([
        ("Research OOS trades", f"{expected['trades']:,}", "Frozen validated portfolio"),
        ("Research cumulative R", f"{expected['total_R']:+.2f}R", "Daily artifact reconciles to trade ledger"),
        ("Paper closed trades", f"{paper_metrics.get('closed_trades',0):,}", "Persisted SQLite records only"),
        ("Paper net P&L", f"${paper_metrics.get('net_pnl_usd',0):+,.2f}" if paper_metrics.get("closed_trades") else "Unavailable", "Actual persisted Paper net USD"),
    ])
    fig = _unified_curve(research["daily"], paper_rows, activation, paper_truncated, extension)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False})
    st.caption("Comparable view uses frozen Research `r_multiple` and Paper net `realized_r` (net P&L / stored initial risk). The Paper line is offset by the Research endpoint only to share a readable axis; it is not a continuous return chain. Any validated historical extension is shown independently from zero; otherwise the transition interval stays blank. Execution and cost assumptions differ; this is a risk-normalized diagnostic, not a parity claim.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Native historical Research units")
        fig = go.Figure(_line(equity_drawdown_r_series(research["daily"]), "timestamp_utc", "cumulative_r", "Research cumulative R", BLUE))
        st.plotly_chart(_figure_layout(fig, height=290), width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("#### Native Paper account equity · USD")
        snapshots = data.get("equity_curve") or []
        points = [row for row in snapshots if row.get("timestamp_utc") and row.get("equity") is not None]
        if points:
            fig = go.Figure(_line(points, "timestamp_utc", "equity", "Persisted Paper account equity · USD", GREEN))
            st.plotly_chart(_figure_layout(fig, height=290), width="stretch", config={"displayModeBar": False})
        else:
            st.info("Paper account snapshots are unavailable. No USD curve is inferred from Research R.")
    scope = data.get("run") or {}
    st.caption(f"Research OOS: 2020-06-23 through 2026-06-19 · Paper activation: {activation or 'Unavailable'} · Paper engine status: {status.get('state','Unavailable')} · balance/equity: {account.get('equity','Unavailable')}")
    if paper_truncated:
        st.warning("Paper trade analytics are truncated to the API’s latest 5,000 closed outcomes; cumulative Paper R begins at the included sample and is not lifetime cumulative.")
    _trade_distributions(research_rows, paper_rows)


def _trade_distributions(research_rows: list[dict[str, Any]], paper_rows: list[dict[str, Any]]) -> None:
    st.markdown("#### Trade outcome distributions · native units")
    left, right = st.columns(2)
    research_values = [float(row["r_multiple"]) for row in research_rows]
    paper_usd = [float(row["net_pnl"]) for row in paper_rows if row.get("net_pnl") is not None]
    with left:
        if research_values:
            fig = go.Figure(go.Histogram(x=research_values, nbinsx=45, marker_color=BLUE,
                                         hovertemplate="Research R %{x:.3f}<br>Trades %{y}<extra></extra>"))
            st.plotly_chart(_figure_layout(fig, "Validated Research trade outcomes · R", height=300, legend=False),
                            width="stretch")
        else:
            st.info("Research trade outcomes unavailable for this filter.")
    with right:
        if paper_usd:
            fig = go.Figure(go.Histogram(x=paper_usd, nbinsx=min(30, max(5, len(paper_usd))), marker_color=GREEN,
                                         hovertemplate="Paper net P&L $%{x:.2f}<br>Trades %{y}<extra></extra>"))
            st.plotly_chart(_figure_layout(fig, "Persisted Paper trade outcomes · net USD", height=300, legend=False),
                            width="stretch")
        else:
            st.info("Paper P&L distribution requires persisted closed trades.")
    durations = [float(row["duration_seconds"]) / 60 for row in paper_rows
                 if row.get("duration_seconds") is not None]
    if durations:
        fig = go.Figure(go.Histogram(x=durations, nbinsx=min(30, max(5, len(durations))), marker_color=AMBER,
                                     hovertemplate="Duration %{x:.1f} minutes<br>Trades %{y}<extra></extra>"))
        st.plotly_chart(_figure_layout(fig, "Paper closed-trade duration · minutes", height=280, legend=False),
                        width="stretch")
    else:
        st.caption("Paper trade-duration distribution unavailable: no persisted duration values in the selected outcomes.")


def _equity_drawdown(research: dict[str, Any], paper_rows: list[dict[str, Any]], data: dict[str, Any],
                     strategy: str | None) -> None:
    research_daily = research["daily"]
    if strategy:
        # The daily benchmark is a portfolio total; strategy-specific daily
        # realizations use only ledger rows with a known exit timestamp.
        rrows = [row for row in research["trades"] if row.get("strategy") == strategy and row.get("exit_timestamp_utc")]
        daily = _r_daily_from_research(rrows)
        st.caption("Strategy-specific Research time series uses only trades with a recorded exit timestamp; missing exits are not assigned to entry dates.")
    else:
        daily = research_daily
    if strategy:
        paper_daily = _r_daily(paper_rows, strategy=strategy)
    else:
        paper_daily = _r_daily(paper_rows)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=("Cumulative realized R", "Underwater drawdown · R"))
    if daily:
        curve = equity_drawdown_r_series(daily)
        fig.add_trace(_line(curve, "timestamp_utc", "cumulative_r", "Research", BLUE), row=1, col=1)
        fig.add_trace(_line(curve, "timestamp_utc", "drawdown_r", "Research DD", BLUE), row=2, col=1)
    if paper_daily:
        fig.add_trace(_line(paper_daily, "timestamp_utc", "cumulative_r", "Paper net R", GREEN), row=1, col=1)
        fig.add_trace(_line(paper_daily, "timestamp_utc", "drawdown_r", "Paper DD R", GREEN), row=2, col=1)
    fig.update_yaxes(title_text="R", row=1, col=1); fig.update_yaxes(title_text="R below peak", row=2, col=1)
    st.plotly_chart(_figure_layout(fig, height=550), width="stretch", config={"displayModeBar": True, "scrollZoom": True})
    native = data.get("equity_curve") or []
    if native:
        fig = go.Figure(_line(native, "timestamp_utc", "drawdown", "Persisted Paper account drawdown · USD", RED))
        st.plotly_chart(_figure_layout(fig, "Paper account drawdown · native USD", height=300), width="stretch")
    else:
        st.info("Paper account drawdown snapshots are not available.")
    win = st.selectbox("Rolling trade window", [25, 50, 100], index=0, key="analytics_roll_window")
    research_trades = research["trades"] if not strategy else [row for row in research["trades"] if row["strategy"] == strategy]
    paper_trades = paper_rows if not strategy else [row for row in paper_rows if row.get("strategy") == strategy]
    research_roll = rolling_metrics(research_trades, value_key="r_multiple", time_key="entry_timestamp_utc", window=win)
    paper_roll = rolling_metrics(paper_trades, value_key="realized_r", time_key="exit_timestamp_utc", window=win)
    for title, field, label in (("Rolling expectancy", "expectancy_r", "Mean R"),
                                ("Rolling profit factor", "profit_factor", "Profit factor"),
                                ("Rolling win rate", "win_rate", "Win rate")):
        fig = go.Figure()
        if research_roll:
            fig.add_trace(_line(research_roll, "timestamp_utc", field, f"Research · {win} trades", BLUE))
        if paper_roll:
            fig.add_trace(_line(paper_roll, "timestamp_utc", field, f"Paper · {win} trades", GREEN))
        if research_roll or paper_roll:
            fig.update_yaxes(title_text=label)
            st.plotly_chart(_figure_layout(fig, title, height=280), width="stretch", config={"displayModeBar": True})
        elif title == "Rolling expectancy":
            st.info(f"Rolling statistics need {win} closed outcomes in each selected series. Current Paper sample: {len(paper_trades)}.")


def _r_daily_from_research(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, float] = defaultdict(float)
    for row in rows:
        stamp = _parse(row.get("exit_timestamp_utc"))
        if stamp:
            groups[stamp.date().isoformat()] += float(row["r_multiple"])
    return [{"timestamp_utc": f"{day}T00:00:00+00:00", "daily_r": val} for day, val in sorted(groups.items())]


def _strategy_breakdown(research: dict[str, Any], paper_rows: list[dict[str, Any]],
                        strategy: str | None) -> None:
    research_groups = {name: [r for r in research["trades"] if r["strategy"] == name] for name in STRATEGIES}
    paper_groups = {name: [r for r in paper_rows if r.get("strategy") == name] for name in STRATEGIES}
    rows = []
    contrib = go.Figure()
    for name in STRATEGIES:
        r_values = [float(row["r_multiple"]) for row in research_groups[name]]
        p_values = [float(row["realized_r"]) for row in paper_groups[name] if row.get("realized_r") is not None]
        rm, pm = _metrics(r_values), _metrics(p_values)
        paper_usd = sum(float(row["net_pnl"]) for row in paper_groups[name] if row.get("net_pnl") is not None)
        rows.append({"Strategy": name, "Research trades": rm["trades"], "Research total R": rm["total_r"],
                     "Research expectancy R": rm["expectancy_r"], "Research PF": rm["profit_factor"],
                     "Paper closed": pm["trades"], "Paper net R": pm["total_r"], "Paper net USD": paper_usd})
        contrib.add_bar(name=name, x=[name], y=[rm["total_r"]], marker_color=BLUE)
        if pm["trades"]:
            contrib.add_bar(name=name + " · Paper", x=[name], y=[pm["total_r"]], marker_color=GREEN)
    st.dataframe(rows, width="stretch", hide_index=True)
    contrib.update_layout(barmode="group", yaxis_title="Total R", showlegend=True)
    st.plotly_chart(_figure_layout(contrib, "Strategy contribution · risk-normalized R", height=340), width="stretch")

    st.markdown("#### Strategy equity paths")
    fig = go.Figure()
    for name, color in zip(STRATEGIES, (BLUE, AMBER, "#b4a0cf", "#a9c6b4")):
        research_exits = [row for row in research_groups[name] if row.get("exit_timestamp_utc")]
        curve = cumulative_r_series(research_exits, value_key="r_multiple", time_key="exit_timestamp_utc")
        if curve:
            fig.add_trace(_line(curve, "timestamp_utc", "cumulative_r", f"{name} Research", color))
        paper_curve = cumulative_r_series(paper_groups[name], value_key="realized_r", time_key="exit_timestamp_utc")
        if paper_curve:
            fig.add_trace(_line(paper_curve, "timestamp_utc", "cumulative_r", f"{name} Paper", GREEN, dash="dash"))
    if fig.data:
        st.plotly_chart(_figure_layout(fig, "Cumulative R by strategy · known exit times", height=400), width="stretch")
    else:
        st.info("No exit-timestamped strategy outcomes are available for this filter.")
    st.caption("Research benchmark contains no persisted HMM-state or volatility-regime columns. Historical regime attribution is unavailable; Paper attribution uses raw persisted state IDs only.")
    hmm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paper_rows:
        hmm_groups[str(row.get("hmm_state") if row.get("hmm_state") is not None else "UNAVAILABLE")].append(row)
    hmm_table = []
    for state, group in sorted(hmm_groups.items()):
        metrics = _paper_metrics(group)
        hmm_table.append({"Persisted raw HMM state": state, "Trades": metrics["closed_trades"],
                          "Net P&L USD": metrics["net_pnl_usd"], "Net R": metrics["total_r"]})
    st.markdown("#### Paper outcomes by raw HMM state")
    if hmm_table:
        st.dataframe(hmm_table, width="stretch", hide_index=True)
        st.caption("Raw component IDs are shown without semantic relabeling. No state is inferred when the persisted trade field is empty.")
    else:
        st.info("Paper HMM attribution unavailable: no persisted closed trade with a state ID.")


def _alpha_decay(research: dict[str, Any], paper_rows: list[dict[str, Any]], strategy: str | None,
                 sample_truncated: bool) -> None:
    rrows = [row for row in research["trades"] if not strategy or row["strategy"] == strategy]
    prows = [row for row in paper_rows if not strategy or row.get("strategy") == strategy]
    research_values = [float(row["r_multiple"]) for row in rrows]
    paper_values = [float(row["realized_r"]) for row in prows if row.get("realized_r") is not None]
    window = st.selectbox("Forward evaluation window", [25, 50, 100], key="alpha_window")
    diagnostic = forward_comparison(research["trades"], paper_rows, window=window)
    group = (diagnostic.get("groups") or {}).get(strategy or "PORTFOLIO", {})
    st.caption(f"Diagnostic state · {group.get('state', 'UNAVAILABLE')} · "
               "Descriptive only; sequential significance and automatic trading actions are disabled.")
    band = moving_block_expectancy_band(research_values, window=window,
                                       block_length=max(2, min(10, window // 5)), samples=2000)
    research_stat, paper_stat = _metrics(research_values), _metrics(paper_values)
    st.dataframe([
        {"Series": "Validated Research OOS", "Closed trades": research_stat["trades"], "Total R": research_stat["total_r"],
         "Expectancy R": research_stat["expectancy_r"], "PF · R": research_stat["profit_factor"], "Win rate": research_stat["win_rate"]},
        {"Series": "Causal delayed Paper", "Closed trades": paper_stat["trades"], "Total net R": paper_stat["total_r"],
         "Expectancy net R": paper_stat["expectancy_r"], "PF · R": paper_stat["profit_factor"], "Win rate": paper_stat["win_rate"]},
    ], width="stretch", hide_index=True)
    if sample_truncated:
        st.warning("Paper outcomes are truncated. Alpha comparison uses only the most recent API sample and is not a lifetime comparison.")
    if not band.get("available"):
        st.info(band.get("reason", "Research reference band unavailable."))
        return
    st.caption(f"Research reference: {window}-trade rolling mean R, moving-block bootstrap (block={band['block_length']}, 2,000 draws, fixed seed); 95% empirical range {band['lower']:+.3f}R to {band['upper']:+.3f}R. Blocks retain short-range serial dependence. This is a descriptive reference band, not an adjusted hypothesis test.")
    if len(paper_values) < window:
        st.warning(f"Insufficient forward sample: {len(paper_values)} of {window} closed Paper trades. No alpha-decay conclusion is available.")
        return
    rolling = rolling_metrics(prows, value_key="realized_r", time_key="exit_timestamp_utc", window=window)
    fig = go.Figure()
    if rolling:
        fig.add_trace(_line(rolling, "timestamp_utc", "expectancy_r", f"Paper rolling expectancy · {window}", GREEN))
        fig.add_hrect(y0=band["lower"], y1=band["upper"], fillcolor="rgba(127,159,180,.13)", line_width=0,
                      annotation_text="Research bootstrap reference range", annotation_position="top left")
        fig.add_hline(y=band["lower"], line_color=BLUE, line_dash="dot")
        fig.add_hline(y=band["upper"], line_color=BLUE, line_dash="dot")
        fig.add_hline(y=0, line_color=GRID, line_dash="dash")
        st.plotly_chart(_figure_layout(fig, "Forward Paper expectancy against Research reference", height=380), width="stretch")
    st.caption("Execution and cost assumptions differ. A drawdown or isolated underperformance is not, by itself, evidence of alpha decay. Statistical results are separate from operational/feed incidents.")


def _execution_quality(data: dict[str, Any], paper_rows: list[dict[str, Any]], paper_truncated: bool) -> None:
    fills = data.get("fills") or []
    modeled = []
    for row in fills:
        value = row.get("artificial_slippage_ticks")
        try:
            if value is not None and math.isfinite(float(value)):
                modeled.append(float(value))
        except (TypeError, ValueError):
            continue
    _metric_cards([
        ("Persisted fills", str(len(fills)), "Dashboard API provides latest 100 fill records"),
        ("Modeled slippage ticks", _fmt(sum(modeled), 2, " ticks") if modeled else "Unavailable", "Explicit simulator tick field; not observed broker slippage"),
        ("Observed spread", "Unavailable", "No bid/ask spread persisted on these fills"),
        ("Signal-to-order latency", "Unavailable", "Signal/order timestamps are absent on current closed trade"),
    ])
    if modeled:
        fig = go.Figure(go.Histogram(x=modeled, nbinsx=20, marker_color=BLUE,
                                     hovertemplate="Modeled slippage %{x:.2f} ticks<br>Fills %{y}<extra></extra>"))
        st.plotly_chart(_figure_layout(fig, "Modeled simulator slippage · ticks", height=300), width="stretch")
        st.caption("This chart reports the explicitly named persisted simulator slippage tick parameter. It is not an estimate of market slippage or IBKR execution quality.")
    else:
        st.info("No modeled slippage observations are available.")
    latencies = []
    for row in paper_rows:
        signal, order, fill = (_parse(row.get(key)) for key in ("signal_timestamp_utc", "order_timestamp_utc", "entry_timestamp_utc"))
        if order and fill and order <= fill:
            latencies.append({"signal_to_order_seconds": (order-signal).total_seconds() if signal and signal <= order else None,
                              "order_to_fill_seconds": (fill-order).total_seconds(), "fill_time": fill.isoformat()})
    if latencies:
        st.dataframe(latencies, width="stretch", hide_index=True)
    else:
        st.info("Signal-to-order, order-to-fill, acquisition, and acquisition-to-commit latency are unavailable from the persisted fields for this run.")
    st.caption(f"Broker acknowledgment/fill latency is disabled: this system uses simulated fills only. {('Trade sample is truncated.' if paper_truncated else 'Only persisted closed trades are included.')}")


def _period_chart(research_daily: list[dict[str, Any]], paper_rows: list[dict[str, Any]], period: str) -> None:
    research_periods = _period_rows(research_daily, period)
    paper_periods = _period_rows(_r_daily(paper_rows), period)
    keys = sorted({row["period"] for row in research_periods} | {row["period"] for row in paper_periods})
    rmap = {row["period"]: row["r"] for row in research_periods}
    pmap = {row["period"]: row["r"] for row in paper_periods}
    if not keys:
        st.info("No period outcomes available.")
        return
    fig = go.Figure()
    fig.add_bar(x=keys, y=[rmap.get(key) for key in keys], name="Research daily R", marker_color=BLUE)
    fig.add_bar(x=keys, y=[pmap.get(key) for key in keys], name="Paper net R", marker_color=GREEN)
    fig.update_layout(barmode="group", yaxis_title="R")
    st.plotly_chart(_figure_layout(fig, f"{period.title()} realized performance · R", height=360), width="stretch")
    st.caption("Only periods with actual recorded trade outcomes appear. No zero-return dates or missing bars are synthesized.")


def render_analytics(data: dict[str, Any], api_json: Callable[[str], Any], output_dir: Path,
                     research: dict[str, Any] | None, research_error: str | None) -> None:
    st.markdown('<div class="mnq-eyebrow">RESEARCH / FORWARD ANALYTICS</div>', unsafe_allow_html=True)
    st.caption("Read-only · source data are selected-run SQLite and the frozen independent Research OOS artifacts. No Paper account or execution state is changed.")
    if not research or not research.get("available"):
        st.error(f"Validated Research reference unavailable: {research_error or 'integrity validation failed'}. Research comparison is disabled rather than substituting another replay artifact.")
        return
    response = api_json("/v1/analytics?limit=5000")
    if isinstance(response, dict) and (response.get("_error") or response.get("_unavailable")):
        st.warning(response.get("_error") or response.get("_unavailable"))
        paper_analytics: dict[str, Any] = {}
    else:
        paper_analytics = response if isinstance(response, dict) else {}
    paper_result = reconstruct_paper_outcomes(paper_analytics.get("trade_outcomes"))
    paper_rows = paper_result.get("trades", []) if paper_result.get("available") else []
    paper_truncated = bool((paper_analytics.get("sample") or {}).get("truncated"))
    run_manifest = _safe_manifest(output_dir)
    activation = run_manifest.get("activation_timestamp_utc")
    root = Path(__file__).resolve().parents[1]
    extension = _extension_data(str(root))
    with st.expander('Data coverage and model timeline', expanded=True):
        st.dataframe(source_timeline(root, activation), hide_index=True, width='stretch')
        if not extension.get('available'):
            st.info('June 20–October 7: local historical bars exist, but performance is unavailable until an activation-specific past-only seed and reviewed session coverage validate. October Paper state is never used retroactively.')
        else:
            st.caption('Validated historical causal simulation is a separate account/curve, not forward Paper and not an extension of the frozen Research certification.')
    if not activation:
        st.warning("The selected run has no valid delayed-Paper manifest activation timestamp. The transition marker and normalized Paper baseline are omitted.")

    chosen = st.selectbox("Analytics view", ["Overview", "Equity & Drawdown", "Strategy Breakdown", "Alpha Decay", "Execution Quality"],
                          key="research_paper_analytics_view")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    strategy_choice = c1.selectbox("Strategy", ["All", *STRATEGIES], key="research_paper_strategy")
    start = c2.date_input("Entry date from · UTC", value=None, key="research_paper_start")
    end = c3.date_input("Entry date through · UTC", value=None, key="research_paper_end")
    if start and end and start > end:
        st.error("Start date must be on or before end date.")
        return
    strategy = None if strategy_choice == "All" else strategy_choice
    extension = {**extension, 'trades': _filter_rows(extension.get('trades', []), strategy, start, end, 'entry_timestamp_utc')}
    research_trades = _filter_rows(research["trades"], strategy, start, end, "entry_timestamp_utc")
    paper_filtered = _filter_rows(paper_rows, strategy, start, end, "entry_timestamp_utc")
    paper_metrics = _paper_metrics(paper_filtered)
    research_daily = research["daily"]
    if start or end:
        research_daily = [row for row in research_daily if (not start or (_parse(row["timestamp_utc"]) or datetime.min.replace(tzinfo=timezone.utc)).date() >= start)
                          and (not end or (_parse(row["timestamp_utc"]) or datetime.max.replace(tzinfo=timezone.utc)).date() <= end)]
    if strategy:
        st.caption("Portfolio daily Research R remains a portfolio series; strategy-filtered curves use only trades with known exit timestamps.")
    st.caption(f"Research source · {research['provenance']['model_version']} · {research['provenance']['trade_rows']:,} trades · SHA-256 `{research['provenance']['trades_sha256'][:16]}…` · Paper outcomes {len(paper_filtered)}" + (" · Paper sample truncated" if paper_truncated else ""))

    if chosen == "Overview":
        _overview(research, research_trades, paper_filtered, paper_metrics, data, activation, paper_truncated, extension)
        _period_chart(research_daily, paper_filtered, "month")
    elif chosen == "Equity & Drawdown":
        _equity_drawdown(research, paper_filtered, data, strategy)
        period = st.radio("Period aggregation", ["week", "month"], horizontal=True, key="period_granularity")
        _period_chart(research_daily, paper_filtered, period)
    elif chosen == "Strategy Breakdown":
        _strategy_breakdown(research, paper_filtered, strategy)
        left, right = st.columns(2)
        with left:
            _period_chart(research_daily, paper_filtered, "month")
        with right:
            _period_chart(research_daily, paper_filtered, "week")
    elif chosen == "Alpha Decay":
        if not paper_result.get("available"):
            st.info("Paper trade outcomes unavailable; statistical comparison is not reported as healthy or zero.")
        else:
            _alpha_decay(research, paper_filtered, strategy, paper_truncated)
    else:
        _execution_quality(data, paper_filtered, paper_truncated)
    if extension.get('available'):
        st.markdown('#### Historical causal simulation extension · separate account')
        extension_metrics = _paper_metrics(extension['trades'])
        _metric_cards([
            ('Extension closed trades', str(extension_metrics['closed_trades']), 'Filtered completed simulation only'),
            ('Extension net USD', _fmt(extension_metrics['net_pnl_usd'], suffix=' USD'), 'Includes the persisted cost profile'),
            ('Extension net R', _fmt(extension_metrics['total_r'], suffix=' R'), 'Independent baseline; not forward Paper')])
        extension_curve = cumulative_r_series(extension['trades'], value_key='realized_r', time_key='exit_timestamp_utc')
        st.plotly_chart(_figure_layout(go.Figure(_line(extension_curve, 'timestamp_utc', 'cumulative_r',
            'Historical causal simulation · net R', AMBER))), width='stretch')
    st.markdown("#### Normalization and data boundaries")
    st.markdown("- **Research native units:** daily and trade-level R from the validated independent reproduction artifacts. Its daily curve includes the full 3,255-trade benchmark; the trade ledger itself has missing exit timestamps and those rows are excluded from exit-time strategy curves.\n- **Paper native units:** persisted account snapshots and closed-trade net USD from the selected run. Research USD is not derived from R.\n- **Comparable diagnostic:** Research `r_multiple` and Paper `realized_r` (stored net P&L divided by stored initial risk), plotted as separate segments. The transition interval stays blank until an independently validated historical simulation is available. Costs and fill assumptions differ, so the comparison is not execution parity.\n- **Future LIVE:** schema/legend placeholder only; no live trades or returns are fabricated.")
