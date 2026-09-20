"""
NQ TRADE VISUALIZER — V2
=========================

Purpose
-------
Research/debugging visualizer for frozen strategy trades.

V1 features:
- Loads one or more trade CSVs.
- Loads the canonical Databento MNQ market data through the project loader.
- Candlestick chart around the selected trade.
- Entry / exit / SL / TP overlays when available.
- Trade metadata and lifecycle information.
- Filters by strategy, side, result and volatility regime.
- Previous / next / random trade navigation.
- Uses the project's canonical market loader; it does NOT create a second
  market-data source.

Default trade file:
    src/research/mean_reversion/results/research_08aa_modular_reproduction_trades.csv

Usage
-----
From project root:

    python trade_visualizer.py

Optional:

    python trade_visualizer.py path/to/trades.csv

Multiple files:

    python trade_visualizer.py path/to/mrs.csv path/to/s2r.csv

The script intentionally does not optimize any strategy parameters.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists():
    # Also support placing the script inside src/research/mean_reversion/research
    ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# GUI / plotting
# ---------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Canonical project market loader
# ---------------------------------------------------------------------------

try:
    from src.databento_loader import load_databento_mnq
except Exception as exc:
    load_databento_mnq = None
    _MARKET_IMPORT_ERROR = exc
else:
    _MARKET_IMPORT_ERROR = None


DEFAULT_TRADE_FILE = (
    ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv"
)

NY_TZ = "America/New_York"

# Frozen strategy-level execution parameters.
# These are fallback display values ONLY when the trade CSV does not contain
# explicit SL/TP levels for the individual trade.
FROZEN_STRATEGY_LEVELS = {
    "S2R": {"sl_points": 25.0, "tp_points": 43.75},
    "MRL1": {"sl_points": 37.5, "tp_points": 25.0},
    "MRS2": {"sl_points": 25.0, "tp_points": 27.5},
}


# =============================================================================
# DATA HELPERS
# =============================================================================

def normalize_timestamp(series: pd.Series) -> pd.Series:
    """Normalize timestamps to UTC-aware pandas timestamps."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def find_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        col = find_column(df, [name])
        if col is not None:
            return col
    return None


def normalize_trade_file(path: Path) -> pd.DataFrame:
    """
    Normalize a strategy trade CSV into a stable visualizer schema.

    We preserve the original columns and add normalized columns where possible.
    No values are inferred unless they can be derived unambiguously.
    """
    df = pd.read_csv(path)

    if df.empty:
        return df

    df = df.copy()
    df["_source_file"] = path.name

    # Strategy
    strategy_col = first_existing(
        df,
        ["strategy_name", "strategy", "candidate_id", "name"],
    )
    if strategy_col:
        df["_strategy"] = df[strategy_col].astype(str).str.upper().str.strip()
    else:
        # S2R authoritative trade exports do not necessarily contain a
        # strategy_name column. Identify them from the source filename.
        if "s2r" in path.name.lower():
            df["_strategy"] = "S2R"
        elif "mrl1" in path.name.lower():
            df["_strategy"] = "MRL1"
        elif "mrs2" in path.name.lower():
            df["_strategy"] = "MRS2"
        else:
            df["_strategy"] = path.stem.upper()

    # Entry timestamp
    entry_col = first_existing(
        df,
        [
            "entry_timestamp",
            "timestamp",
            "entry_time",
            "entry_datetime",
        ],
    )
    if entry_col is None:
        raise ValueError(
            f"{path.name}: could not find an entry timestamp column."
        )
    df["_entry_timestamp"] = normalize_timestamp(df[entry_col])

    # Exit timestamp, if present.
    exit_col = first_existing(
        df,
        [
            "exit_timestamp",
            "exit_time",
            "exit_datetime",
            "timestamp_exit",
        ],
    )
    if exit_col:
        df["_exit_timestamp"] = normalize_timestamp(df[exit_col])
    else:
        df["_exit_timestamp"] = pd.NaT

    # Common numeric fields.
    aliases = {
        "_entry_price": ["entry_price", "entry", "fill_price"],
        "_exit_price": ["exit_price", "exit", "close_price"],
        "_r": ["r_multiple", "r", "net_R"],
        "_bars": ["bars_elapsed", "bars", "holding_bars"],
        "_sl": ["stop_loss", "sl", "stop"],
        "_tp": ["take_profit", "tp", "target"],
        "_mae": ["mae", "mae_r", "mae_multiple"],
        "_mfe": ["mfe", "mfe_r", "mfe_multiple"],
    }

    for normalized, candidates in aliases.items():
        col = first_existing(df, candidates)
        if col:
            df[normalized] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[normalized] = np.nan

    # Side.
    side_col = first_existing(df, ["side", "direction", "position"])
    if side_col:
        df["_side"] = (
            df[side_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        # Frozen strategy direction fallback.
        df["_side"] = ""
        for strategy, direction in {
            "S2R": "SHORT",
            "MRL1": "LONG",
            "MRS2": "SHORT",
        }.items():
            mask = df["_strategy"].astype(str).str.upper() == strategy
            df.loc[mask, "_side"] = direction

    # Exit reason.
    reason_col = first_existing(
        df,
        ["exit_reason", "reason", "outcome", "result"],
    )
    if reason_col:
        df["_exit_reason"] = df[reason_col].astype(str)
    else:
        df["_exit_reason"] = ""

    # Regime / signal context.
    # Volatility: keep the raw percentile separate from the display bucket.
    # S2R stores vol_percentile as float64; mixing numeric values and strings
    # in the same pandas column caused the previous V7 dtype error.
    vol_col = first_existing(
        df,
        [
            "entry_vol_percentile",
            "vol_percentile",
            "entry_vol",
            "volatility_percentile",
            "entry_vol_bucket",
            "vol_bucket",
            "volatility_bucket",
        ],
    )

    if vol_col:
        df["_vol_percentile"] = pd.to_numeric(df[vol_col], errors="coerce")
        df["_vol_bucket"] = df["_vol_percentile"].map(classify_vol_bucket)
    else:
        df["_vol_percentile"] = np.nan
        df["_vol_bucket"] = np.nan

    for normalized, candidates in {
        "_hmm_state": ["entry_hmm_state", "hmm_state"],
        "_zscore": ["entry_zscore", "zscore", "zscore_30"],
        "_quality": ["entry_quality", "quality"],
    }.items():
        col = first_existing(df, candidates)
        if col:
            df[normalized] = df[col]
        else:
            df[normalized] = np.nan

    # Strategy-aware fallback context and SL/TP.
    # If the source has no volatility percentile, use the frozen strategy
    # regime only as a display fallback.
    #
    # Explicitly make this column object before assigning string regime labels.
    # Some source CSVs arrive here as float64 when their volatility column is
    # entirely numeric.
    df["_vol_bucket"] = df["_vol_bucket"].astype(object)

    for strategy, vol_bucket in {
        "S2R": "VOL40-60",
        "MRL1": "VOL20-40",
        "MRS2": "VOL80-100",
    }.items():
        mask = (
            df["_strategy"].astype(str).str.upper() == strategy
        )
        missing_vol = mask & (
            df["_vol_bucket"].isna()
            | df["_vol_bucket"].astype(str).isin(["", "nan", "None"])
        )
        df.loc[missing_vol, "_vol_bucket"] = vol_bucket
    for i in df.index:
        strategy = str(df.at[i, "_strategy"]).upper()
        entry = df.at[i, "_entry_price"]
        side = str(df.at[i, "_side"]).upper()

        cfg = FROZEN_STRATEGY_LEVELS.get(strategy)
        if cfg is None or pd.isna(entry):
            continue

        if pd.isna(df.at[i, "_sl"]):
            sl_points = cfg["sl_points"]
            if side == "LONG":
                df.at[i, "_sl"] = float(entry) - sl_points
            elif side == "SHORT":
                df.at[i, "_sl"] = float(entry) + sl_points

        if pd.isna(df.at[i, "_tp"]):
            tp_points = cfg["tp_points"]
            if side == "LONG":
                df.at[i, "_tp"] = float(entry) + tp_points
            elif side == "SHORT":
                df.at[i, "_tp"] = float(entry) - tp_points

    # Stable ordering.
    df = (
        df.dropna(subset=["_entry_timestamp"])
        .sort_values(["_entry_timestamp", "_strategy"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Stable visualizer ID.
    df["_trade_id"] = np.arange(1, len(df) + 1)

    return df


def load_trade_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(normalize_trade_file(path))
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: this CSV is not a trade-level file compatible "
                f"with the visualizer. {exc}"
            ) from exc

    if not frames:
        raise ValueError("No trade files supplied.")

    trades = pd.concat(frames, ignore_index=True)

    # Deterministic tie-break.
    trades["_strategy_order"] = trades["_strategy"].map(
        {
            "S2R": 0,
            "MRL1": 1,
            "MRS2": 2,
        }
    ).fillna(99)

    trades = (
        trades
        .sort_values(
            ["_entry_timestamp", "_strategy_order", "_strategy"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    trades["_trade_id"] = np.arange(1, len(trades) + 1)
    return trades


def classify_vol_bucket(value):
    """Map a 0-100 volatility percentile to the frozen regime bucket."""
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if 0 <= v < 20:
        return "VOL0-20"
    if 20 <= v < 40:
        return "VOL20-40"
    if 40 <= v < 60:
        return "VOL40-60"
    if 60 <= v < 80:
        return "VOL60-80"
    if 80 <= v <= 100:
        return "VOL80-100"
    return str(value)


def load_market() -> pd.DataFrame:
    """Load the project's canonical full MNQ market dataset."""
    if load_databento_mnq is None:
        raise RuntimeError(
            "Could not import src.databento_loader.load_databento_mnq.\n\n"
            f"Import error: {_MARKET_IMPORT_ERROR}"
        )

    market = load_databento_mnq()
    market = market.copy()

    timestamp_col = first_existing(
        market,
        ["timestamp ET", "timestamp", "datetime", "time"],
    )
    if timestamp_col is None:
        raise ValueError(
            "Canonical market data has no recognizable timestamp column."
        )

    market["_timestamp"] = normalize_timestamp(market[timestamp_col])

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in market.columns]
    if missing:
        raise ValueError(
            f"Canonical market data is missing OHLC columns: {missing}"
        )

    for c in required:
        market[c] = pd.to_numeric(market[c], errors="coerce")

    if "volume" in market.columns:
        market["volume"] = pd.to_numeric(market["volume"], errors="coerce")

    market = (
        market
        .dropna(subset=["_timestamp", "open", "high", "low", "close"])
        .sort_values("_timestamp")
        .reset_index(drop=True)
    )

    return market


def attach_market_prices(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing entry/exit prices from the canonical market bars.

    S2R authoritative reproduction intentionally stores timestamps and R/point
    outcomes but not entry/exit price columns. For visualization, exact
    timestamp-to-bar joins are used. No price is guessed from the P&L.
    """
    out = trades.copy()

    price_lookup = (
        market[["_timestamp", "close"]]
        .drop_duplicates("_timestamp")
        .set_index("_timestamp")["close"]
    )

    entry_missing = out["_entry_price"].isna()
    out.loc[entry_missing, "_entry_price"] = (
        out.loc[entry_missing, "_entry_timestamp"].map(price_lookup)
    )

    exit_missing = out["_exit_price"].isna() & out["_exit_timestamp"].notna()
    out.loc[exit_missing, "_exit_price"] = (
        out.loc[exit_missing, "_exit_timestamp"].map(price_lookup)
    )

    return out


# =============================================================================
# VISUALIZER
# =============================================================================

class TradeVisualizer(tk.Tk):
    def __init__(
        self,
        trades: pd.DataFrame,
        market: pd.DataFrame,
        window_bars: int = 60,
    ):
        super().__init__()

        self.title("NQ Trade Visualizer — V1")
        self.geometry("1500x950")
        self.minsize(1100, 750)

        self.all_trades = trades
        self.market = market
        self.window_bars = int(window_bars)

        self.filtered = trades.copy()
        self.position = 0

        self._build_ui()
        self._refresh_filters()
        self._apply_filters()

        self.bind("<Left>", lambda _: self.previous_trade())
        self.bind("<Right>", lambda _: self.next_trade())
        self.bind("<Escape>", lambda _: self.destroy())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Strategy").grid(row=0, column=0, sticky="w")
        self.strategy_var = tk.StringVar(value="ALL")
        self.strategy_box = ttk.Combobox(
            top,
            textvariable=self.strategy_var,
            state="readonly",
            width=12,
        )
        self.strategy_box.grid(row=0, column=1, padx=5)
        self.strategy_box.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Label(top, text="Side").grid(row=0, column=2, sticky="w")
        self.side_var = tk.StringVar(value="ALL")
        self.side_box = ttk.Combobox(
            top,
            textvariable=self.side_var,
            state="readonly",
            width=10,
        )
        self.side_box.grid(row=0, column=3, padx=5)
        self.side_box.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Label(top, text="Result").grid(row=0, column=4, sticky="w")
        self.result_var = tk.StringVar(value="ALL")
        self.result_box = ttk.Combobox(
            top,
            textvariable=self.result_var,
            state="readonly",
            width=12,
        )
        self.result_box.grid(row=0, column=5, padx=5)
        self.result_box.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Label(top, text="Vol regime").grid(row=0, column=6, sticky="w")
        self.vol_var = tk.StringVar(value="ALL")
        self.vol_box = ttk.Combobox(
            top,
            textvariable=self.vol_var,
            state="readonly",
            width=14,
        )
        self.vol_box.grid(row=0, column=7, padx=5)
        self.vol_box.bind("<<ComboboxSelected>>", lambda _: self._apply_filters())

        ttk.Label(top, text="Bars around entry").grid(row=0, column=8, sticky="w")
        self.window_var = tk.IntVar(value=self.window_bars)
        self.window_spin = ttk.Spinbox(
            top,
            from_=10,
            to=300,
            textvariable=self.window_var,
            width=7,
            command=self._redraw,
        )
        self.window_spin.grid(row=0, column=9, padx=5)

        ttk.Button(top, text="Add trade CSV...", command=self.add_trade_files).grid(
            row=0, column=10, padx=5
        )
        ttk.Button(top, text="Random", command=self.random_trade).grid(
            row=0, column=11, padx=5
        )
        ttk.Button(top, text="Previous", command=self.previous_trade).grid(
            row=0, column=11, padx=5
        )
        ttk.Button(top, text="Next", command=self.next_trade).grid(
            row=0, column=12, padx=5
        )
        ttk.Button(top, text="Jump to ID", command=self.jump_dialog).grid(
            row=0, column=13, padx=5
        )

        self.status_var = tk.StringVar()
        ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(10, 4),
        ).pack(fill="x")

        main = ttk.Panedwindow(self, orient="vertical")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        chart_frame = ttk.Frame(main)
        info_frame = ttk.Frame(main)

        main.add(chart_frame, weight=4)
        main.add(info_frame, weight=1)

        self.figure = plt.Figure(figsize=(14, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=chart_frame,
        )
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.info_text = tk.Text(
            info_frame,
            height=8,
            wrap="word",
            font=("Consolas", 10),
        )
        self.info_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _refresh_filters(self):
        def values(col):
            vals = self.all_trades[col].dropna().astype(str).unique().tolist()
            return ["ALL"] + sorted(vals)

        self.strategy_box["values"] = values("_strategy")
        self.side_box["values"] = values("_side")
        self.result_box["values"] = ["ALL", "WIN", "LOSS", "TIMEOUT", "OTHER"]
        self.vol_box["values"] = values("_vol_bucket")

    def _result_category(self, row) -> str:
        reason = str(row["_exit_reason"]).upper()
        r = row["_r"]

        if "TIME" in reason:
            return "TIMEOUT"

        if pd.notna(r):
            if float(r) > 0:
                return "WIN"
            if float(r) < 0:
                return "LOSS"

        if "WIN" in reason:
            return "WIN"
        if "LOSS" in reason or "SL" in reason:
            return "LOSS"

        return "OTHER"

    def _apply_filters(self):
        df = self.all_trades.copy()

        strategy = self.strategy_var.get()
        side = self.side_var.get()
        result = self.result_var.get()
        vol = self.vol_var.get()

        if strategy != "ALL":
            df = df[df["_strategy"] == strategy]

        if side != "ALL":
            df = df[df["_side"] == side]

        if vol != "ALL":
            df = df[df["_vol_bucket"].astype(str) == vol]

        if result != "ALL":
            categories = df.apply(self._result_category, axis=1)
            df = df[categories == result]

        self.filtered = df.reset_index(drop=True)

        if self.filtered.empty:
            self.position = 0
            self.status_var.set("No trades match the current filters.")
            self._clear_chart()
            return

        self.position = min(self.position, len(self.filtered) - 1)
        self._redraw()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def add_trade_files(self):
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Add strategy trade CSV(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return

        try:
            extra = load_trade_files([Path(p) for p in paths])
        except Exception as exc:
            messagebox.showerror("Could not load trade file", str(exc))
            return

        combined = pd.concat(
            [self.all_trades, extra],
            ignore_index=True,
        )

        combined["_strategy_order"] = combined["_strategy"].map(
            {"S2R": 0, "MRL1": 1, "MRS2": 2}
        ).fillna(99)

        combined = (
            combined
            .sort_values(
                ["_entry_timestamp", "_strategy_order", "_strategy"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        combined["_trade_id"] = np.arange(1, len(combined) + 1)

        self.all_trades = combined
        self.position = 0
        self._refresh_filters()
        self._apply_filters()

    def previous_trade(self):
        if self.filtered.empty:
            return
        self.position = max(0, self.position - 1)
        self._redraw()

    def next_trade(self):
        if self.filtered.empty:
            return
        self.position = min(len(self.filtered) - 1, self.position + 1)
        self._redraw()

    def random_trade(self):
        if self.filtered.empty:
            return
        self.position = random.randrange(len(self.filtered))
        self._redraw()

    def jump_dialog(self):
        if self.filtered.empty:
            return

        dialog = tk.Toplevel(self)
        dialog.title("Jump to trade ID")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Trade ID:").pack(padx=15, pady=(15, 5))
        entry = ttk.Entry(dialog, width=15)
        entry.pack(padx=15, pady=5)
        entry.focus_set()

        def go():
            try:
                trade_id = int(entry.get())
            except ValueError:
                messagebox.showerror("Invalid ID", "Trade ID must be an integer.")
                return

            matches = self.filtered.index[
                self.filtered["_trade_id"] == trade_id
            ].tolist()

            if not matches:
                messagebox.showerror(
                    "Not found",
                    "That trade ID is not in the current filtered set.",
                )
                return

            self.position = matches[0]
            dialog.destroy()
            self._redraw()

        ttk.Button(dialog, text="Go", command=go).pack(pady=(5, 15))
        entry.bind("<Return>", lambda _: go())

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    def _clear_chart(self):
        self.ax.clear()
        self.ax.set_title("No trades selected")
        self.canvas.draw()
        self.info_text.delete("1.0", "end")

    def _find_market_window(self, timestamp: pd.Timestamp):
        idx = self.market["_timestamp"].searchsorted(timestamp)

        if idx >= len(self.market):
            idx = len(self.market) - 1

        left = max(0, int(idx) - self.window_bars)
        right = min(len(self.market), int(idx) + self.window_bars + 1)

        return self.market.iloc[left:right].copy(), int(idx)

    def _price_level(self, row, key):
        value = row[key]
        return float(value) if pd.notna(value) else None

    def _draw_candles(self, data: pd.DataFrame):
        self.ax.clear()

        x = mdates.date2num(data["_timestamp"].dt.tz_convert(NY_TZ).dt.to_pydatetime())

        # Approximate candle width for 1-minute bars.
        width = 0.00045

        for xi, (_, bar) in zip(x, data.iterrows()):
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])

            # Wick
            self.ax.plot(
                [xi, xi],
                [l, h],
                linewidth=0.8,
                alpha=0.9,
            )

            # Body
            bottom = min(o, c)
            height = max(abs(c - o), 1e-8)

            self.ax.add_patch(
                Rectangle(
                    (xi - width / 2, bottom),
                    width,
                    height,
                    fill=(c >= o),
                    alpha=0.75,
                    linewidth=0.7,
                )
            )

        self.ax.grid(True, alpha=0.18)
        self.ax.xaxis_date()
        # Include the trading date so overnight / multi-session inspection
        # cannot be confused by identical clock times on different dates.
        self.ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=None)
        )
        self.ax.set_ylabel("MNQ price")

    def _draw_trade_levels(self, row):
        entry = self._price_level(row, "_entry_price")
        exit_price = self._price_level(row, "_exit_price")
        sl = self._price_level(row, "_sl")
        tp = self._price_level(row, "_tp")

        levels = [
            (entry, "ENTRY", "-"),
            (exit_price, "EXIT", ":"),
            (sl, "SL", "--"),
            (tp, "TP", "--"),
        ]

        for price, label, linestyle in levels:
            if price is not None:
                self.ax.axhline(
                    price,
                    linestyle=linestyle,
                    linewidth=1.2 if label in {"SL", "TP"} else 1.0,
                    alpha=0.95,
                    label=f"{label} {price:.2f}",
                )

        # Put the exact TP / SL values at the right side of the chart as well.
        x_right = self.ax.get_xlim()[1]
        if sl is not None:
            self.ax.annotate(
                f"SL  {sl:.2f}",
                xy=(x_right, sl),
                xycoords=("data", "data"),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        if tp is not None:
            self.ax.annotate(
                f"TP  {tp:.2f}",
                xy=(x_right, tp),
                xycoords=("data", "data"),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    def _redraw(self):
        if self.filtered.empty:
            return

        try:
            self.window_bars = int(self.window_var.get())
        except Exception:
            self.window_bars = 60

        row = self.filtered.iloc[self.position]

        entry_ts = row["_entry_timestamp"]
        data, entry_idx = self._find_market_window(entry_ts)

        self._draw_candles(data)
        self._draw_trade_levels(row)

        entry_ny = entry_ts.tz_convert(NY_TZ)
        self.ax.axvline(
            mdates.date2num(entry_ny.to_pydatetime()),
            linestyle="-",
            linewidth=1.2,
            label="Entry time",
        )

        exit_ts = row["_exit_timestamp"]
        if pd.notna(exit_ts):
            exit_ny = exit_ts.tz_convert(NY_TZ)
            self.ax.axvline(
                mdates.date2num(exit_ny.to_pydatetime()),
                linestyle=":",
                linewidth=1.2,
                label="Exit time",
            )

        strategy = row["_strategy"]
        side = row["_side"]

        result = self._result_category(row)
        r = row["_r"]

        title = (
            f"Trade {int(row['_trade_id'])} | "
            f"{strategy} | {side or 'SIDE ?'} | {result} | "
            f"{entry_ts.tz_convert(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

        if pd.notna(r):
            title += f" | R = {float(r):+.3f}"

        self.ax.set_title(title)
        self.ax.legend(
            loc="upper left",
            fontsize=8,
            ncol=2,
        )

        self.figure.tight_layout()
        self.canvas.draw()

        self._update_info(row)

        self.status_var.set(
            f"Showing filtered trade {self.position + 1}/{len(self.filtered)} "
            f"| Global trade ID {int(row['_trade_id'])}"
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _fmt(self, value, decimals=4):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value)

    def _update_info(self, row):
        entry = row["_entry_timestamp"].tz_convert(NY_TZ)

        strategy = str(row["_strategy"]).upper()
        cfg = FROZEN_STRATEGY_LEVELS.get(strategy)

        lines = [
            f"TRADE ID        : {int(row['_trade_id'])}",
            f"STRATEGY        : {row['_strategy']}",
            f"FROZEN SL/TP    : {cfg['sl_points']:.2f} / {cfg['tp_points']:.2f} pts" if cfg else "FROZEN SL/TP    : N/A",
            f"SIDE            : {row['_side'] or 'N/A'}",
            f"VOL REGIME       : {self._fmt(row['_vol_bucket'])}",
            f"ENTRY           : {entry}",
            f"EXIT            : {row['_exit_timestamp'].tz_convert(NY_TZ) if pd.notna(row['_exit_timestamp']) else 'N/A'}",
            f"ENTRY PRICE     : {self._fmt(row['_entry_price'], 2)}",
            f"EXIT PRICE      : {self._fmt(row['_exit_price'], 2)}",
            f"R MULTIPLE      : {self._fmt(row['_r'], 4)}",
            f"BARS            : {self._fmt(row['_bars'], 0)}",
            f"EXIT REASON     : {row['_exit_reason'] or 'N/A'}",
            f"SL               : {self._fmt(row['_sl'], 2)}",
            f"TP               : {self._fmt(row['_tp'], 2)}",
            f"MAE              : {self._fmt(row['_mae'], 4)}",
            f"MFE              : {self._fmt(row['_mfe'], 4)}",
            f"VOL BUCKET       : {self._fmt(row['_vol_bucket'])}",
            f"HMM STATE        : {self._fmt(row['_hmm_state'])}",
            f"ZSCORE           : {self._fmt(row['_zscore'])}",
            f"QUALITY          : {self._fmt(row['_quality'])}",
            f"SOURCE FILE      : {row['_source_file']}",
        ]

        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NQ frozen-strategy trade visualizer."
    )
    parser.add_argument(
        "trade_files",
        nargs="*",
        help=(
            "One or more trade CSV files. For example, pass the 08AA MR "
            "trades CSV and the S2R trades CSV together."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Number of market bars shown before and after entry.",
    )
    return parser.parse_args()


def choose_trade_files() -> list[Path]:
    root = tk.Tk()
    root.withdraw()

    paths = filedialog.askopenfilenames(
        title="Select trade CSV file(s) — MRL1 / S2R / MRS2",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    root.destroy()
    return [Path(p) for p in paths]


def main():
    args = parse_args()

    if args.trade_files:
        trade_paths = [Path(p).expanduser().resolve() for p in args.trade_files]
    elif DEFAULT_TRADE_FILE.exists():
        # Start with the validated MR trade file. Also auto-discover S2R trade
        # CSVs in the mean-reversion results directory when present.
        trade_paths = [DEFAULT_TRADE_FILE]

        # S2R authoritative trade stream:
        # s2r_modular_authoritative_reproduction.csv has 537 trades and is
        # the validated modular reproduction. The 5231-row
        # s2r_modular_full_databento_trades.csv is a broader event/export
        # stream and must NOT be treated as the frozen trade list.
        authoritative_s2r = (
            ROOT
            / "src"
            / "research"
            / "results"
            / "s2_extended"
            / "s2r_modular_authoritative_reproduction.csv"
        )

        if authoritative_s2r.exists():
            trade_paths.append(authoritative_s2r)
            print("\nUsing AUTHORITATIVE S2R trade file:")
            print(f"  {authoritative_s2r}")
            print("  Expected validated trade count: 537")
        else:
            print(
                "\nWARNING: Authoritative S2R trade file was not found:"
            )
            print(f"  {authoritative_s2r}")
            print(
                "Use Add trade CSV... to load "
                "s2r_modular_authoritative_reproduction.csv manually."
            )

    else:
        trade_paths = choose_trade_files()

    if not trade_paths:
        print("No trade files selected.")
        return

    print("=" * 80)
    print("NQ TRADE VISUALIZER — V2")
    print("=" * 80)
    print("\nTrade files:")
    for p in trade_paths:
        print(f"  {p}")

    try:
        trades = load_trade_files(trade_paths)
    except Exception as exc:
        raise SystemExit(f"\nCould not load trades:\n{exc}") from exc

    print(f"\nTrades loaded: {len(trades):,}")
    print(trades["_strategy"].value_counts().to_string())

    loaded_strategies = set(trades["_strategy"].astype(str).str.upper())
    missing_strategies = {"S2R", "MRL1", "MRS2"} - loaded_strategies
    if missing_strategies:
        print(
            "\nWARNING: These expected frozen strategies were not loaded: "
            + ", ".join(sorted(missing_strategies))
        )
        if "S2R" in missing_strategies:
            print(
                "S2R is NOT contained in research_08aa_modular_reproduction_trades.csv. "
                "Its own trade CSV must be loaded."
            )

    print("\nStrategy display configuration:")
    for strategy, cfg in FROZEN_STRATEGY_LEVELS.items():
        print(
            f"  {strategy}: SL={cfg['sl_points']:.2f} pts | "
            f"TP={cfg['tp_points']:.2f} pts"
        )

    print("\nLoading canonical Databento market data...")
    print("This can take a little while because the visualizer uses the full")
    print("market path so the chart is based on the same data source as research.")

    try:
        market = load_market()
    except Exception as exc:
        raise SystemExit(f"\nCould not load market data:\n{exc}") from exc

    print(f"Market bars loaded: {len(market):,}")

    trades = attach_market_prices(trades, market)

    print("\nTrade-source audit:")
    print(trades["_strategy"].value_counts().to_string())

    for strategy in ["S2R", "MRL1", "MRS2"]:
        subset = trades[trades["_strategy"] == strategy]
        if subset.empty:
            continue
        print(
            f"  {strategy}: {len(subset):,} trades | "
            f"entry prices available {subset['_entry_price'].notna().sum():,} | "
            f"exit prices available {subset['_exit_price'].notna().sum():,}"
        )

    app = TradeVisualizer(
        trades=trades,
        market=market,
        window_bars=args.window,
    )

    app.mainloop()


if __name__ == "__main__":
    main()
