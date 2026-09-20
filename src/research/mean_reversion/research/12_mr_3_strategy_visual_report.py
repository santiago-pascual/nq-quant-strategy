"""
12_mr_3_strategy_visual_report.py

FINAL 3-STRATEGY VISUAL REPORT

Strategies:
    MRL1 = 483
    S2R  = 537
    MRS2 = 1052

Total:
    2072 trades

This script only creates visualization files.
It does not modify research outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# PROJECT ROOT
# ============================================================


def find_project_root():
    current = Path(__file__).resolve()

    for path in [current, *current.parents]:
        if (path / "src").exists() and (path / "data").exists():
            return path

    raise RuntimeError("Could not locate project root.")


PROJECT_ROOT = find_project_root()

PORTFOLIO_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio_3_strategy"

S2R_DIR = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

OUTPUT_DIR = PORTFOLIO_DIR / "png"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FILES
# ============================================================

MR_FILE = PORTFOLIO_DIR / "portfolio_3_strategy_trades.csv"

S2R_FILE = S2R_DIR / "s2r_modular_authoritative_reproduction.csv"


# ============================================================
# PLOT CONFIG
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# ============================================================
# HELPERS
# ============================================================


def save_fig(fig, filename):

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"[OK] {filename}")


def fmt_r(value, _):
    return f"{value:.0f}R"


# ============================================================
# LOAD MRL1 + MRS2
# ============================================================

print("=" * 70)
print("3-STRATEGY PORTFOLIO VISUAL REPORT")
print("=" * 70)

print()
print("Loading MRL1 + MRS2...")

mr = pd.read_csv(MR_FILE)

print(f"MR rows: {len(mr):,}")

print(mr["strategy_name"].value_counts().to_dict())


# ============================================================
# LOAD S2R
# ============================================================

print()
print("Loading authoritative S2R...")

s2r = pd.read_csv(S2R_FILE)

print(f"S2R rows: {len(s2r):,}")

print("S2R columns verified:")

print(
    s2r[
        [
            "entry_timestamp",
            "exit_timestamp",
            "net_R",
        ]
    ]
    .head(3)
    .to_string(index=False)
)


# ============================================================
# BUILD MR STREAM
# ============================================================

mr_clean = pd.DataFrame(
    {
        "strategy": mr["strategy_name"].astype(str),
        "entry_timestamp": pd.to_datetime(
            mr["entry_timestamp"],
            utc=True,
            errors="coerce",
        ),
        "exit_timestamp": pd.to_datetime(
            mr["exit_timestamp"],
            utc=True,
            errors="coerce",
        ),
        "R": pd.to_numeric(
            mr["r_multiple"],
            errors="coerce",
        ),
    }
)


# ============================================================
# BUILD S2R STREAM
# ============================================================

s2r_clean = pd.DataFrame(
    {
        "strategy": "S2R",
        "entry_timestamp": pd.to_datetime(
            s2r["entry_timestamp"],
            utc=True,
            errors="coerce",
        ),
        "exit_timestamp": pd.to_datetime(
            s2r["exit_timestamp"],
            utc=True,
            errors="coerce",
        ),
        "R": pd.to_numeric(
            s2r["net_R"],
            errors="coerce",
        ),
    }
)


# ============================================================
# S2R SANITY CHECK BEFORE CONCAT
# ============================================================

print()
print("=" * 70)
print("S2R SANITY CHECK")
print("=" * 70)

print("rows:", len(s2r_clean))

print("valid entry timestamps:", s2r_clean["entry_timestamp"].notna().sum())

print("valid R:", s2r_clean["R"].notna().sum())

print("strategy values:", s2r_clean["strategy"].value_counts().to_dict())

print(
    "total S2R R:",
    round(
        s2r_clean["R"].sum(),
        6,
    ),
)


if len(s2r_clean) != 537:
    raise RuntimeError("S2R row count changed.")

if s2r_clean["entry_timestamp"].notna().sum() != 537:
    raise RuntimeError("Some S2R entry timestamps failed parsing.")

if s2r_clean["R"].notna().sum() != 537:
    raise RuntimeError("Some S2R R values failed parsing.")


# ============================================================
# COMBINE
# ============================================================

trades = pd.concat(
    [
        mr_clean,
        s2r_clean,
    ],
    ignore_index=True,
)


# ============================================================
# FINAL CLEANUP
# ============================================================

trades = trades.dropna(
    subset=[
        "strategy",
        "entry_timestamp",
        "R",
    ]
).copy()


trades = trades.sort_values(
    [
        "entry_timestamp",
        "strategy",
    ]
).reset_index(drop=True)


# ============================================================
# FINAL AUDIT
# ============================================================

counts = trades["strategy"].value_counts()


print()
print("=" * 70)
print("FINAL 3-STRATEGY AUDIT")
print("=" * 70)

print(f"MRL1: {counts.get('MRL1', 0):,}")

print(f"S2R : {counts.get('S2R', 0):,}")

print(f"MRS2: {counts.get('MRS2', 0):,}")

print("-" * 30)

print(f"TOTAL: {len(trades):,}")


expected = {
    "MRL1": 483,
    "S2R": 537,
    "MRS2": 1052,
}


for strategy, expected_count in expected.items():
    actual = int(
        counts.get(
            strategy,
            0,
        )
    )

    if actual != expected_count:
        raise RuntimeError(f"{strategy}: expected {expected_count}, got {actual}")


if len(trades) != 2072:
    raise RuntimeError(f"Expected 2072 trades, got {len(trades)}")


print()
print("[PASS] MRL1 = 483")
print("[PASS] S2R  = 537")
print("[PASS] MRS2 = 1052")
print("[PASS] TOTAL = 2072")


# ============================================================
# DUPLICATE AUDIT
# ============================================================

duplicates = trades.duplicated(
    subset=[
        "entry_timestamp",
        "strategy",
    ]
).sum()


print(f"[PASS] duplicates = {duplicates}")


if duplicates != 0:
    raise RuntimeError("Duplicate strategy-entry timestamps detected.")


# ============================================================
# EQUITY
# ============================================================

trades["cumulative_R"] = trades["R"].cumsum()


total_R = trades["R"].sum()


# ============================================================
# 01 EQUITY CURVE
# ============================================================

print()
print("[1/10] Equity curve")

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    trades["entry_timestamp"],
    trades["cumulative_R"],
    linewidth=2,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_title(
    "3-Strategy Mean Reversion Portfolio — Cumulative R",
    loc="left",
    fontweight="bold",
)

ax.set_ylabel("Cumulative R")

ax.yaxis.set_major_formatter(FuncFormatter(fmt_r))

ax.text(
    0.99,
    0.03,
    (f"{len(trades):,} trades  |  +{total_R:.2f}R"),
    transform=ax.transAxes,
    ha="right",
)

save_fig(
    fig,
    "01_equity_curve.png",
)


# ============================================================
# 02 DRAWDOWN
# ============================================================

print("[2/10] Drawdown")

running_max = trades["cumulative_R"].cummax()

drawdown = trades["cumulative_R"] - running_max

max_dd = drawdown.min()


fig, ax = plt.subplots(figsize=(13, 5))

ax.fill_between(
    trades["entry_timestamp"],
    drawdown,
    0,
    alpha=0.30,
)

ax.plot(
    trades["entry_timestamp"],
    drawdown,
    linewidth=1.5,
)

ax.set_title(
    "3-Strategy Portfolio — Drawdown",
    loc="left",
    fontweight="bold",
)

ax.set_ylabel("Drawdown (R)")

ax.yaxis.set_major_formatter(FuncFormatter(fmt_r))

ax.text(
    0.99,
    0.05,
    f"Max DD: {max_dd:.2f}R",
    transform=ax.transAxes,
    ha="right",
)

save_fig(
    fig,
    "02_drawdown.png",
)


# ============================================================
# 03 STRATEGY CONTRIBUTION
# ============================================================

print("[3/10] Strategy contribution")

contribution = (
    trades.groupby("strategy")["R"]
    .agg(
        trades="count",
        total_R="sum",
        expectancy="mean",
    )
    .reindex(
        [
            "MRL1",
            "S2R",
            "MRS2",
        ]
    )
    .reset_index()
)


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    contribution["strategy"],
    contribution["total_R"],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_title(
    "Strategy Contribution to Portfolio Return",
    loc="left",
    fontweight="bold",
)

ax.set_ylabel("Total R")

for bar, value in zip(
    bars,
    contribution["total_R"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:+.2f}R",
        ha="center",
        va="bottom",
    )

save_fig(
    fig,
    "03_strategy_contribution.png",
)


# ============================================================
# 04 YEARLY RETURNS
# ============================================================

print("[4/10] Yearly returns")

yearly = (
    trades.assign(year=trades["entry_timestamp"].dt.year).groupby("year")["R"].sum()
)


fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    yearly.index.astype(str),
    yearly.values,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_title(
    "Annual Portfolio Return",
    loc="left",
    fontweight="bold",
)

ax.set_ylabel("Net R")

ax.set_xlabel("Year")

for bar, value in zip(
    bars,
    yearly.values,
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:+.1f}R",
        ha="center",
        va=("bottom" if value >= 0 else "top"),
    )

save_fig(
    fig,
    "04_yearly_returns.png",
)


# ============================================================
# 05 MONTHLY HEATMAP
# ============================================================

print("[5/10] Monthly heatmap")

monthly = (
    trades.assign(
        year=trades["entry_timestamp"].dt.year,
        month=trades["entry_timestamp"].dt.month,
    )
    .groupby(
        [
            "year",
            "month",
        ]
    )["R"]
    .sum()
    .reset_index()
)


pivot = monthly.pivot(
    index="year",
    columns="month",
    values="R",
)

pivot = pivot.reindex(columns=range(1, 13))


fig, ax = plt.subplots(figsize=(13, 6))

im = ax.imshow(
    pivot.values,
    aspect="auto",
)

ax.set_title(
    "Monthly Portfolio Returns",
    loc="left",
    fontweight="bold",
)

ax.set_xlabel("Month")

ax.set_ylabel("Year")

ax.set_xticks(range(12))

ax.set_xticklabels(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
)

ax.set_yticks(range(len(pivot.index)))

ax.set_yticklabels(pivot.index)

for i in range(len(pivot.index)):
    for j in range(12):
        value = pivot.iloc[
            i,
            j,
        ]

        if pd.notna(value):
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )


fig.colorbar(
    im,
    ax=ax,
    label="R",
)

save_fig(
    fig,
    "05_monthly_heatmap.png",
)


# ============================================================
# 06 STRATEGY METRICS
# ============================================================

print("[6/10] Strategy metrics")

metric_rows = []

for strategy, group in trades.groupby("strategy"):
    values = group["R"].dropna()

    wins = values[values > 0]

    losses = values[values < 0]

    gross_profit = wins.sum()

    gross_loss = abs(losses.sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

    metric_rows.append(
        {
            "strategy": strategy,
            "win_rate": (values.gt(0).mean() * 100),
            "profit_factor": pf,
            "expectancy": values.mean(),
        }
    )


metrics = (
    pd.DataFrame(metric_rows)
    .set_index("strategy")
    .reindex(
        [
            "MRL1",
            "S2R",
            "MRS2",
        ]
    )
    .reset_index()
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)


specs = [
    (
        "win_rate",
        "Win Rate",
        "%",
    ),
    (
        "profit_factor",
        "Profit Factor",
        "",
    ),
    (
        "expectancy",
        "Expectancy",
        "R",
    ),
]


for ax, (
    column,
    title,
    suffix,
) in zip(
    axes,
    specs,
):
    bars = ax.bar(
        metrics["strategy"],
        metrics[column],
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    for bar, value in zip(
        bars,
        metrics[column],
    ):
        if suffix == "%":
            text = f"{value:.1f}%"
        else:
            text = f"{value:.3f}{suffix}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            text,
            ha="center",
            va="bottom",
        )


fig.suptitle(
    "Strategy-Level Performance",
    fontsize=17,
    fontweight="bold",
    x=0.03,
    ha="left",
)

fig.tight_layout()

save_fig(
    fig,
    "06_strategy_metrics.png",
)


# ============================================================
# 07 TRADE DISTRIBUTION
# ============================================================

print("[7/10] Trade distribution")

fig, ax = plt.subplots(figsize=(11, 6))

for strategy, group in trades.groupby("strategy"):
    ax.hist(
        group["R"].dropna(),
        bins=45,
        alpha=0.35,
        label=strategy,
    )


ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_title(
    "Trade Outcome Distribution",
    loc="left",
    fontweight="bold",
)

ax.set_xlabel("R per trade")

ax.set_ylabel("Number of trades")

ax.legend()

save_fig(
    fig,
    "07_trade_distribution.png",
)


# ============================================================
# 08 DAILY DISTRIBUTION
# ============================================================

print("[8/10] Daily P&L distribution")

daily = trades.assign(date=trades["entry_timestamp"].dt.date).groupby("date")["R"].sum()


fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    daily.values,
    bins=40,
    alpha=0.75,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_title(
    "Daily Portfolio Return Distribution",
    loc="left",
    fontweight="bold",
)

ax.set_xlabel("Daily R")

ax.set_ylabel("Number of days")

ax.text(
    0.99,
    0.95,
    (
        f"Mean: {daily.mean():+.3f}R\n"
        f"Median: {daily.median():+.3f}R\n"
        f"Best: {daily.max():+.2f}R\n"
        f"Worst: {daily.min():+.2f}R"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
)

save_fig(
    fig,
    "08_daily_pnl_distribution.png",
)


# ============================================================
# 09 CORRELATION
# ============================================================

print("[9/10] Strategy correlation")

daily_strategy = (
    trades.assign(date=trades["entry_timestamp"].dt.date)
    .groupby(
        [
            "date",
            "strategy",
        ]
    )["R"]
    .sum()
    .unstack("strategy")
)

daily_strategy = daily_strategy.reindex(
    columns=[
        "MRL1",
        "S2R",
        "MRS2",
    ]
)

corr = daily_strategy.corr()


fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    corr.values,
    aspect="auto",
    vmin=-1,
    vmax=1,
)

ax.set_title(
    "Daily Strategy Return Correlation",
    loc="left",
    fontweight="bold",
)

ax.set_xticks(range(3))

ax.set_xticklabels(
    corr.columns,
    rotation=45,
    ha="right",
)

ax.set_yticks(range(3))

ax.set_yticklabels(corr.index)

for i in range(3):
    for j in range(3):
        value = corr.iloc[
            i,
            j,
        ]

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
        )


fig.colorbar(
    im,
    ax=ax,
    label="Correlation",
)

save_fig(
    fig,
    "09_strategy_correlation.png",
)


# ============================================================
# 10 PORTFOLIO SUMMARY
# ============================================================

print("[10/10] Portfolio summary")

values = trades["R"].dropna()


wins = values[values > 0]

losses = values[values < 0]


gross_profit = wins.sum()

gross_loss = abs(losses.sum())

profit_factor = gross_profit / gross_loss

win_rate = values.gt(0).mean()

expectancy = values.mean()


running_equity = values.cumsum()

running_max = running_equity.cummax()

portfolio_dd = running_equity - running_max

max_dd = portfolio_dd.min()


daily_mean = daily.mean()

daily_std = daily.std(ddof=1)

sharpe = daily_mean / daily_std * np.sqrt(252)


downside = daily[daily < 0]

if len(downside) > 1:
    downside_std = downside.std(ddof=1)

    sortino = daily_mean / downside_std * np.sqrt(252)

else:
    sortino = np.nan


fig, ax = plt.subplots(figsize=(12, 7))

ax.axis("off")


fig.text(
    0.07,
    0.90,
    "3-STRATEGY MEAN REVERSION PORTFOLIO",
    fontsize=22,
    fontweight="bold",
)


fig.text(
    0.07,
    0.84,
    "Historical validation summary",
    fontsize=12,
)


summary = [
    (
        "Trades",
        f"{len(values):,}",
    ),
    (
        "Total return",
        f"{values.sum():+.2f}R",
    ),
    (
        "Expectancy",
        f"{expectancy:+.4f}R",
    ),
    (
        "Win rate",
        f"{win_rate * 100:.2f}%",
    ),
    (
        "Profit factor",
        f"{profit_factor:.3f}",
    ),
    (
        "Max drawdown",
        f"{max_dd:.2f}R",
    ),
    (
        "Sharpe",
        f"{sharpe:.3f}",
    ),
    (
        "Sortino",
        f"{sortino:.3f}",
    ),
]


x_positions = [
    0.08,
    0.55,
]

y_start = 0.70
y_step = 0.14


for i, (
    label,
    value,
) in enumerate(summary):
    col = i % 2
    row = i // 2

    x = x_positions[col]

    y = y_start - row * y_step

    fig.text(
        x,
        y,
        label.upper(),
        fontsize=9,
        fontweight="bold",
    )

    fig.text(
        x,
        y - 0.055,
        value,
        fontsize=20,
        fontweight="bold",
    )


fig.text(
    0.07,
    0.08,
    (
        "MRL1 + S2R + MRS2  |  "
        "483 + 537 + 1052 trades  |  "
        "Historical trade-stream analysis"
    ),
    fontsize=9,
)


save_fig(
    fig,
    "10_portfolio_summary.png",
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("VISUAL REPORT COMPLETE")
print("=" * 70)

print()
print(f"MRL1 = {counts['MRL1']:,}")

print(f"S2R  = {counts['S2R']:,}")

print(f"MRS2 = {counts['MRS2']:,}")

print(f"TOTAL = {len(trades):,}")

print(f"TOTAL R = {total_R:+.4f}R")

print(f"PF = {profit_factor:.4f}")

print(f"MAX DD = {max_dd:.4f}R")

print()
print(f"PNG output: {OUTPUT_DIR}")

print()

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {file.name}")
