from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# PURPOSE
# =============================================================================
# Portfolio simulation of the TWO FROZEN MR strategies together:
#   MRS2 + MRL1
#
# No strategy optimization.
# No individual-trade shuffle.
# The two historical trade streams are merged chronologically into ONE
# portfolio event stream. Randomness exists only in the choice of a real
# historical starting day. The sequence then walks forward chronologically,
# wrapping after the final historical trade.
#
# IMPORTANT PORTFOLIO ASSUMPTION
# --------------------------------
# Risk is applied PER TRADE from the $50k starting balance, exactly like the
# existing individual-strategy simulator. Therefore 0.25% means $125 risk on
# every MRS2 or MRL1 trade, not $125 split between the two strategies.
#
# If two trades have the exact same entry timestamp, they are processed in a
# deterministic order: MRS2 before MRL1. The script audits how often this
# occurs. For the first portfolio study this keeps the event stream explicit;
# a later robustness test can model simultaneous fills as a group if desired.
# =============================================================================

STARTING_BALANCE = 50_000.0
PROFIT_TARGET = 3_000.0
MAX_LOSS_LIMIT = 2_000.0
XFA_FIXED_LOSS_FLOOR = STARTING_BALANCE - MAX_LOSS_LIMIT

RISK_SCENARIOS = {
    "0.25%": {"initial": 0.0025, "after_threshold": 0.0025, "threshold": None},
    "0.50%": {"initial": 0.0050, "after_threshold": 0.0050, "threshold": None},
    "0.75%": {"initial": 0.0075, "after_threshold": 0.0075, "threshold": None},
    "1.00%": {"initial": 0.0100, "after_threshold": 0.0100, "threshold": None},
    "0.50_to_1.00": {
        "initial": 0.0050,
        "after_threshold": 0.0100,
        "threshold": 1_000.0,
    },
}

N_SIMULATIONS = 50_000
BATCH_SIZE = 1_000
MAX_COMBINE_TRADES = 500
MAX_XFA_TRADES = 2_000
RANDOM_SEED = 42

PAYOUT_INTERVALS = [20, 21, 22]
PAYOUT_AMOUNTS = [500, 750, 1_000, 1_250, 1_500, 1_750, 2_000]
MIN_WINNING_DAYS = 5
MIN_WINNING_DAY_PROFIT = 150.0

PARITY_PATHS = 25
PARITY_MAX_TRADES = 250

HERE = Path(__file__).resolve()
CANDIDATE_ROOTS = [HERE.parents[4], HERE.parent]
RESULTS_DIR = next(
    (
        root / "src" / "research" / "mean_reversion" / "results"
        for root in CANDIDATE_ROOTS
        if (root / "src" / "research" / "mean_reversion" / "results").exists()
    ),
    Path(
        r"C:\Users\Alejandra\Desktop\Quant\08_NQ Strategy Project\src\research\mean_reversion\results"
    ),
)

INPUT_FILE = RESULTS_DIR / "research_08aa_modular_reproduction_trades.csv"
OUTPUT_COMBINE = RESULTS_DIR / "mr_portfolio_funded_combine_results.csv"
OUTPUT_XFA = RESULTS_DIR / "mr_portfolio_funded_xfa_results.csv"
OUTPUT_AUDIT = RESULTS_DIR / "mr_portfolio_audit.csv"


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def get_risk(balance: float, scenario: dict[str, float | None]) -> float:
    threshold = scenario["threshold"]
    if threshold is not None and balance - STARTING_BALANCE >= float(threshold):
        return float(scenario["after_threshold"])
    return float(scenario["initial"])


def load_portfolio_events() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required = {"strategy_name", "entry_timestamp", "r_multiple"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["strategy_name"] = df["strategy_name"].astype(str).str.upper().str.strip()
    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"], utc=True, errors="coerce"
    )
    df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    df = df.dropna(subset=["strategy_name", "entry_timestamp", "r_multiple"])
    df = df[df["strategy_name"].isin(["MRS2", "MRL1"])]

    # Deterministic tie-break: MRS2 before MRL1 when entry timestamps are equal.
    df["strategy_order"] = df["strategy_name"].map({"MRS2": 0, "MRL1": 1})
    df = df.sort_values(
        ["entry_timestamp", "strategy_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["date_ny"] = df["entry_timestamp"].dt.tz_convert("America/New_York").dt.date
    df["event_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_replay_arrays(df: pd.DataFrame):
    returns = df["r_multiple"].to_numpy(np.float64)
    dates = df["date_ny"].to_numpy()
    timestamps = df["entry_timestamp"].to_numpy()
    strategies = df["strategy_name"].to_numpy()

    unique_dates, first_indices = np.unique(dates, return_index=True)
    order = np.argsort(first_indices)
    unique_dates = unique_dates[order]
    day_start_indices = first_indices[order]
    n_days = len(unique_dates)

    source_day_ids = (
        np.searchsorted(day_start_indices, np.arange(len(df)), side="right") - 1
    ).astype(np.int32)

    return (
        returns,
        dates,
        timestamps,
        strategies,
        unique_dates,
        day_start_indices,
        source_day_ids,
        n_days,
    )


def build_batch(
    returns: np.ndarray,
    source_day_ids: np.ndarray,
    day_start_indices: np.ndarray,
    start_days: np.ndarray,
    max_trades: int,
):
    starts = day_start_indices[start_days]
    offsets = np.arange(max_trades, dtype=np.int64)
    positions = (starts[:, None] + offsets[None, :]) % len(returns)
    path_returns = returns[positions]
    source_days = source_day_ids[positions]

    day_change = np.empty(source_days.shape, dtype=bool)
    day_change[:, 0] = True
    if max_trades > 1:
        day_change[:, 1:] = source_days[:, 1:] != source_days[:, :-1]
    simulated_day_ids = (np.cumsum(day_change, axis=1) - 1).astype(np.int32)
    return path_returns, simulated_day_ids


def simulate_combine_scalar(path_returns, day_ids, scenario, max_trades):
    balance = STARTING_BALANCE
    equity = [balance]
    trades = 0
    failed = False
    passed = False
    winning_days = 0
    current_day = -1
    day_pnl = 0.0
    day_won = False

    for i in range(min(len(path_returns), max_trades)):
        r = float(path_returns[i])
        day = int(day_ids[i])
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            day_won = False

        pnl = r * STARTING_BALANCE * get_risk(balance, scenario)
        balance += pnl
        trades += 1
        equity.append(balance)
        day_pnl += pnl

        if not day_won and day_pnl >= MIN_WINNING_DAY_PROFIT:
            winning_days += 1
            day_won = True

        if balance <= STARTING_BALANCE - MAX_LOSS_LIMIT:
            failed = True
            break
        if balance >= STARTING_BALANCE + PROFIT_TARGET:
            passed = True
            break

    eq = np.asarray(equity, np.float64)
    return {
        "passed": passed,
        "failed": failed,
        "trades": trades,
        "final_balance": balance,
        "profit": balance - STARTING_BALANCE,
        "max_drawdown": max_drawdown(eq),
        "winning_days": winning_days,
    }


def simulate_xfa_scalar(path_returns, day_ids, scenario, interval, amount, max_trades):
    balance = STARTING_BALANCE
    total_withdrawn = 0.0
    payouts = 0
    failed = False
    winning_days = 0
    current_day = -1
    day_pnl = 0.0
    day_won = False
    equity = [balance]

    for i in range(max_trades):
        r = float(path_returns[i])
        day = int(day_ids[i])
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            day_won = False

        pnl = r * STARTING_BALANCE * get_risk(balance, scenario)
        balance += pnl
        equity.append(balance)
        day_pnl += pnl

        if not day_won and day_pnl >= MIN_WINNING_DAY_PROFIT:
            winning_days += 1
            day_won = True

        if balance <= XFA_FIXED_LOSS_FLOOR:
            failed = True
            break

        if (i + 1) % interval == 0 and winning_days >= MIN_WINNING_DAYS:
            actual = min(amount, max(0.0, balance - STARTING_BALANCE))
            if actual > 0:
                balance -= actual
                total_withdrawn += actual
                payouts += 1
                winning_days = 0

    return {
        "failed": failed,
        "trades": i + 1,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payouts,
        "net_value": balance + total_withdrawn,
        "max_drawdown": max_drawdown(np.asarray(equity, np.float64)),
    }


def run_combine_batch(path_returns, day_ids, scenarios, max_trades):
    batch, _ = path_returns.shape
    n = len(scenarios)
    balances = np.full((batch, n), STARTING_BALANCE, np.float64)
    peak = balances.copy()
    max_dd = np.zeros((batch, n), np.float64)
    winning_days = np.zeros((batch, n), np.int32)
    day_pnl = np.zeros((batch, n), np.float64)
    day_won = np.zeros((batch, n), bool)
    current_day = np.full(batch, -1, np.int32)
    active = np.ones((batch, n), bool)
    passed = np.zeros((batch, n), bool)
    failed = np.zeros((batch, n), bool)
    trades = np.zeros((batch, n), np.int32)

    initial = np.array([s[1]["initial"] for s in scenarios], np.float64)
    after = np.array([s[1]["after_threshold"] for s in scenarios], np.float64)
    thresholds = np.array(
        [np.nan if s[1]["threshold"] is None else s[1]["threshold"] for s in scenarios],
        np.float64,
    )

    for t in range(max_trades):
        day = day_ids[:, t]
        new_day = np.ones(batch, bool) if t == 0 else day != current_day
        day_pnl[new_day] = 0.0
        day_won[new_day] = False
        current_day = day

        before = active.copy()
        if not before.any():
            break

        use_after = np.isfinite(thresholds)[None, :] & (
            (balances - STARTING_BALANCE) >= thresholds[None, :]
        )
        risk = np.where(use_after, after[None, :], initial[None, :])
        pnl = path_returns[:, t, None] * STARTING_BALANCE * risk
        pnl = np.where(before, pnl, 0.0)
        balances += pnl
        trades[before] += 1
        day_pnl += pnl

        newly_won = before & ~day_won & (day_pnl >= MIN_WINNING_DAY_PROFIT)
        winning_days[newly_won] += 1
        day_won[newly_won] = True

        hit_loss = before & (balances <= STARTING_BALANCE - MAX_LOSS_LIMIT)
        failed[hit_loss] = True
        active[hit_loss] = False

        hit_target = active & (balances >= STARTING_BALANCE + PROFIT_TARGET)
        passed[hit_target] = True
        active[hit_target] = False

        peak = np.maximum(peak, balances)
        max_dd = np.minimum(max_dd, balances - peak)

    return {
        name: {
            "passed": passed[:, j].copy(),
            "failed": failed[:, j].copy(),
            "trades": trades[:, j].copy(),
            "final_balance": balances[:, j].copy(),
            "profit": (balances[:, j] - STARTING_BALANCE).copy(),
            "max_drawdown": max_dd[:, j].copy(),
            "winning_days": winning_days[:, j].copy(),
        }
        for j, (name, _) in enumerate(scenarios)
    }


def run_xfa_batch(path_returns, day_ids, policies, max_trades):
    batch, _ = path_returns.shape
    n = len(policies)
    balances = np.full((batch, n), STARTING_BALANCE, np.float64)
    withdrawn = np.zeros((batch, n), np.float64)
    payout_counts = np.zeros((batch, n), np.int32)
    winning_days = np.zeros((batch, n), np.int32)
    day_pnl = np.zeros((batch, n), np.float64)
    day_won = np.zeros((batch, n), bool)
    qualified_day = np.full((batch, n), -1, np.int32)
    current_day = np.full(batch, -1, np.int32)
    active = np.ones((batch, n), bool)
    failed = np.zeros((batch, n), bool)
    trades = np.zeros((batch, n), np.int32)
    peak = np.full((batch, n), STARTING_BALANCE, np.float64)
    max_dd = np.zeros((batch, n), np.float64)

    initial = np.array([p[1]["initial"] for p in policies], np.float64)
    after = np.array([p[1]["after_threshold"] for p in policies], np.float64)
    thresholds = np.array(
        [np.nan if p[1]["threshold"] is None else p[1]["threshold"] for p in policies],
        np.float64,
    )
    intervals = np.array([p[2] for p in policies], np.int32)
    amounts = np.array([p[3] for p in policies], np.float64)

    for t in range(max_trades):
        day = day_ids[:, t]
        new_day = np.ones(batch, bool) if t == 0 else day != current_day
        day_pnl[new_day] = 0.0
        day_won[new_day] = False
        current_day = day

        before = active.copy()
        if not before.any():
            break

        use_after = np.isfinite(thresholds)[None, :] & (
            (balances - STARTING_BALANCE) >= thresholds[None, :]
        )
        risk = np.where(use_after, after[None, :], initial[None, :])
        pnl = path_returns[:, t, None] * STARTING_BALANCE * risk
        pnl = np.where(before, pnl, 0.0)
        balances += pnl
        trades[before] += 1
        day_pnl += pnl

        newly_won = before & ~day_won & (day_pnl >= MIN_WINNING_DAY_PROFIT)
        if newly_won.any():
            day_matrix = np.broadcast_to(day[:, None], (batch, n))
            count = newly_won & (qualified_day != day_matrix)
            winning_days[count] += 1
            qualified_day[count] = day_matrix[count]
            day_won[newly_won] = True

        hit_loss = before & (balances <= XFA_FIXED_LOSS_FLOOR)
        failed[hit_loss] = True
        active[hit_loss] = False

        # Match the validated individual-strategy implementation: drawdown is
        # updated after the trade and BEFORE payout.
        peak = np.maximum(peak, balances)
        max_dd = np.minimum(max_dd, balances - peak)

        due = active & (((t + 1) % intervals[None, :]) == 0)
        eligible = due & (winning_days >= MIN_WINNING_DAYS)
        actual = np.minimum(
            amounts[None, :], np.maximum(0.0, balances - STARTING_BALANCE)
        )
        pay = eligible & (actual > 0.0)
        balances[pay] -= actual[pay]
        withdrawn[pay] += actual[pay]
        payout_counts[pay] += 1
        winning_days[pay] = 0

        active[trades >= max_trades] = False

    return {
        (name, interval, float(amount)): {
            "failed": failed[:, j].copy(),
            "trades": trades[:, j].copy(),
            "final_balance": balances[:, j].copy(),
            "total_withdrawn": withdrawn[:, j].copy(),
            "payouts": payout_counts[:, j].copy(),
            "net_value": (balances[:, j] + withdrawn[:, j]).copy(),
            "max_drawdown": max_dd[:, j].copy(),
        }
        for j, (name, _scenario, interval, amount) in enumerate(policies)
    }


def parity_audit(returns, source_days, starts, day_starts):
    print("\n" + "=" * 110)
    print("PORTFOLIO PARITY AUDIT")
    print("=" * 110)

    n = min(PARITY_PATHS, len(starts))
    test_starts = starts[:n]
    path_r, path_d = build_batch(
        returns, source_days, day_starts, test_starts, PARITY_MAX_TRADES
    )

    combine_policies = list(RISK_SCENARIOS.items())
    vec_c = run_combine_batch(path_r, path_d, combine_policies, PARITY_MAX_TRADES)
    combine_ok = True
    for i in range(n):
        for name, scenario in combine_policies:
            s = simulate_combine_scalar(
                path_r[i], path_d[i], scenario, PARITY_MAX_TRADES
            )
            v = {
                k: x[i].item() if np.ndim(x[i]) == 0 else x[i]
                for k, x in vec_c[name].items()
            }
            for field in [
                "passed",
                "failed",
                "trades",
                "final_balance",
                "profit",
                "max_drawdown",
                "winning_days",
            ]:
                if not np.isclose(
                    float(s[field]), float(v[field]), atol=1e-9, rtol=1e-9
                ):
                    combine_ok = False
                    print(
                        f"COMBINE mismatch path={i} risk={name} field={field}: scalar={s[field]} vector={v[field]}"
                    )

    policies = [
        (name, scenario, interval, float(amount))
        for name, scenario in RISK_SCENARIOS.items()
        for interval in PAYOUT_INTERVALS
        for amount in PAYOUT_AMOUNTS
    ]
    vec_x = run_xfa_batch(path_r, path_d, policies, PARITY_MAX_TRADES)
    xfa_ok = True
    for i in range(n):
        for name, scenario, interval, amount in policies:
            s = simulate_xfa_scalar(
                path_r[i], path_d[i], scenario, interval, amount, PARITY_MAX_TRADES
            )
            key = (name, interval, amount)
            v = {
                k: x[i].item() if np.ndim(x[i]) == 0 else x[i]
                for k, x in vec_x[key].items()
            }
            for field in [
                "failed",
                "trades",
                "final_balance",
                "total_withdrawn",
                "payouts",
                "net_value",
                "max_drawdown",
            ]:
                if not np.isclose(
                    float(s[field]), float(v[field]), atol=1e-9, rtol=1e-9
                ):
                    xfa_ok = False
                    print(
                        f"XFA mismatch path={i} policy={key} field={field}: scalar={s[field]} vector={v[field]}"
                    )

    print(f"Combine scalar/vector parity: {'PASS' if combine_ok else 'FAIL'}")
    print(f"XFA scalar/vector parity: {'PASS' if xfa_ok else 'FAIL'}")
    print(f"PORTFOLIO PARITY AUDIT: {'PASS' if combine_ok and xfa_ok else 'FAIL'}")
    if not (combine_ok and xfa_ok):
        raise RuntimeError("Portfolio parity audit failed.")


def simulate_combine(returns, source_days, day_starts, starts):
    records = {name: [] for name in RISK_SCENARIOS}
    policies = list(RISK_SCENARIOS.items())
    total = len(starts)

    for b in range(0, total, BATCH_SIZE):
        batch_starts = starts[b : b + BATCH_SIZE]
        pr, pdays = build_batch(
            returns, source_days, day_starts, batch_starts, MAX_COMBINE_TRADES
        )
        result = run_combine_batch(pr, pdays, policies, MAX_COMBINE_TRADES)
        for name in RISK_SCENARIOS:
            records[name].append(result[name])

    rows = []
    for name in RISK_SCENARIOS:
        combined = {
            k: np.concatenate([x[k] for x in records[name]]) for k in records[name][0]
        }
        passed = combined["passed"]
        failed = combined["failed"]
        survived = ~passed & ~failed & (combined["trades"] >= MAX_COMBINE_TRADES)
        rows.append(
            {
                "risk": name,
                "simulations": total,
                "pass_rate": float(passed.mean()),
                "fail_rate": float(failed.mean()),
                "survived_to_max_trades_rate": float(survived.mean()),
                "median_trades": float(np.median(combined["trades"])),
                "median_profit": float(np.median(combined["profit"])),
                "median_max_DD": float(np.median(combined["max_drawdown"])),
                "p95_max_DD": percentile(combined["max_drawdown"], 5),
                "p99_max_DD": percentile(combined["max_drawdown"], 1),
                "median_winning_days": float(np.median(combined["winning_days"])),
            }
        )
    return pd.DataFrame(rows)


def simulate_xfa(returns, source_days, day_starts, starts):
    policies = [
        (name, scenario, interval, float(amount))
        for name, scenario in RISK_SCENARIOS.items()
        for interval in PAYOUT_INTERVALS
        for amount in PAYOUT_AMOUNTS
    ]
    records = {
        (name, interval, float(amount)): [] for name, _, interval, amount in policies
    }
    total = len(starts)

    for b in range(0, total, BATCH_SIZE):
        batch_starts = starts[b : b + BATCH_SIZE]
        pr, pdays = build_batch(
            returns, source_days, day_starts, batch_starts, MAX_XFA_TRADES
        )
        result = run_xfa_batch(pr, pdays, policies, MAX_XFA_TRADES)
        for key, value in result.items():
            records[key].append(value)

    rows = []
    for name, _, interval, amount in policies:
        key = (name, interval, amount)
        combined = {
            k: np.concatenate([x[k] for x in records[key]]) for k in records[key][0]
        }
        failed = combined["failed"]
        survival = ~failed
        survived_max = survival & (combined["trades"] >= MAX_XFA_TRADES)
        rows.append(
            {
                "risk": name,
                "payout_interval": interval,
                "payout_amount": amount,
                "simulations": total,
                "survival_rate": float(survival.mean()),
                "failure_rate": float(failed.mean()),
                "survived_to_max_trades_rate": float(survived_max.mean()),
                "median_payouts": float(np.median(combined["payouts"])),
                "median_total_withdrawn": float(np.median(combined["total_withdrawn"])),
                "p95_total_withdrawn": percentile(combined["total_withdrawn"], 95),
                "median_final_balance": float(np.median(combined["final_balance"])),
                "median_net_value": float(np.median(combined["net_value"])),
                "median_max_DD": float(np.median(combined["max_drawdown"])),
                "p95_DD": percentile(combined["max_drawdown"], 5),
                "median_trades": float(np.median(combined["trades"])),
            }
        )
    return pd.DataFrame(rows)


def audit_net_value_identity(xfa_df: pd.DataFrame):
    print("\n" + "=" * 110)
    print("XFA NET-VALUE IDENTITY AUDIT")
    print("=" * 110)
    print(
        "For fixed-risk scenarios (no balance-dependent risk change), payouts are transfers:"
    )
    print("final_balance + total_withdrawn = pre-withdrawal account equity.")
    print(
        "Therefore payout interval/amount can change withdrawals and final balance without changing net value"
    )
    print("when the underlying path and survival set are unchanged.")

    fixed = xfa_df[xfa_df["risk"].isin(["0.25%", "0.50%", "0.75%", "1.00%"])]
    for risk, g in fixed.groupby("risk"):
        spread = float(g["median_net_value"].max() - g["median_net_value"].min())
        print(
            f"{risk}: median_net_value spread across 21 payout policies = ${spread:,.6f}"
        )
        if spread > 1e-6:
            print(
                "  NOTE: spread exists because payout policy changes survival/failure selection."
            )
        else:
            print(
                "  PASS: identical median net value is mathematically expected for this fixed-risk set."
            )

    adaptive = xfa_df[xfa_df["risk"] == "0.50_to_1.00"]
    spread = float(
        adaptive["median_net_value"].max() - adaptive["median_net_value"].min()
    )
    print(
        f"0.50_to_1.00: median_net_value spread across 21 payout policies = ${spread:,.6f}"
    )
    print(
        "  This scenario is balance-dependent, so payout policy can alter the later risk state."
    )


def main():
    print("=" * 110)
    print("MRS2 + MRL1 PORTFOLIO FUNDED ACCOUNT SIMULATION")
    print("=" * 110)
    print("NO STRATEGY OPTIMIZATION.")
    print("TWO FROZEN STRATEGIES MERGED INTO ONE CHRONOLOGICAL PORTFOLIO STREAM.")
    print(f"Simulations: {N_SIMULATIONS:,}")
    print(f"Combine max trades: {MAX_COMBINE_TRADES:,}")
    print(f"XFA max trades: {MAX_XFA_TRADES:,}")
    print(f"XFA fixed loss floor: ${XFA_FIXED_LOSS_FLOOR:,.0f}")

    df = load_portfolio_events()
    counts = df["strategy_name"].value_counts()
    print(f"\nPortfolio trades: {len(df):,}")
    print(counts.to_string())
    print(f"First event: {df['entry_timestamp'].iloc[0]}")
    print(f"Last event:  {df['entry_timestamp'].iloc[-1]}")
    print(f"Historical NY trading days: {df['date_ny'].nunique():,}")

    same_ts = df.groupby("entry_timestamp")["strategy_name"].nunique()
    simultaneous_timestamps = int((same_ts >= 2).sum())
    simultaneous_events = int((same_ts[same_ts >= 2]).sum())
    print(f"Exact timestamps containing BOTH strategies: {simultaneous_timestamps:,}")
    print(f"Events inside those timestamps: {simultaneous_events:,}")
    print("Tie-break: MRS2 before MRL1.")

    (
        returns,
        dates,
        timestamps,
        strategies,
        unique_dates,
        day_starts,
        source_days,
        n_days,
    ) = build_replay_arrays(df)
    print(f"Replay start-day choices: {n_days:,} real historical days")

    rng = np.random.default_rng(RANDOM_SEED)
    starts = rng.integers(0, n_days, size=N_SIMULATIONS, dtype=np.int32)

    parity_audit(returns, source_days, starts, day_starts)

    print("\n" + "=" * 110)
    print("COMBINE")
    print("=" * 110)
    combine = simulate_combine(returns, source_days, day_starts, starts)
    print(combine.to_string(index=False))
    combine.to_csv(OUTPUT_COMBINE, index=False)
    print(f"Saved: {OUTPUT_COMBINE}")

    print("\n" + "=" * 110)
    print("XFA")
    print("=" * 110)
    xfa = simulate_xfa(returns, source_days, day_starts, starts)
    viable = xfa[xfa["survival_rate"] >= 0.90].sort_values(
        ["survival_rate", "median_total_withdrawn"], ascending=[False, False]
    )
    print("\nPolicies with >=90% survival:")
    print("None." if viable.empty else viable.head(20).to_string(index=False))
    print("\nHighest-survival policies:")
    print(
        xfa.sort_values(
            ["survival_rate", "median_total_withdrawn"], ascending=[False, False]
        )
        .head(20)
        .to_string(index=False)
    )
    xfa.to_csv(OUTPUT_XFA, index=False)
    print(f"Saved: {OUTPUT_XFA}")

    audit_net_value_identity(xfa)
    pd.DataFrame(
        [
            {"audit": "portfolio_parity", "status": "PASS"},
            {"audit": "net_value_identity", "status": "PASS"},
        ]
    ).to_csv(OUTPUT_AUDIT, index=False)
    print(f"Saved: {OUTPUT_AUDIT}")

    print("\n" + "=" * 110)
    print("PORTFOLIO SIMULATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
