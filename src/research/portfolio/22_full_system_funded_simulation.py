from __future__ import annotations

import math

import sys

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd
# =============================================================================*
# PATHS*
# =============================================================================*

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "src" / "research" / "results" / "portfolio" / "funded"

MR_RESULTS = PROJECT_ROOT / "src" / "research" / "mean_reversion" / "results"

MR_INPUT = MR_RESULTS / "research_08aa_modular_reproduction_trades.csv"

S2R_INPUT = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

ORB_INPUT = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CANONICAL FUNDED-SIMULATION CONFIGURATION
# =============================================================================

STARTING_BALANCE = 50_000.0
PROFIT_TARGET = 3_000.0
MAX_LOSS_LIMIT = 2_000.0

RISK_SCENARIOS: dict[str, dict[str, float | None]] = {
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
RANDOM_SEED = 42
MAX_COMBINE_TRADES = 500
MAX_XFA_TRADES = 2_000
BATCH_SIZE = 1_000

PARITY_PATHS = 25
PARITY_MAX_TRADES = 250

PAYOUT_INTERVALS = [20, 21, 22]
PAYOUT_AMOUNTS = [500.0, 750.0, 1_000.0, 1_250.0, 1_500.0, 1_750.0, 2_000.0]
MIN_WINNING_DAYS = 5
MIN_WINNING_DAY_PROFIT = 150.0
XFA_STARTING_BALANCE = STARTING_BALANCE
XFA_MAX_LOSS_LIMIT = MAX_LOSS_LIMIT
XFA_FIXED_LOSS_FLOOR = XFA_STARTING_BALANCE - XFA_MAX_LOSS_LIMIT

# =============================================================================*
# LOAD MODULAR TRADES*
# =============================================================================*


# =============================================================================
# CORE HELPERS
# =============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def percentile(values: np.ndarray | pd.Series, p: float) -> float:
    return float(np.percentile(values, p))


def get_risk(
    balance: float,
    scenario: dict[str, float | None],
) -> float:
    """
    Return the risk percentage for the current account state.

    Risk is calculated from STARTING_BALANCE, matching the funded-account
    model used by the original ORB funded simulation.
    """
    threshold = scenario["threshold"]

    if threshold is not None:
        profit_from_start = balance - STARTING_BALANCE
        if profit_from_start >= float(threshold):
            return float(scenario["after_threshold"])

    return float(scenario["initial"])


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough dollar drawdown."""
    if len(equity) == 0:
        return 0.0

    running_peak = np.maximum.accumulate(equity)
    drawdown = equity - running_peak
    return float(drawdown.min())


def _read_trade_file(path: Path, strategy: str) -> pd.DataFrame:
    """Load one frozen strategy stream and normalize it."""
    if not path.exists():
        raise FileNotFoundError(f"\\nMissing {strategy} input file:\\n{path}")

    df = pd.read_csv(path)

    if "entry_timestamp" not in df.columns:
        raise ValueError(f"{strategy}: trade file is missing entry_timestamp.")

    r_column = next(
        (c for c in ["r_multiple", "r", "net_R"] if c in df.columns),
        None,
    )
    if r_column is None:
        raise ValueError(
            f"{strategy}: trade file has no R-multiple column. "
            "Expected r_multiple, r, or net_R."
        )

    out = df.copy()
    out["entry_timestamp"] = pd.to_datetime(
        out["entry_timestamp"],
        utc=True,
        errors="coerce",
    )
    out["r_multiple"] = pd.to_numeric(out[r_column], errors="coerce")
    out["strategy_name"] = strategy
    out = out.dropna(subset=["entry_timestamp", "r_multiple"])

    return out[["entry_timestamp", "r_multiple", "strategy_name"]].copy()


def load_trades() -> pd.DataFrame:
    """
    Build the complete funded-account portfolio sequence from the four
    frozen strategies using the exact common OOS only.
    """
    frames = []
    counts = {}

    oos_start = pd.Timestamp(
        "2020-06-23 00:00:00",
        tz="America/New_York",
    )
    oos_end = pd.Timestamp(
        "2026-06-19 23:59:59",
        tz="America/New_York",
    )

    # -------------------------------------------------------------------------
    # MRL1 + MRS2 — authoritative 08AA modular stream
    # -------------------------------------------------------------------------
    if not MR_INPUT.exists():
        raise FileNotFoundError(f"Missing MRL1/MRS2 input file:\n{MR_INPUT}")

    mr = pd.read_csv(MR_INPUT)

    required_mr = {"strategy_name", "entry_timestamp"}
    missing_mr = required_mr - set(mr.columns)
    if missing_mr:
        raise ValueError(f"08AA MR file missing required columns: {sorted(missing_mr)}")

    r_column = next(
        (c for c in ["r_multiple", "r", "net_R"] if c in mr.columns),
        None,
    )
    if r_column is None:
        raise ValueError("08AA MR file has no R-multiple column.")

    mr["strategy_name"] = mr["strategy_name"].astype(str).str.upper().str.strip()
    mr["entry_timestamp"] = pd.to_datetime(
        mr["entry_timestamp"],
        utc=True,
        errors="coerce",
    )
    mr["r_multiple"] = pd.to_numeric(
        mr[r_column],
        errors="coerce",
    )

    mr = (
        mr[mr["strategy_name"].isin(["MRL1", "MRS2"])]
        .dropna(subset=["entry_timestamp", "r_multiple"])
        .copy()
    )

    mr["entry_timestamp_et"] = mr["entry_timestamp"].dt.tz_convert("America/New_York")

    mr = mr[
        (mr["entry_timestamp_et"] >= oos_start) & (mr["entry_timestamp_et"] <= oos_end)
    ].copy()

    for strategy, expected_count in [
        ("MRL1", 430),
        ("MRS2", 863),
    ]:
        subset = mr[mr["strategy_name"] == strategy].copy()
        if len(subset) != expected_count:
            raise RuntimeError(
                f"{strategy} exact-OOS trade-count audit FAILED: "
                f"expected {expected_count}, found {len(subset)}."
            )
        subset["date_ny"] = subset["entry_timestamp_et"].dt.date
        counts[strategy] = len(subset)
        frames.append(
            subset[
                [
                    "entry_timestamp",
                    "r_multiple",
                    "strategy_name",
                    "entry_timestamp_et",
                    "date_ny",
                ]
            ]
        )

    # -------------------------------------------------------------------------
    # S2R — authoritative modular stream
    # -------------------------------------------------------------------------
    s2r = _read_trade_file(S2R_INPUT, "S2R")
    s2r["entry_timestamp_et"] = s2r["entry_timestamp"].dt.tz_convert("America/New_York")
    s2r = s2r[
        (s2r["entry_timestamp_et"] >= oos_start)
        & (s2r["entry_timestamp_et"] <= oos_end)
    ].copy()

    if len(s2r) != 520:
        raise RuntimeError(
            f"S2R exact-OOS trade-count audit FAILED: expected 520, found {len(s2r)}."
        )
    s2r["date_ny"] = s2r["entry_timestamp_et"].dt.date
    counts["S2R"] = len(s2r)
    frames.append(s2r)

    # -------------------------------------------------------------------------
    # ORB — full reconciliation stream, exact common OOS extracted
    # -------------------------------------------------------------------------
    orb = _read_trade_file(ORB_INPUT, "ORB")
    orb["entry_timestamp_et"] = orb["entry_timestamp"].dt.tz_convert("America/New_York")
    orb = orb[
        (orb["entry_timestamp_et"] >= oos_start)
        & (orb["entry_timestamp_et"] <= oos_end)
    ].copy()

    if len(orb) != 1442:
        raise RuntimeError(
            f"ORB exact-OOS trade-count audit FAILED: expected 1442, found {len(orb)}."
        )
    orb["date_ny"] = orb["entry_timestamp_et"].dt.date
    counts["ORB"] = len(orb)
    frames.append(orb)

    portfolio = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["entry_timestamp", "strategy_name"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if portfolio.duplicated(subset=["entry_timestamp", "strategy_name"]).any():
        raise RuntimeError(
            "Portfolio OOS contains duplicate (entry_timestamp, strategy_name)."
        )

    banner("PORTFOLIO DATA PARTITION AUDIT")
    print("-" * 110)
    for strategy in ["MRL1", "S2R", "MRS2", "ORB"]:
        print(f"{strategy:<34} {counts[strategy]:>8,}")
    print(f"{'TOTAL':<34} {len(portfolio):>8,}")
    print()
    print("Exact COMMON OOS: 2020-06-23 through 2026-06-19")
    print("Funded simulation uses EXACT COMMON OOS ONLY.")
    print("Post-OOS is NOT used.")
    print("Individual trades are NOT shuffled.")

    if len(portfolio) != 3255:
        raise RuntimeError(
            f"Portfolio exact-OOS audit FAILED: expected 3255, found {len(portfolio)}."
        )

    return portfolio


class HistoricalSequence:
    def __init__(
        self,
        strategy: str,
        df: pd.DataFrame,
    ) -> None:

        self.strategy = strategy
        # ORB reconciliation file is already strategy-specific.*
        # There is no strategy_name column to filter on.*

        data = df.sort_values(
            "entry_timestamp",
            kind="mergesort",
        ).reset_index(drop=True)

        if data.empty:
            raise ValueError(f"No trades found for {strategy}.")

        self.df = data

        self.returns = data["r_multiple"].to_numpy(dtype=np.float64)

        self.timestamps = data["entry_timestamp"].to_numpy()

        self.dates = data["date_ny"].to_numpy()

        self.n_trades = len(self.returns)

        unique_dates, first_indices = np.unique(
            self.dates,
            return_index=True,
        )

        order = np.argsort(first_indices)

        self.unique_dates = unique_dates[order]

        self.day_start_indices = first_indices[order]

        self.n_days = len(self.unique_dates)

        self.trade_day_id = (
            np.searchsorted(
                self.day_start_indices,
                np.arange(self.n_trades),
                side="right",
            )
            - 1
        ).astype(np.int32)

        if self.day_start_indices[0] != 0:
            raise RuntimeError(
                f"{strategy}: first historical day does not start at trade 0."
            )

        if np.any(np.diff(self.day_start_indices) <= 0):
            raise RuntimeError(f"{strategy}: invalid historical day boundaries.")

        if np.any(np.diff(self.timestamps) < np.timedelta64(0, "s")):
            raise RuntimeError(f"{strategy}: chronological ordering failed.")

    def summary(self) -> dict:

        return {
            "strategy": self.strategy,
            "trades": self.n_trades,
            "days": self.n_days,
            "first_timestamp": str(self.timestamps[0]),
            "last_timestamp": str(self.timestamps[-1]),
        }

    def build_batch(
        self,
        start_days: np.ndarray,
        max_trades: int,
    ):

        if len(start_days) == 0:
            raise ValueError("Empty start-day array.")

        start_indices = self.day_start_indices[start_days]

        offsets = np.arange(
            max_trades,
            dtype=np.int64,
        )

        positions = (start_indices[:, None] + offsets[None, :]) % self.n_trades

        path_returns = self.returns[positions]

        source_day_ids = self.trade_day_id[positions]

        day_change = np.empty(
            (
                len(start_days),
                max_trades,
            ),
            dtype=bool,
        )

        day_change[:, 0] = True

        if max_trades > 1:
            day_change[:, 1:] = source_day_ids[:, 1:] != source_day_ids[:, :-1]

        simulated_day_ids = (
            np.cumsum(
                day_change,
                axis=1,
            )
            - 1
        ).astype(np.int32)

        return (
            path_returns,
            simulated_day_ids,
        )


# =============================================================================*
# SCALAR REFERENCE ENGINE*
# =============================================================================*
# *
# This is intentionally simple.*
# *
# It exists as the correctness reference for the vectorized engine.*
# *
# We do NOT use this for the 50k production run.*
# *


def simulate_combine_scalar(
    path_returns: np.ndarray,
    day_ids: np.ndarray,
    scenario: dict[str, float | None],
    max_trades: int,
) -> dict[str, Any]:

    balance = STARTING_BALANCE

    equity = [balance]

    trades = 0

    failed = False

    passed = False

    winning_days = 0

    current_day = -1

    day_pnl = 0.0

    day_won = False

    for i in range(
        min(
            len(path_returns),
            max_trades,
        )
    ):
        r = float(path_returns[i])

        day = int(day_ids[i])

        if day != current_day:
            current_day = day

            day_pnl = 0.0

            day_won = False

        risk_pct = get_risk(
            balance,
            scenario,
        )

        risk_dollars = STARTING_BALANCE * risk_pct

        pnl = r * risk_dollars

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

    equity_array = np.asarray(
        equity,
        dtype=np.float64,
    )

    return {
        "passed": passed,
        "failed": failed,
        "survived_to_max_trades": (not passed and not failed and trades >= max_trades),
        "trades": trades,
        "final_balance": balance,
        "profit": (balance - STARTING_BALANCE),
        "max_drawdown": max_drawdown(equity_array),
        "loss_from_start": (balance - STARTING_BALANCE),
        "winning_days": winning_days,
    }


def simulate_xfa_scalar(
    path_returns: np.ndarray,
    day_ids: np.ndarray,
    scenario: dict[str, float | None],
    payout_interval: int,
    payout_amount: float,
    max_trades: int,
) -> dict[str, Any]:

    balance = XFA_STARTING_BALANCE

    loss_floor = XFA_FIXED_LOSS_FLOOR

    trades = 0

    total_withdrawn = 0.0

    payout_count = 0

    failed = False

    failure_reason = None

    winning_days = 0

    current_day = -1

    day_pnl = 0.0

    day_won = False
    # This prevents the same simulated day from being counted again after*
    # a payout clears the winning-day count.*

    qualified_day_token = -1

    equity = [balance]

    while trades < max_trades:
        i = trades

        r = float(path_returns[i])

        day = int(day_ids[i])

        if day != current_day:
            current_day = day

            day_pnl = 0.0

            day_won = False

        risk_pct = get_risk(
            balance,
            scenario,
        )

        risk_dollars = XFA_STARTING_BALANCE * risk_pct

        pnl = r * risk_dollars

        balance += pnl

        trades += 1

        equity.append(balance)

        day_pnl += pnl

        if not day_won and day_pnl >= MIN_WINNING_DAY_PROFIT:
            day_won = True
            # Only count this day once.*

            if qualified_day_token != current_day:
                winning_days += 1

                qualified_day_token = current_day
        # ---------------------------------------------------------------------*
        # MLL CHECK COMES BEFORE PAYOUT*
        # ---------------------------------------------------------------------*

        if balance <= loss_floor:
            failed = True

            failure_reason = "MLL"

            break
        # ---------------------------------------------------------------------*
        # PAYOUT CHECK*
        # ---------------------------------------------------------------------*

        if trades % payout_interval == 0:
            if winning_days >= MIN_WINNING_DAYS:
                profit_above_start = max(
                    0.0,
                    balance - XFA_STARTING_BALANCE,
                )

                actual_payout = min(
                    payout_amount,
                    profit_above_start,
                )

                if actual_payout > 0:
                    balance -= actual_payout

                    total_withdrawn += actual_payout

                    payout_count += 1
                    # Consume the qualifying days.*

                    winning_days = 0
    # ---------------------------------------------------------------------*
    # Continue.*
    # ---------------------------------------------------------------------*

    equity_array = np.asarray(
        equity,
        dtype=np.float64,
    )

    return {
        "failed": failed,
        "failure_reason": failure_reason,
        "survived_to_max_trades": (not failed and trades >= max_trades),
        "trades": trades,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payouts": payout_count,
        "net_value": (balance + total_withdrawn),
        "max_drawdown": max_drawdown(equity_array),
        "loss_from_start": (balance - XFA_STARTING_BALANCE),
        "winning_days": winning_days,
    }


# =============================================================================*
# VECTOR SAMPLE PATH ENGINE*
# =============================================================================*


def run_xfa_batch(
    path_returns: np.ndarray,
    day_ids: np.ndarray,
    scenarios: list[
        tuple[
            str,
            dict[str, float | None],
            int,
            float,
        ]
    ],
    max_trades: int,
) -> dict[
    tuple[str, int, float],
    dict[str, np.ndarray],
]:
    """

    *       Vectorized XFA engine.**

    *       Input:**

    *           path_returns:**

    *               shape = (batch, max_trades)**

    *           day_ids:**

    *               shape = (batch, max_trades)**

    *       One return path is shared across all account policies.**

    *       This is important:**

    *           the policy changes the account trajectory,**

    *           NOT the underlying market/trade path.**

    *"""

    batch_size = path_returns.shape[0]

    n_policies = len(scenarios)

    balances = np.full(
        (
            batch_size,
            n_policies,
        ),
        XFA_STARTING_BALANCE,
        dtype=np.float64,
    )

    total_withdrawn = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.float64,
    )

    payout_counts = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.int32,
    )

    winning_days = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.int32,
    )

    day_pnl = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.float64,
    )

    day_won = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )
    # Current simulated day for each path.*

    current_day = np.full(
        (batch_size,),
        -1,
        dtype=np.int32,
    )
    # Day token on which a qualifying day was last counted.*

    qualified_day_token = np.full(
        (
            batch_size,
            n_policies,
        ),
        -1,
        dtype=np.int32,
    )

    active = np.ones(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    failed = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    trades_completed = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.int32,
    )
    # Peak equity for drawdown.*

    peak_balance = np.full(
        (
            batch_size,
            n_policies,
        ),
        XFA_STARTING_BALANCE,
        dtype=np.float64,
    )

    max_dd = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.float64,
    )
    # -------------------------------------------------------------------------*
    # PREPARE POLICY VECTORS*
    # -------------------------------------------------------------------------*

    initial_risk = np.asarray(
        [
            float(scenario["initial"])
            for (
                _,
                scenario,
                _,
                _,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    after_threshold_risk = np.asarray(
        [
            float(scenario["after_threshold"])
            for (
                _,
                scenario,
                _,
                _,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    thresholds = np.asarray(
        [
            (np.nan if scenario["threshold"] is None else float(scenario["threshold"]))
            for (
                _,
                scenario,
                _,
                _,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    payout_intervals = np.asarray(
        [
            interval
            for (
                _,
                _,
                interval,
                _,
            ) in scenarios
        ],
        dtype=np.int32,
    )

    payout_amounts = np.asarray(
        [
            amount
            for (
                _,
                _,
                _,
                amount,
            ) in scenarios
        ],
        dtype=np.float64,
    )
    # -------------------------------------------------------------------------*
    # TRADE LOOP*
    # -------------------------------------------------------------------------*
    # *
    # We still iterate over chronological trades.*
    # *
    # This is deliberate.*
    # *
    # The expensive dimension is the number of simulations/policies, and*
    # those are vectorized.*
    # *
    # The temporal dimension must remain ordered.*
    # *

    for t in range(max_trades):
        returns_t = path_returns[:, t]

        day_t = day_ids[:, t]
        # ---------------------------------------------------------------------*
        # DAY TRANSITION*
        # ---------------------------------------------------------------------*

        if t == 0:
            new_day = np.ones(
                batch_size,
                dtype=bool,
            )

        else:
            new_day = day_t != current_day

        if np.any(new_day):
            day_pnl[
                new_day,
                :,
            ] = 0.0

            day_won[
                new_day,
                :,
            ] = False

        current_day = day_t
        # ---------------------------------------------------------------------*
        # ACTIVE POLICIES ONLY*
        # ---------------------------------------------------------------------*

        active_before = active.copy()

        if not np.any(active_before):
            break
        # ---------------------------------------------------------------------*
        # RISK*
        # ---------------------------------------------------------------------*

        threshold_active = np.isfinite(thresholds)

        profit_from_start = balances - STARTING_BALANCE

        use_after_threshold = threshold_active[None, :] & (
            profit_from_start >= thresholds[None, :]
        )

        risk_pct = np.where(
            use_after_threshold,
            after_threshold_risk[
                None,
                :,
            ],
            initial_risk[
                None,
                :,
            ],
        )

        risk_dollars = XFA_STARTING_BALANCE * risk_pct

        pnl = returns_t[:, None] * risk_dollars

        pnl = np.where(
            active_before,
            pnl,
            0.0,
        )
        # ---------------------------------------------------------------------*
        # BALANCE*
        # ---------------------------------------------------------------------*

        balances += pnl

        trades_completed[active_before] += 1
        # ---------------------------------------------------------------------*
        # DAY P&L*
        # ---------------------------------------------------------------------*

        day_pnl += pnl

        newly_winning = active_before & ~day_won & (day_pnl >= MIN_WINNING_DAY_PROFIT)

        if np.any(newly_winning):
            # Count this day only once.*
            # *
            # The same simulated day cannot be counted again after a payout.*
            # *

            not_already_counted = qualified_day_token != day_t[:, None]

            count_mask = newly_winning & not_already_counted

            winning_days[count_mask] += 1

            current_day_matrix = np.broadcast_to(
                day_t[:, None],
                (
                    batch_size,
                    n_policies,
                ),
            )

        qualified_day_token[count_mask] = current_day_matrix[count_mask]

        day_won[newly_winning] = True
        # ---------------------------------------------------------------------*
        # MLL*
        # ---------------------------------------------------------------------*

        hit_mll = active_before & (balances <= XFA_FIXED_LOSS_FLOOR)

        if np.any(hit_mll):
            failed[hit_mll] = True

            active[hit_mll] = False
        # ---------------------------------------------------------------------*
        # DRAW DOWN — BEFORE PAYOUT*
        # *
        # Match the scalar reference exactly: the trade equity point is*
        # recorded before the payout withdrawal is applied.*
        # *

        peak_balance = np.maximum(
            peak_balance,
            balances,
        )

        current_dd = balances - peak_balance

        max_dd = np.minimum(
            max_dd,
            current_dd,
        )
        # ---------------------------------------------------------------------*
        # PAYOUT*
        # ---------------------------------------------------------------------*
        # The payout itself is not treated as a trading drawdown event.*

        payout_due = active & ((t + 1) % payout_intervals[None, :] == 0)

        eligible = payout_due & (winning_days >= MIN_WINNING_DAYS)

        profit_above_start = np.maximum(
            0.0,
            balances - XFA_STARTING_BALANCE,
        )

        actual_payout = np.minimum(
            payout_amounts[None, :],
            profit_above_start,
        )

        payout_mask = eligible & (actual_payout > 0.0)

        if np.any(payout_mask):
            balances[payout_mask] -= actual_payout[payout_mask]

            total_withdrawn[payout_mask] += actual_payout[payout_mask]

            payout_counts[payout_mask] += 1

            winning_days[payout_mask] = 0
        # COMPLETION*
        # ---------------------------------------------------------------------*

        reached_max_trades = trades_completed >= max_trades

        active[reached_max_trades] = False
    # -------------------------------------------------------------------------*
    # OUTPUT*
    # -------------------------------------------------------------------------*

    result: dict[
        tuple[str, int, float],
        dict[str, np.ndarray],
    ] = {}

    for policy_index, (
        name,
        _scenario,
        interval,
        amount,
    ) in enumerate(scenarios):
        key = (
            name,
            interval,
            float(amount),
        )

        policy_failed = failed[
            :,
            policy_index,
        ]

        policy_trades = trades_completed[
            :,
            policy_index,
        ]

        result[key] = {
            "failed": policy_failed.copy(),
            "trades": policy_trades.copy(),
            "final_balance": (
                balances[
                    :,
                    policy_index,
                ].copy()
            ),
            "total_withdrawn": (
                total_withdrawn[
                    :,
                    policy_index,
                ].copy()
            ),
            "payouts": (
                payout_counts[
                    :,
                    policy_index,
                ].copy()
            ),
            "winning_days": (
                winning_days[
                    :,
                    policy_index,
                ].copy()
            ),
            "max_drawdown": (
                max_dd[
                    :,
                    policy_index,
                ].copy()
            ),
        }

    return result


# =============================================================================*
# COMBINE VECTOR ENGINE*
# =============================================================================*


def run_combine_batch(
    path_returns: np.ndarray,
    day_ids: np.ndarray,
    scenarios: list[
        tuple[
            str,
            dict[str, float | None],
        ]
    ],
    max_trades: int,
) -> dict[
    str,
    dict[str, np.ndarray],
]:

    batch_size = path_returns.shape[0]

    n_policies = len(scenarios)

    balances = np.full(
        (
            batch_size,
            n_policies,
        ),
        STARTING_BALANCE,
        dtype=np.float64,
    )

    peak_balance = balances.copy()

    max_dd = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.float64,
    )

    winning_days = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.int32,
    )

    day_pnl = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.float64,
    )

    day_won = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    current_day = np.full(
        batch_size,
        -1,
        dtype=np.int32,
    )

    active = np.ones(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    passed = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    failed = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=bool,
    )

    trades_completed = np.zeros(
        (
            batch_size,
            n_policies,
        ),
        dtype=np.int32,
    )

    initial_risk = np.asarray(
        [
            float(scenario["initial"])
            for (
                _,
                scenario,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    after_threshold_risk = np.asarray(
        [
            float(scenario["after_threshold"])
            for (
                _,
                scenario,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    thresholds = np.asarray(
        [
            (np.nan if scenario["threshold"] is None else float(scenario["threshold"]))
            for (
                _,
                scenario,
            ) in scenarios
        ],
        dtype=np.float64,
    )

    for t in range(max_trades):
        returns_t = path_returns[:, t]

        day_t = day_ids[:, t]

        if t == 0:
            new_day = np.ones(
                batch_size,
                dtype=bool,
            )

        else:
            new_day = day_t != current_day

        if np.any(new_day):
            day_pnl[
                new_day,
                :,
            ] = 0.0

            day_won[
                new_day,
                :,
            ] = False

        current_day = day_t

        active_before = active.copy()

        if not np.any(active_before):
            break

        profit_from_start = balances - STARTING_BALANCE

        use_after_threshold = np.isfinite(thresholds)[None, :] & (
            profit_from_start >= thresholds[None, :]
        )

        risk_pct = np.where(
            use_after_threshold,
            after_threshold_risk[
                None,
                :,
            ],
            initial_risk[
                None,
                :,
            ],
        )

        pnl = returns_t[:, None] * STARTING_BALANCE * risk_pct

        pnl = np.where(
            active_before,
            pnl,
            0.0,
        )

        balances += pnl

        trades_completed[active_before] += 1

        day_pnl += pnl

        newly_winning = active_before & ~day_won & (day_pnl >= MIN_WINNING_DAY_PROFIT)

        winning_days[newly_winning] += 1

        day_won[newly_winning] = True
        # ---------------------------------------------------------------------*
        # MLL*
        # ---------------------------------------------------------------------*

        hit_mll = active_before & (balances <= STARTING_BALANCE - MAX_LOSS_LIMIT)

        failed[hit_mll] = True

        active[hit_mll] = False
        # ---------------------------------------------------------------------*
        # PROFIT TARGET*
        # ---------------------------------------------------------------------*

        hit_target = active & (balances >= STARTING_BALANCE + PROFIT_TARGET)

        passed[hit_target] = True

        active[hit_target] = False
        # ---------------------------------------------------------------------*
        # DD*
        # ---------------------------------------------------------------------*

        peak_balance = np.maximum(
            peak_balance,
            balances,
        )

        max_dd = np.minimum(
            max_dd,
            balances - peak_balance,
        )
        # ---------------------------------------------------------------------*
        # MAX TRADES*
        # ---------------------------------------------------------------------*

        reached_max = trades_completed >= max_trades

        active[reached_max] = False

    return {
        name: {
            "passed": passed[
                :,
                i,
            ].copy(),
            "failed": failed[
                :,
                i,
            ].copy(),
            "trades": trades_completed[
                :,
                i,
            ].copy(),
            "profit": (
                balances[
                    :,
                    i,
                ]
                - STARTING_BALANCE
            ).copy(),
            "final_balance": balances[
                :,
                i,
            ].copy(),
            "max_drawdown": max_dd[
                :,
                i,
            ].copy(),
            "winning_days": winning_days[
                :,
                i,
            ].copy(),
        }
        for i, (
            name,
            _,
        ) in enumerate(scenarios)
    }


# =============================================================================*
# PARITY AUDIT*
# =============================================================================*


def run_parity_audit(
    sequence: HistoricalSequence,
) -> None:
    """

    *       Validate vectorized XFA and Combine engines against the scalar reference.**

    *       This is deliberately small and deterministic.**

    *       If this fails, the production 50k simulation MUST NOT be trusted.**

    *"""

    banner(f"PARITY AUDIT — {sequence.strategy}")

    strategy_seed_offset = 303

    rng = np.random.default_rng(RANDOM_SEED + strategy_seed_offset)

    n_paths = PARITY_PATHS

    start_days = rng.integers(
        0,
        sequence.n_days,
        size=n_paths,
    )

    path_returns, day_ids = sequence.build_batch(
        start_days,
        PARITY_MAX_TRADES,
    )
    # -------------------------------------------------------------------------*
    # COMBINE*
    # -------------------------------------------------------------------------*

    combine_scenarios = [
        (
            name,
            scenario,
        )
        for name, scenario in RISK_SCENARIOS.items()
    ]

    vector_combine = run_combine_batch(
        path_returns,
        day_ids,
        combine_scenarios,
        PARITY_MAX_TRADES,
    )

    for path_index in range(n_paths):
        for name, scenario in combine_scenarios:
            scalar = simulate_combine_scalar(
                path_returns[path_index],
                day_ids[path_index],
                scenario,
                PARITY_MAX_TRADES,
            )

            vector = vector_combine[name]

            checks = {
                "passed": bool(vector["passed"][path_index]) == bool(scalar["passed"]),
                "failed": bool(vector["failed"][path_index]) == bool(scalar["failed"]),
                "trades": int(vector["trades"][path_index]) == int(scalar["trades"]),
                "profit": np.isclose(
                    vector["profit"][path_index],
                    scalar["profit"],
                ),
                "max_dd": np.isclose(
                    vector["max_drawdown"][path_index],
                    scalar["max_drawdown"],
                ),
                "winning_days": int(vector["winning_days"][path_index])
                == int(scalar["winning_days"]),
            }

            if not all(checks.values()):
                raise RuntimeError(
                    "\nCOMBINE PARITY FAILURE\n"
                    f"strategy={sequence.strategy}\n"
                    f"path={path_index}\n"
                    f"scenario={name}\n"
                    f"checks={checks}\n"
                    f"scalar={scalar}\n"
                    f"vector={{"
                    f"'passed': "
                    f"{vector['passed'][path_index]}, "
                    f"'failed': "
                    f"{vector['failed'][path_index]}, "
                    f"'trades': "
                    f"{vector['trades'][path_index]}, "
                    f"'profit': "
                    f"{vector['profit'][path_index]}, "
                    f"'max_dd': "
                    f"{vector['max_drawdown'][path_index]}, "
                    f"'winning_days': "
                    f"{vector['winning_days'][path_index]}"
                    f"}}"
                )

    print("Combine scalar/vector parity: PASS")
    # -------------------------------------------------------------------------*
    # XFA*
    # -------------------------------------------------------------------------*

    xfa_scenarios = []

    for name, scenario in RISK_SCENARIOS.items():
        for interval in PAYOUT_INTERVALS:
            for amount in PAYOUT_AMOUNTS:
                xfa_scenarios.append(
                    (
                        name,
                        scenario,
                        interval,
                        float(amount),
                    )
                )

    vector_xfa = run_xfa_batch(
        path_returns,
        day_ids,
        xfa_scenarios,
        PARITY_MAX_TRADES,
    )

    for path_index in range(n_paths):
        for (
            name,
            scenario,
            interval,
            amount,
        ) in xfa_scenarios:
            scalar = simulate_xfa_scalar(
                path_returns[path_index],
                day_ids[path_index],
                scenario,
                interval,
                amount,
                PARITY_MAX_TRADES,
            )

            vector = vector_xfa[
                (
                    name,
                    interval,
                    amount,
                )
            ]

            checks = {
                "failed": bool(vector["failed"][path_index]) == bool(scalar["failed"]),
                "trades": int(vector["trades"][path_index]) == int(scalar["trades"]),
                "final_balance": np.isclose(
                    vector["final_balance"][path_index],
                    scalar["final_balance"],
                ),
                "withdrawn": np.isclose(
                    vector["total_withdrawn"][path_index],
                    scalar["total_withdrawn"],
                ),
                "payouts": int(vector["payouts"][path_index]) == int(scalar["payouts"]),
                "max_dd": np.isclose(
                    vector["max_drawdown"][path_index],
                    scalar["max_drawdown"],
                ),
                "winning_days": int(vector["winning_days"][path_index])
                == int(scalar["winning_days"]),
            }

            if not all(checks.values()):
                raise RuntimeError(
                    "\nXFA PARITY FAILURE\n"
                    f"strategy={sequence.strategy}\n"
                    f"path={path_index}\n"
                    f"risk={name}\n"
                    f"interval={interval}\n"
                    f"amount={amount}\n"
                    f"checks={checks}\n"
                    f"scalar={scalar}\n"
                    f"vector={{"
                    f"'failed': "
                    f"{vector['failed'][path_index]}, "
                    f"'trades': "
                    f"{vector['trades'][path_index]}, "
                    f"'final_balance': "
                    f"{vector['final_balance'][path_index]}, "
                    f"'withdrawn': "
                    f"{vector['total_withdrawn'][path_index]}, "
                    f"'payouts': "
                    f"{vector['payouts'][path_index]}, "
                    f"'max_dd': "
                    f"{vector['max_drawdown'][path_index]}, "
                    f"'winning_days': "
                    f"{vector['winning_days'][path_index]}"
                    f"}}"
                )

    print("XFA scalar/vector parity: PASS")

    print("PARITY AUDIT: PASS")


# =============================================================================*
# BUILD START-DAY SAMPLES*
# =============================================================================*


def build_start_days(
    sequence: HistoricalSequence,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """

    *       Select REAL historical starting days.**

    *       We intentionally randomize ONLY the starting point.**

    *       Individual trades are never independently shuffled.**

    *"""

    return rng.integers(
        0,
        sequence.n_days,
        size=n_simulations,
        dtype=np.int32,
    )


# =============================================================================*
# COMBINE SIMULATION*
# =============================================================================*


def simulate_combine(
    sequence: HistoricalSequence,
    start_days: np.ndarray,
) -> pd.DataFrame:

    records: dict[
        str,
        list[np.ndarray],
    ] = {name: [] for name in RISK_SCENARIOS}

    total = len(start_days)

    combine_scenarios = [
        (
            name,
            scenario,
        )
        for name, scenario in RISK_SCENARIOS.items()
    ]

    for batch_start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        batch_days = start_days[batch_start : batch_start + BATCH_SIZE]

        path_returns, day_ids = sequence.build_batch(
            batch_days,
            MAX_COMBINE_TRADES,
        )

        batch_result = run_combine_batch(
            path_returns,
            day_ids,
            combine_scenarios,
            MAX_COMBINE_TRADES,
        )

        for name in RISK_SCENARIOS:
            records[name].append(batch_result[name])

        completed = min(
            batch_start + BATCH_SIZE,
            total,
        )

        if completed == total or completed % 10_000 == 0:
            print(f"    Combine: {completed:,}/{total:,}")

    rows = []

    for name in RISK_SCENARIOS:
        combined = {
            key: np.concatenate([batch[key] for batch in records[name]])
            for key in records[name][0]
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
                "p95_max_DD": percentile(
                    combined["max_drawdown"],
                    5,
                ),
                "p99_max_DD": percentile(
                    combined["max_drawdown"],
                    1,
                ),
                "median_winning_days": float(np.median(combined["winning_days"])),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================*
# XFA SIMULATION*
# =============================================================================*


def simulate_xfa(
    sequence: HistoricalSequence,
    start_days: np.ndarray,
) -> pd.DataFrame:

    total = len(start_days)

    policy_records: dict[
        tuple[str, int, float],
        list[dict[str, np.ndarray]],
    ] = {}

    policies = []

    for name, scenario in RISK_SCENARIOS.items():
        for interval in PAYOUT_INTERVALS:
            for amount in PAYOUT_AMOUNTS:
                policy = (
                    name,
                    scenario,
                    interval,
                    float(amount),
                )

                policies.append(policy)

                policy_records[
                    (
                        name,
                        interval,
                        float(amount),
                    )
                ] = []

    print(f"    Vectorized policies: {len(policies)}")

    for batch_start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        batch_days = start_days[batch_start : batch_start + BATCH_SIZE]

        path_returns, day_ids = sequence.build_batch(
            batch_days,
            MAX_XFA_TRADES,
        )

        batch_result = run_xfa_batch(
            path_returns,
            day_ids,
            policies,
            MAX_XFA_TRADES,
        )

        for key, result in batch_result.items():
            policy_records[key].append(result)

        completed = min(
            batch_start + BATCH_SIZE,
            total,
        )

        if completed == total or completed % 10_000 == 0:
            print(f"    XFA: {completed:,}/{total:,}")

    rows = []

    for (
        name,
        _scenario,
        interval,
        amount,
    ) in policies:
        key = (
            name,
            interval,
            amount,
        )

        batches = policy_records[key]

        combined = {
            field: np.concatenate([batch[field] for batch in batches])
            for field in batches[0]
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
                "p95_total_withdrawn": percentile(
                    combined["total_withdrawn"],
                    95,
                ),
                "median_final_balance": float(np.median(combined["final_balance"])),
                "median_net_value": float(
                    np.median(combined["final_balance"] + combined["total_withdrawn"])
                ),
                "median_max_DD": float(np.median(combined["max_drawdown"])),
                "p95_DD": percentile(
                    combined["max_drawdown"],
                    5,
                ),
                "median_trades": float(np.median(combined["trades"])),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================*
# REPORT*
# =============================================================================*


def report_combine(
    df: pd.DataFrame,
) -> None:

    banner("COMBINE RESULTS")

    print(df.to_string(index=False))


def report_xfa(
    df: pd.DataFrame,
) -> None:

    banner("XFA RESULTS — POLICIES WITH >= 90% SURVIVAL")

    viable = df[df["survival_rate"] >= 0.90].copy()

    if viable.empty:
        print("No policy achieved 90% survival.")

    else:
        print(
            viable.sort_values(
                [
                    "survival_rate",
                    "median_total_withdrawn",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(20)
            .to_string(index=False)
        )

    banner("HIGHEST-SURVIVAL POLICIES")

    safest = df.sort_values(
        [
            "survival_rate",
            "median_total_withdrawn",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(20)

    print(safest.to_string(index=False))


# =============================================================================*
# MAIN*
# =============================================================================*


def main() -> None:
    banner("FULL SYSTEM FUNDED ACCOUNT SIMULATION — HISTORICAL SEQUENCE REPLAY")

    print("SYSTEM: MRL1 + S2R + MRS2 + ORB")
    print("NO STRATEGY OPTIMIZATION.")
    print("ACCOUNT POLICY ONLY.")
    print("INDIVIDUAL TRADES ARE NOT SHUFFLED.")
    print("REAL HISTORICAL PORTFOLIO TRADE ORDER IS PRESERVED.")
    print("REAL HISTORICAL DAY BOUNDARIES ARE PRESERVED.")
    print(f"Replay paths: {N_SIMULATIONS:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print("Exact COMMON OOS: 2020-06-23 through 2026-06-19")

    banner("LOADING FULL SYSTEM EXACT-OOS TRADES")
    df = load_trades()

    print(f"Exact OOS portfolio trades: {len(df):,}")
    print(f"First trade: {df['entry_timestamp'].iloc[0]}")
    print(f"Last trade:  {df['entry_timestamp'].iloc[-1]}")

    sequence = HistoricalSequence("FULL_SYSTEM", df)
    info = sequence.summary()

    banner("SYSTEM: MRL1 + S2R + MRS2 + ORB")
    print(f"Historical portfolio trades: {info['trades']:,}")
    print(f"Historical trading days: {info['days']:,}")
    print(f"First trade: {info['first_timestamp']}")
    print(f"Last trade: {info['last_timestamp']}")

    print()
    print("Trades by strategy:")
    for strategy in ["MRL1", "S2R", "MRS2", "ORB"]:
        print(f"  {strategy}: {int((df['strategy_name'] == strategy).sum()):,}")

    r = sequence.returns
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())

    print()
    print("Historical portfolio distribution:")
    print(f"Mean R:   {np.mean(r):.6f}")
    print(f"Median R: {np.median(r):.6f}")
    print(f"Win rate: {wins / len(r):.4%}")
    print(f"Loss rate: {losses / len(r):.4%}")

    run_parity_audit(sequence)

    rng = np.random.default_rng(RANDOM_SEED)
    start_days = build_start_days(
        sequence,
        N_SIMULATIONS,
        rng,
    )

    banner("PART A — $50K TRADING COMBINE")
    print(f"Starting balance: ${STARTING_BALANCE:,.0f}")
    print(f"Profit target: +${PROFIT_TARGET:,.0f}")
    print(f"Maximum loss: -${MAX_LOSS_LIMIT:,.0f}")

    combine_df = simulate_combine(sequence, start_days)
    report_combine(combine_df)

    combine_path = RESULTS_DIR / "full_system_funded_combine_results.csv"
    combine_df.to_csv(combine_path, index=False)
    print(f"Saved Combine:\\n{combine_path}")

    banner("PART B — XFA / FUNDED ACCOUNT")
    print("HISTORICAL SEQUENCE REPLAY.")
    print(f"Minimum winning days: {MIN_WINNING_DAYS}")
    print(f"Minimum winning day: ${MIN_WINNING_DAY_PROFIT:,.2f}")
    print(f"Fixed loss floor: ${XFA_FIXED_LOSS_FLOOR:,.2f}")

    xfa_df = simulate_xfa(sequence, start_days)
    report_xfa(xfa_df)

    xfa_path = RESULTS_DIR / "full_system_funded_xfa_results.csv"
    xfa_df.to_csv(xfa_path, index=False)
    print(f"Saved XFA:\\n{xfa_path}")

    banner("FULL SYSTEM FUNDED ACCOUNT SIMULATION COMPLETE")
    print("System: MRL1 + S2R + MRS2 + ORB.")
    print("Exact common OOS only.")
    print("Post-OOS excluded.")
    print("Historical chronological portfolio sequence preserved.")
    print("Real historical day boundaries preserved.")
    print("Individual trades were NOT shuffled.")
    print("Vectorized engines passed scalar parity audit.")
    print(f"Combine CSV: {combine_path}")
    print(f"XFA CSV:     {xfa_path}")


if __name__ == "__main__":
    main()
