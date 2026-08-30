from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.strategies.s2r import S2RStrategy, fit_s2_model
from src.strategies.s2r.recovery import RecoveryState


ROOT = Path(__file__).resolve().parent

STOP_POINTS = 25.0
RR = 1.75
HORIZON = 20

TRAIN_FRACTION = 0.70

REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "hmm_state",
    "past_return_30",
    "directional_pressure_30",
    "close_location_30",
    "normalized_momentum_30",
    "realized_vol_30",
}


def find_data_file() -> Path:
    """
    Find the first CSV containing all columns required by the modular S2R
    strategy.
    """

    candidates = []

    for path in ROOT.rglob("*.csv"):
        if "results" in path.parts:
            continue

        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue

        if REQUIRED_COLUMNS.issubset(columns):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not find a CSV containing all required S2R columns:\n"
            + "\n".join(sorted(REQUIRED_COLUMNS))
        )

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)

    return candidates[0]


def load_data(path: Path) -> pd.DataFrame:
    print("=" * 90)
    print("S2R MODULAR BACKTEST")
    print("=" * 90)
    print()
    print("DATA:")
    print(path)
    print()

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(sorted(missing)))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])

    else:
        df["timestamp"] = pd.RangeIndex(len(df))

    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "hmm_state",
        "past_return_30",
        "directional_pressure_30",
        "close_location_30",
        "normalized_momentum_30",
        "realized_vol_30",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)

    return df


def build_model(train: pd.DataFrame):
    rows = train[
        [
            "hmm_state",
            "past_return_30",
            "directional_pressure_30",
            "close_location_30",
            "normalized_momentum_30",
            "realized_vol_30",
        ]
    ].to_dict("records")

    return fit_s2_model(rows)


def resolve_trade(
    df: pd.DataFrame,
    entry_index: int,
    strategy: S2RStrategy,
):
    """
    Run the modular S2R recovery state machine bar-by-bar.

    Entry is at the close of entry_index.

    For a short:
        MAE_R   = (high - entry) / STOP
        close_R = (entry - close) / STOP

    Recovery:
        MAE >= 0.70R
        then close >= +0.20R
        deadline = 6 bars after MAE

    If recovery never happens, exit at the recovery deadline.

    If MAE never occurs, the original S2R TP/SL/horizon logic is used.
    """

    entry_price = float(df.iloc[entry_index]["close"])

    strategy.start_trade(
        entry_price=entry_price,
        entry_bar=entry_index,
    )

    target_price = entry_price - STOP_POINTS * RR
    stop_price = entry_price + STOP_POINTS

    last_index = min(
        entry_index + HORIZON,
        len(df) - 1,
    )

    recovery_started = False

    for i in range(entry_index + 1, last_index + 1):
        row = df.iloc[i]

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        result = strategy.update_trade_from_market(
            bar_index=i,
            high=high,
            close=close,
        )

        if result.state is RecoveryState.RECOVERED:
            exit_price = entry_price - STOP_POINTS * 0.20

            strategy.finish_trade()

            return {
                "entry_index": entry_index,
                "exit_index": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_points": entry_price - exit_price,
                "R": 0.20,
                "reason": "recovery",
            }

        if result.state is RecoveryState.FAILED_TO_RECOVER:
            exit_bar = result.exit_bar

            exit_price = float(df.iloc[exit_bar]["close"])

            raw_points = entry_price - exit_price

            strategy.finish_trade()

            return {
                "entry_index": entry_index,
                "exit_index": exit_bar,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_points": raw_points,
                "R": raw_points / STOP_POINTS,
                "reason": "recovery_failed",
            }

        if result.state is RecoveryState.ADVERSE:
            recovery_started = True

        # Before MAE/recovery activates, retain the original
        # target/stop mechanics.
        if not recovery_started:
            target_hit = low <= target_price
            stop_hit = high >= stop_price

            if target_hit and stop_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "raw_points": -STOP_POINTS,
                    "R": -1.0,
                    "reason": "both_hit_conservative_stop",
                }

            if target_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": target_price,
                    "raw_points": STOP_POINTS * RR,
                    "R": RR,
                    "reason": "target",
                }

            if stop_hit:
                strategy.finish_trade()

                return {
                    "entry_index": entry_index,
                    "exit_index": i,
                    "entry_price": entry_price,
                    "exit_price": stop_price,
                    "raw_points": -STOP_POINTS,
                    "R": -1.0,
                    "reason": "stop",
                }

    exit_price = float(df.iloc[last_index]["close"])
    raw_points = entry_price - exit_price

    strategy.finish_trade()

    return {
        "entry_index": entry_index,
        "exit_index": last_index,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "raw_points": raw_points,
        "R": raw_points / STOP_POINTS,
        "reason": "timeout",
    }


def main():
    path = find_data_file()

    df = load_data(path)

    split = int(len(df) * TRAIN_FRACTION)

    train = df.iloc[:split].copy()
    oos = df.iloc[split:].copy()

    print(f"Rows:        {len(df):,}")
    print(f"Train rows:  {len(train):,}")
    print(f"OOS rows:    {len(oos):,}")
    print()

    model = build_model(train)

    print("FROZEN MODEL")
    print("------------")
    print("Thresholds:")
    for key, value in model.signal_model.thresholds.items():
        print(f"  {key}: {value}")

    print()
    print("Scales:")
    for key, value in model.signal_model.scales.items():
        print(f"  {key}: {value}")

    print()
    print("Recovery:")
    print("  MAE threshold : 0.70R")
    print("  Recovery      : +0.20R")
    print("  Deadline      : 6 bars")
    print()

    strategy = S2RStrategy(
        fitted_model=model,
    )

    trades = []

    i = 0

    while i < len(oos) - 1:
        row = oos.iloc[i]

        context = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "hmm_state": int(row["hmm_state"]),
            "past_return_30": float(row["past_return_30"]),
            "directional_pressure_30": float(row["directional_pressure_30"]),
            "close_location_30": float(row["close_location_30"]),
            "normalized_momentum_30": float(row["normalized_momentum_30"]),
            "realized_vol_30": float(row["realized_vol_30"]),
        }

        decision = strategy.evaluate(context)

        if decision.action.value == "enter":
            trade = resolve_trade(
                oos,
                i,
                strategy,
            )

            trades.append(trade)

            i = trade["exit_index"] + 1

        else:
            i += 1

    if not trades:
        print("NO TRADES GENERATED.")
        return

    result = pd.DataFrame(trades)

    total_R = result["R"].sum()
    mean_R = result["R"].mean()
    win_rate = (result["R"] > 0).mean()

    gross_profit = result.loc[result["R"] > 0, "R"].sum()
    gross_loss = abs(result.loc[result["R"] < 0, "R"].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = result["R"].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()

    print("=" * 90)
    print("MODULAR S2R RESULT")
    print("=" * 90)
    print()
    print(f"Trades:          {len(result)}")
    print(f"Total R:         {total_R:.6f}")
    print(f"Mean R:          {mean_R:.6f}")
    print(f"Win rate:        {win_rate:.4%}")
    print(f"Profit factor:   {profit_factor:.6f}")
    print(f"Max drawdown:    {max_dd:.6f}R")
    print()

    print("EXIT REASONS")
    print("------------")

    print(result["reason"].value_counts().to_string())

    print()

    output = ROOT / "s2r_modular_backtest_results.csv"

    result.to_csv(
        output,
        index=False,
    )

    print(f"Trades saved to:")
    print(output)
    print()


if __name__ == "__main__":
    main()
