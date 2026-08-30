from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2_benchmark_trades_enriched.csv"
)

MODULAR_PATH = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_databento_trades.csv"
)


# ============================================================
# HELPERS
# ============================================================


def load_trades(path: Path, name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"{name} file not found:\n{path}")

    df = pd.read_csv(path)

    if "entry_timestamp" not in df.columns:
        raise KeyError(f"{name} does not contain entry_timestamp.")

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["entry_timestamp"].isna().any():
        raise ValueError(f"{name} contains invalid entry_timestamp values.")

    return df


def normalize_session_id(df: pd.DataFrame) -> pd.Series:

    if "session_id" not in df.columns:
        return pd.Series(
            "",
            index=df.index,
            dtype="object",
        )

    return df["session_id"].astype(str)


# ============================================================
# MAIN COMPARISON
# ============================================================


def main() -> None:

    print("=" * 100)
    print("S2R MODULAR vs AUTHORITATIVE S2 BENCHMARK")
    print("=" * 100)

    benchmark = load_trades(
        BENCHMARK_PATH,
        "BENCHMARK",
    )

    modular = load_trades(
        MODULAR_PATH,
        "MODULAR",
    )

    print()
    print("FILES")
    print("-" * 100)
    print(f"Benchmark : {BENCHMARK_PATH}")
    print(f"Modular   : {MODULAR_PATH}")

    print()
    print("TRADE COUNTS")
    print("-" * 100)
    print(f"Benchmark trades : {len(benchmark):,}")
    print(f"Modular trades   : {len(modular):,}")
    print(f"Difference       : {len(modular) - len(benchmark):+,}")

    # --------------------------------------------------------
    # NORMALIZE TRADE IDENTITY
    # --------------------------------------------------------

    benchmark["_session"] = normalize_session_id(benchmark)
    modular["_session"] = normalize_session_id(modular)

    benchmark["_trade_key"] = (
        benchmark["entry_timestamp"].astype(str) + "|" + benchmark["_session"]
    )

    modular["_trade_key"] = (
        modular["entry_timestamp"].astype(str) + "|" + modular["_session"]
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    print()
    print("TRADE IDENTITY")
    print("-" * 100)

    print(f"Benchmark unique keys : {benchmark['_trade_key'].nunique():,}")

    print(f"Modular unique keys   : {modular['_trade_key'].nunique():,}")

    print(f"Benchmark duplicates   : {benchmark['_trade_key'].duplicated().sum():,}")

    print(f"Modular duplicates     : {modular['_trade_key'].duplicated().sum():,}")

    # --------------------------------------------------------
    # SET COMPARISON
    # --------------------------------------------------------

    benchmark_keys = set(benchmark["_trade_key"])

    modular_keys = set(modular["_trade_key"])

    common_keys = benchmark_keys & modular_keys

    benchmark_only = benchmark_keys - modular_keys
    modular_only = modular_keys - benchmark_keys

    print()
    print("ENTRY COMPARISON")
    print("-" * 100)

    print(f"Entries in BOTH      : {len(common_keys):,}")

    print(f"Benchmark ONLY        : {len(benchmark_only):,}")

    print(f"Modular ONLY          : {len(modular_only):,}")

    if len(benchmark_keys):
        match_pct = len(common_keys) / len(benchmark_keys) * 100.0
    else:
        match_pct = 0.0

    print(f"Benchmark entry match : {match_pct:.2f}%")

    # --------------------------------------------------------
    # BUILD DIFFERENCE TABLES
    # --------------------------------------------------------

    benchmark_only_df = (
        benchmark[benchmark["_trade_key"].isin(benchmark_only)]
        .sort_values("entry_timestamp")
        .copy()
    )

    modular_only_df = (
        modular[modular["_trade_key"].isin(modular_only)]
        .sort_values("entry_timestamp")
        .copy()
    )

    # --------------------------------------------------------
    # SAVE DIFFERENCES
    # --------------------------------------------------------

    output_dir = PROJECT_ROOT / "src" / "research" / "results" / "s2_extended"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_only_path = output_dir / "s2r_comparison_benchmark_only.csv"

    modular_only_path = output_dir / "s2r_comparison_modular_only.csv"

    benchmark_only_df.to_csv(
        benchmark_only_path,
        index=False,
    )

    modular_only_df.to_csv(
        modular_only_path,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT DIFFERENCES
    # --------------------------------------------------------

    print()
    print("BENCHMARK-ONLY ENTRIES")
    print("-" * 100)

    if benchmark_only_df.empty:
        print("NONE")

    else:
        columns = [
            c
            for c in [
                "entry_timestamp",
                "exit_timestamp",
                "quality",
                "rr",
                "raw_points",
                "net_R",
                "exit_reason",
                "holding_bars",
                "window",
                "candidate",
            ]
            if c in benchmark_only_df.columns
        ]

        print(benchmark_only_df[columns].head(30).to_string(index=False))

        if len(benchmark_only_df) > 30:
            print(
                f"\n... {len(benchmark_only_df) - 30:,} "
                "additional benchmark-only entries."
            )

    print()
    print("MODULAR-ONLY ENTRIES")
    print("-" * 100)

    if modular_only_df.empty:
        print("NONE")

    else:
        columns = [
            c
            for c in [
                "entry_timestamp",
                "exit_timestamp",
                "quality",
                "hmm_state",
                "vol_percentile",
                "raw_points",
                "net_R",
                "exit_reason",
                "holding_bars",
                "window",
            ]
            if c in modular_only_df.columns
        ]

        print(modular_only_df[columns].head(30).to_string(index=False))

        if len(modular_only_df) > 30:
            print(
                f"\n... {len(modular_only_df) - 30:,} additional modular-only entries."
            )

    # --------------------------------------------------------
    # COMMON TRADE COMPARISON
    # --------------------------------------------------------

    if common_keys:
        b_common = benchmark[benchmark["_trade_key"].isin(common_keys)][
            [
                "_trade_key",
                "entry_timestamp",
                "exit_timestamp",
                "net_R",
                "exit_reason",
                "holding_bars",
            ]
        ].rename(
            columns={
                "exit_timestamp": "benchmark_exit",
                "net_R": "benchmark_R",
                "exit_reason": "benchmark_exit_reason",
                "holding_bars": "benchmark_holding_bars",
            }
        )

        m_common = modular[modular["_trade_key"].isin(common_keys)][
            [
                "_trade_key",
                "exit_timestamp",
                "net_R",
                "exit_reason",
                "holding_bars",
            ]
        ].rename(
            columns={
                "exit_timestamp": "modular_exit",
                "net_R": "modular_R",
                "exit_reason": "modular_exit_reason",
                "holding_bars": "modular_holding_bars",
            }
        )

        common = b_common.merge(
            m_common,
            on="_trade_key",
            how="inner",
        )

        common["R_difference"] = common["modular_R"] - common["benchmark_R"]

        common["exit_match"] = common["benchmark_exit"] == common["modular_exit"]

        common["reason_match"] = (
            common["benchmark_exit_reason"] == common["modular_exit_reason"]
        )

        common_path = output_dir / "s2r_comparison_common_trades.csv"

        common.to_csv(
            common_path,
            index=False,
        )

        print()
        print("COMMON TRADE EXECUTION COMPARISON")
        print("-" * 100)

        print(f"Common entries             : {len(common):,}")

        print(f"Exit timestamp matches     : {common['exit_match'].sum():,}")

        print(f"Exit timestamp differences : {(~common['exit_match']).sum():,}")

        print(f"Exit reason matches        : {common['reason_match'].sum():,}")

        print(f"Exit reason differences    : {(~common['reason_match']).sum():,}")

        print(f"Benchmark common R         : {common['benchmark_R'].sum():.6f}")

        print(f"Modular common R           : {common['modular_R'].sum():.6f}")

        print(f"Difference common R        : {common['R_difference'].sum():+.6f}")

        print()
        print("COMMON TRADES WITH DIFFERENT EXITS")
        print("-" * 100)

        different_exits = common[~common["exit_match"]].sort_values("entry_timestamp")

        if different_exits.empty:
            print("NONE")

        else:
            print(different_exits.head(30).to_string(index=False))

    # --------------------------------------------------------
    # FINAL DIAGNOSIS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    if len(benchmark_only) == 0 and len(modular_only) == 0:
        print()
        print("ENTRY UNIVERSE: EXACT MATCH")
        print(
            "The modular implementation generates exactly "
            "the same entries as the benchmark."
        )

    elif len(benchmark_only) == 0:
        print()
        print("ENTRY UNIVERSE: MODULAR HAS EXTRA TRADES")
        print(
            "The benchmark entries are all present, but "
            "the modular implementation adds additional entries."
        )

    elif len(modular_only) == 0:
        print()
        print("ENTRY UNIVERSE: MODULAR MISSES TRADES")
        print("The modular implementation does not generate all benchmark entries.")

    else:
        print()
        print("ENTRY UNIVERSE: DIFFERENT")
        print(
            "The modular implementation both misses benchmark "
            "entries and generates additional entries."
        )

    print()
    print("OUTPUT FILES")
    print("-" * 100)
    print(benchmark_only_path)
    print(modular_only_path)

    if common_keys:
        print(common_path)

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
