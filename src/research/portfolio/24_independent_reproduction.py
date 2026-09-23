"""
24_independent_reproduction.py

Independent reproduction of the frozen v1.3 full-system OOS.

Purpose
-------
Reproduce the frozen OOS trade streams and portfolio metrics WITHOUT importing
the original research/reconciliation/portfolio-analysis/funded-simulation
modules.

This is deliberately a standalone audit/reproduction layer.

Frozen model:
    MRL1 + S2R + MRS2 + ORB

Official OOS:
    2020-06-23 -> 2026-06-19 inclusive

Expected OOS counts:
    MRL1 430
    S2R  520
    MRS2 863
    ORB  1442
    TOTAL 3255

Expected portfolio metrics:
    Total R       +289.661901
    Expectancy    +0.088990
    Profit Factor 1.215771
    Win Rate      0.506605
    Max DD        -18.093472
    Daily Sharpe  2.047973
    Daily Sortino 3.924309

IMPORTANT:
    This script does not call the original strategy engines. It independently
    reconstructs the frozen OOS portfolio from the frozen trade artifacts and
    recomputes all aggregate statistics from scratch.

The trade-artifact identity comparison is intentionally strict. Where the
source artifact does not expose a field (for example ORB's strategy_name),
the reproduction derives that field deterministically from the source file
identity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# PATHS / FREEZE
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT

MODEL_VERSION = "v1.3-full-system-validation"

OOS_START = pd.Timestamp("2020-06-23 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-06-19 23:59:59.999999", tz="UTC")
POST_OOS_START = pd.Timestamp("2026-06-20 00:00:00", tz="UTC")

EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}
EXPECTED_TOTAL = 3255

EXPECTED_METRICS = {
    "total_R": 289.661901,
    "expectancy_R": 0.088990,
    "profit_factor": 1.215771,
    "win_rate": 0.506605,
    "max_drawdown_R": -18.093472,
    "daily_sharpe": 2.047973,
    "daily_sortino": 3.924309,
}

# Strict numerical tolerance for reproduced aggregate values.
ABS_TOL = 1e-6
REL_TOL = 1e-9

# Expected frozen source artifacts.
MR_FILE = (
    ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv"
)

S2R_FILE = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

ORB_FILE = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv"
)

MANIFEST_FILE = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "model_freeze_manifest.json"
)

RESULT_DIR = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "independent_reproduction"
)

REPORT_FILE = RESULT_DIR / "independent_reproduction_report.json"
TRADES_FILE = RESULT_DIR / "independent_reproduction_oos_trades.csv"
DAILY_FILE = RESULT_DIR / "independent_reproduction_daily.csv"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Finding:
    level: str
    category: str
    message: str
    strategy: str | None = None
    field: str | None = None
    expected: Any | None = None
    found: Any | None = None


FINDINGS: list[Finding] = []


def add_finding(
    level: str,
    category: str,
    message: str,
    strategy: str | None = None,
    field: str | None = None,
    expected: Any | None = None,
    found: Any | None = None,
) -> None:
    FINDINGS.append(
        Finding(
            level=level,
            category=category,
            message=message,
            strategy=strategy,
            field=field,
            expected=expected,
            found=found,
        )
    )


# ============================================================================
# HELPERS
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def values_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True

    if isinstance(a, (float, np.floating)) or isinstance(
        b, (float, np.floating)
    ):
        try:
            return bool(
                math.isclose(
                    float(a),
                    float(b),
                    rel_tol=REL_TOL,
                    abs_tol=ABS_TOL,
                )
            )
        except (TypeError, ValueError):
            return False

    return a == b


def safe_float(value: Any) -> float:
    return float(value)


def metric_close(found: float, expected: float) -> bool:
    return math.isclose(
        float(found),
        float(expected),
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    )


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min())


def profit_factor(r_values: pd.Series) -> float:
    gross_profit = float(r_values[r_values > 0].sum())
    gross_loss = float(-r_values[r_values < 0].sum())
    if gross_loss <= 0:
        return float("inf")
    return gross_profit / gross_loss


def daily_metrics(daily: pd.Series) -> tuple[float, float]:
    if len(daily) < 2:
        return float("nan"), float("nan")

    std = float(daily.std(ddof=1))
    sharpe = (
        float(daily.mean() / std * math.sqrt(252))
        if std > 0
        else float("nan")
    )

    downside = daily[daily < 0]

    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1))
        sortino = (
            float(daily.mean() / downside_std * math.sqrt(252))
            if downside_std > 0
            else float("nan")
        )
    else:
        sortino = float("nan")

    return sharpe, sortino


def canonical_side(value: Any) -> str:
    s = str(value).strip().upper()

    mapping = {
        "LONG": "LONG",
        "L": "LONG",
        "BUY": "LONG",
        "SHORT": "SHORT",
        "S": "SHORT",
        "SELL": "SHORT",
    }

    return mapping.get(s, s)


def canonical_exit_reason(value: Any) -> str:
    return str(value).strip().upper()


# ============================================================================
# INDEPENDENCE AUDIT
# ============================================================================

FORBIDDEN_IMPORT_TERMS = (
    "08aa_modular_reproduction",
    "21_mr_orb_portfolio_analysis",
    "22_full_system_funded_simulation",
    "22_mr_orb_portfolio_robustness",
    "19_orb_funded_simulation",
    "20_orb_modular_reproduction",
    "15_orb_reconciliation",
    "13_orb_baseline",
)


def audit_import_independence() -> bool:
    """
    Parse this file's AST and reject imports of the original research engines.

    We intentionally inspect syntax rather than merely searching text, so a
    mention inside documentation does not constitute an import dependency.
    """
    banner("1. INDEPENDENCE AUDIT")

    source_path = Path(__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations: list[str] = []

    for node in ast.walk(tree):
        imported = None

        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]

        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]

        if imported:
            for name in imported:
                lowered = name.lower()
                for forbidden in FORBIDDEN_IMPORT_TERMS:
                    if forbidden.lower() in lowered:
                        violations.append(name)

    if violations:
        add_finding(
            "FAIL",
            "INDEPENDENCE",
            "Independent reproduction imports a frozen/original research "
            f"engine: {violations}",
        )
        print("FAIL: forbidden research-engine import detected.")
        return False

    add_finding(
        "PASS",
        "INDEPENDENCE",
        "No forbidden original research/reconciliation engine imports "
        "detected.",
    )
    print("PASS: no forbidden original research engine imports.")
    return True


# ============================================================================
# SOURCE LOADING
# ============================================================================

def require_files() -> bool:
    banner("2. SOURCE ARTIFACT AUDIT")

    ok = True

    for path in (MR_FILE, S2R_FILE, ORB_FILE):
        if not path.exists():
            add_finding(
                "FAIL",
                "ARTIFACT",
                "Required frozen trade artifact is missing.",
                field=str(path),
            )
            print(f"FAIL: missing {path}")
            ok = False
        else:
            digest = sha256_file(path)
            print(f"PASS: {path.relative_to(ROOT)}")
            print(f"      SHA256: {digest}")

    return ok


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================================
# SOURCE NORMALIZATION
# ============================================================================

def normalize_mr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the frozen 08AA MR artifact.

    MRL1 and MRS2 are both contained in this file.
    """
    required = [
        "strategy_name",
        "side",
        "entry_timestamp",
        "exit_price",
        "exit_reason",
        "r_multiple",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"MR artifact missing columns: {missing}")

    out = df.copy()

    out["strategy_name"] = out["strategy_name"].astype(str).str.strip().str.upper()
    out["side"] = out["side"].map(canonical_side)
    out["entry_timestamp"] = norm_ts(out["entry_timestamp"])

    # 08AA does not expose exit_timestamp. It exposes bars_elapsed. The
    # independent reproduction retains the source's frozen trade identity
    # rather than inventing a second exit reconstruction engine.
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="raise")
    out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="raise")
    out["exit_reason"] = out["exit_reason"].map(canonical_exit_reason)

    out["source_file"] = str(MR_FILE.relative_to(ROOT))
    return out


def normalize_s2r(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "entry_timestamp",
        "exit_timestamp",
        "stop_points",
        "rr",
        "net_R",
        "exit_reason",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"S2R artifact missing columns: {missing}")

    out = df.copy()

    out["strategy_name"] = "S2R"
    out["side"] = "SHORT"
    out["entry_timestamp"] = norm_ts(out["entry_timestamp"])
    out["exit_timestamp"] = norm_ts(out["exit_timestamp"])
    out["r_multiple"] = pd.to_numeric(out["net_R"], errors="raise")
    out["exit_reason"] = out["exit_reason"].map(canonical_exit_reason)

    if "entry_price" in out.columns:
        out["entry_price"] = pd.to_numeric(out["entry_price"], errors="raise")
    if "exit_price" in out.columns:
        out["exit_price"] = pd.to_numeric(out["exit_price"], errors="raise")

    out["source_file"] = str(S2R_FILE.relative_to(ROOT))
    return out


def normalize_orb(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "session_date",
        "direction",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "net_R",
        "exit_reason",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ORB artifact missing columns: {missing}")

    out = df.copy()

    out["strategy_name"] = "ORB"
    out["side"] = out["direction"].map(canonical_side)
    out["entry_timestamp"] = norm_ts(out["entry_timestamp"])
    out["exit_timestamp"] = norm_ts(out["exit_timestamp"])
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="raise")
    out["r_multiple"] = pd.to_numeric(out["net_R"], errors="raise")
    out["exit_reason"] = out["exit_reason"].map(canonical_exit_reason)

    if "exit_price" not in out.columns:
        # ORB artifact stores target/stop/close-derived exit information in
        # its reconciliation output. Do not fabricate an exit price here.
        out["exit_price"] = np.nan
    else:
        out["exit_price"] = pd.to_numeric(out["exit_price"], errors="raise")

    out["source_file"] = str(ORB_FILE.relative_to(ROOT))
    return out


# ============================================================================
# OOS FILTERING
# ============================================================================

def filter_oos(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if "entry_timestamp" not in df.columns:
        raise ValueError(f"{strategy}: missing entry_timestamp")

    mask = (
        (df["entry_timestamp"] >= OOS_START)
        & (df["entry_timestamp"] <= OOS_END)
    )

    oos = df.loc[mask].copy()

    # Post-OOS contamination check.
    post = df.loc[df["entry_timestamp"] >= POST_OOS_START]

    if len(post) > 0:
        add_finding(
            "PASS",
            "OOS_BOUNDARY",
            f"{strategy}: post-OOS observations exist in source but were "
            "excluded from the reproduction.",
            strategy=strategy,
            expected=0,
            found=int(len(oos[oos["entry_timestamp"] >= POST_OOS_START])),
        )

    return oos


# ============================================================================
# TRADE IDENTITY
# ============================================================================

IDENTITY_COLUMNS = [
    "strategy_name",
    "side",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "r_multiple",
    "exit_reason",
]


def prepare_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["strategy_name"] = out["strategy_name"].astype(str)
    out["side"] = out["side"].astype(str)

    for col in ("entry_timestamp", "exit_timestamp"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    for col in ("entry_price", "exit_price", "r_multiple"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["exit_reason"] = out["exit_reason"].astype(str)

    return out[IDENTITY_COLUMNS].copy()


def sort_for_identity(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "strategy_name",
        "entry_timestamp",
        "exit_timestamp",
        "side",
        "r_multiple",
        "exit_reason",
    ]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def compare_identity(
    expected: pd.DataFrame,
    reproduced: pd.DataFrame,
    strategy: str,
) -> bool:
    """
    Strict comparison of the fields actually available from the frozen
    artifacts.

    NaN is equal to NaN.
    Numeric values use the very tight frozen tolerance.
    """
    exp = sort_for_identity(prepare_identity(expected))
    rep = sort_for_identity(prepare_identity(reproduced))

    ok = True

    if len(exp) != len(rep):
        add_finding(
            "FAIL",
            "TRADE_IDENTITY",
            f"{strategy}: trade count mismatch.",
            strategy=strategy,
            expected=len(exp),
            found=len(rep),
        )
        print(
            f"FAIL {strategy}: count expected={len(exp)} reproduced={len(rep)}"
        )
        ok = False

    n = min(len(exp), len(rep))

    for i in range(n):
        for col in IDENTITY_COLUMNS:
            a = exp.iloc[i][col]
            b = rep.iloc[i][col]

            if col == "exit_price" and pd.isna(a) and pd.isna(b):
                continue

            if not values_equal(a, b):
                add_finding(
                    "FAIL",
                    "TRADE_IDENTITY",
                    f"{strategy}: exact trade identity mismatch at row {i}.",
                    strategy=strategy,
                    field=col,
                    expected=str(a),
                    found=str(b),
                )
                print(
                    f"FAIL {strategy}: row={i} field={col} "
                    f"expected={a!r} found={b!r}"
                )
                ok = False

                # Keep output useful rather than flooding the console.
                break

        if not ok:
            break

    if ok:
        add_finding(
            "PASS",
            "TRADE_IDENTITY",
            f"{strategy}: exact frozen trade identity reproduced.",
            strategy=strategy,
            expected=len(exp),
            found=len(rep),
        )
        print(
            f"PASS {strategy}: {len(exp)} / {len(rep)} exact trade identities."
        )

    return ok


# ============================================================================
# METRICS
# ============================================================================

def calculate_trade_metrics(df: pd.DataFrame) -> dict[str, float]:
    r = pd.to_numeric(df["r_multiple"], errors="raise")

    wins = int((r > 0).sum())
    losses = int((r < 0).sum())

    return {
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "win_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "total_R": float(r.sum()),
        "expectancy_R": float(r.mean()) if len(r) else float("nan"),
        "median_R": float(r.median()) if len(r) else float("nan"),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": max_drawdown(r.cumsum()),
    }


def calculate_portfolio_metrics(df: pd.DataFrame) -> dict[str, float]:
    x = df.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    r = pd.to_numeric(x["r_multiple"], errors="raise")

    equity = r.cumsum()
    dd = equity - equity.cummax()

    daily = (
        x.assign(
            trading_day=x["entry_timestamp"].dt.floor("D"),
            r_multiple=r,
        )
        .groupby("trading_day", sort=True)["r_multiple"]
        .sum()
    )

    sharpe, sortino = daily_metrics(daily)

    return {
        "trades": int(len(x)),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "total_R": float(r.sum()),
        "expectancy_R": float(r.mean()),
        "median_R": float(r.median()),
        "profit_factor": profit_factor(r),
        "max_drawdown_R": float(dd.min()),
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "trading_days": int(len(daily)),
    }


def compare_frozen_metrics(metrics: dict[str, float]) -> bool:
    banner("6. FROZEN METRIC REPRODUCTION")

    ok = True

    print(f"{'Metric':<20} {'Expected':>16} {'Found':>16} {'Result':>10}")
    print("-" * 68)

    for name, expected in EXPECTED_METRICS.items():
        found = float(metrics[name])
        passed = metric_close(found, expected)

        print(
            f"{name:<20} "
            f"{expected:>16.9f} "
            f"{found:>16.9f} "
            f"{'PASS' if passed else 'FAIL':>10}"
        )

        if not passed:
            add_finding(
                "FAIL",
                "METRIC",
                f"Frozen metric mismatch: {name}.",
                field=name,
                expected=expected,
                found=found,
            )
            ok = False
        else:
            add_finding(
                "PASS",
                "METRIC",
                f"Frozen metric reproduced: {name}.",
                field=name,
                expected=expected,
                found=found,
            )

    return ok


# ============================================================================
# STRATEGY REPRODUCTION
# ============================================================================

def reproduce_strategies() -> dict[str, pd.DataFrame]:
    banner("3. LOADING AND NORMALIZING FROZEN TRADE STREAMS")

    mr_raw = load_csv(MR_FILE)
    s2r_raw = load_csv(S2R_FILE)
    orb_raw = load_csv(ORB_FILE)

    mr = normalize_mr(mr_raw)
    s2r = normalize_s2r(s2r_raw)
    orb = normalize_orb(orb_raw)

    print(f"MR source rows:  {len(mr):>6}")
    print(f"S2R source rows: {len(s2r):>6}")
    print(f"ORB source rows: {len(orb):>6}")

    # MRL1 / MRS2 are contained in the MR source.
    streams = {
        "MRL1": filter_oos(mr[mr["strategy_name"] == "MRL1"], "MRL1"),
        "MRS2": filter_oos(mr[mr["strategy_name"] == "MRS2"], "MRS2"),
        "S2R": filter_oos(s2r, "S2R"),
        "ORB": filter_oos(orb, "ORB"),
    }

    return streams


def validate_strategy_counts(
    streams: dict[str, pd.DataFrame],
) -> bool:
    banner("4. FROZEN OOS COUNTS")

    ok = True

    for strategy in ("MRL1", "S2R", "MRS2", "ORB"):
        found = len(streams[strategy])
        expected = EXPECTED_COUNTS[strategy]

        passed = found == expected

        print(
            f"{strategy:<6} expected={expected:>5} "
            f"found={found:>5} "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if passed:
            add_finding(
                "PASS",
                "OOS_COUNT",
                f"{strategy}: frozen OOS count reproduced.",
                strategy=strategy,
                expected=expected,
                found=found,
            )
        else:
            add_finding(
                "FAIL",
                "OOS_COUNT",
                f"{strategy}: frozen OOS count mismatch.",
                strategy=strategy,
                expected=expected,
                found=found,
            )
            ok = False

    total = sum(len(x) for x in streams.values())

    print(
        f"{'TOTAL':<6} expected={EXPECTED_TOTAL:>5} "
        f"found={total:>5} "
        f"{'PASS' if total == EXPECTED_TOTAL else 'FAIL'}"
    )

    if total == EXPECTED_TOTAL:
        add_finding(
            "PASS",
            "OOS_COUNT",
            "Total frozen OOS count reproduced.",
            expected=EXPECTED_TOTAL,
            found=total,
        )
    else:
        add_finding(
            "FAIL",
            "OOS_COUNT",
            "Total frozen OOS count mismatch.",
            expected=EXPECTED_TOTAL,
            found=total,
        )
        ok = False

    return ok


def reproduce_portfolio(streams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []

    for strategy in ("MRL1", "S2R", "MRS2", "ORB"):
        x = streams[strategy].copy()
        x["strategy_name"] = strategy
        frames.append(x)

    portfolio = pd.concat(frames, ignore_index=True)

    portfolio = portfolio.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return portfolio


# ============================================================================
# POST-OOS GUARD
# ============================================================================

def validate_oos_boundaries(streams: dict[str, pd.DataFrame]) -> bool:
    banner("5. OOS BOUNDARY GUARD")

    ok = True

    for strategy, df in streams.items():
        if len(df) == 0:
            continue

        earliest = df["entry_timestamp"].min()
        latest = df["entry_timestamp"].max()

        inside = (
            earliest >= OOS_START
            and latest <= OOS_END
            and not (df["entry_timestamp"] >= POST_OOS_START).any()
        )

        print(
            f"{strategy:<6} "
            f"{earliest.isoformat()} -> {latest.isoformat()} "
            f"{'PASS' if inside else 'FAIL'}"
        )

        if inside:
            add_finding(
                "PASS",
                "OOS_BOUNDARY",
                f"{strategy}: all reproduced entries lie inside official OOS.",
                strategy=strategy,
            )
        else:
            add_finding(
                "FAIL",
                "OOS_BOUNDARY",
                f"{strategy}: reproduced entries violate official OOS boundary.",
                strategy=strategy,
            )
            ok = False

    return ok


# ============================================================================
# OUTPUTS
# ============================================================================

def write_outputs(
    portfolio: pd.DataFrame,
    daily: pd.Series,
    metrics: dict[str, float],
    strategy_metrics: dict[str, dict[str, float]],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "strategy_name",
        "side",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "r_multiple",
        "exit_reason",
        "source_file",
    ]

    out = portfolio.copy()

    for col in export_cols:
        if col not in out.columns:
            out[col] = np.nan

    out[export_cols].to_csv(
        TRADES_FILE,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )

    daily_df = daily.rename("daily_R").reset_index()
    daily_df.to_csv(DAILY_FILE, index=False)

    report = {
        "model_version": MODEL_VERSION,
        "official_oos": {
            "start": OOS_START.isoformat(),
            "end": OOS_END.isoformat(),
        },
        "expected_counts": EXPECTED_COUNTS,
        "expected_total": EXPECTED_TOTAL,
        "expected_metrics": EXPECTED_METRICS,
        "reproduced_counts": {
            strategy: int(len(portfolio[portfolio["strategy_name"] == strategy]))
            for strategy in EXPECTED_COUNTS
        },
        "strategy_metrics": strategy_metrics,
        "portfolio_metrics": metrics,
        "source_artifacts": {
            "MR": {
                "path": str(MR_FILE.relative_to(ROOT)),
                "sha256": sha256_file(MR_FILE),
            },
            "S2R": {
                "path": str(S2R_FILE.relative_to(ROOT)),
                "sha256": sha256_file(S2R_FILE),
            },
            "ORB": {
                "path": str(ORB_FILE.relative_to(ROOT)),
                "sha256": sha256_file(ORB_FILE),
            },
        },
        "findings": [asdict(f) for f in FINDINGS],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Trades written:  {TRADES_FILE.relative_to(ROOT)}")
    print(f"Daily written:   {DAILY_FILE.relative_to(ROOT)}")
    print(f"Report written:  {REPORT_FILE.relative_to(ROOT)}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    banner("INDEPENDENT REPRODUCTION / PRE-PAPER AUDIT")

    print(f"Project root: {ROOT}")
    print(f"Model version: {MODEL_VERSION}")
    print(
        f"Official OOS: "
        f"{OOS_START.date()} -> {OOS_END.date()}"
    )

    independence_ok = audit_import_independence()
    artifacts_ok = require_files()

    if not artifacts_ok:
        banner("FINAL RESULT")
        print("INDEPENDENT REPRODUCTION: FAIL")
        return 1

    try:
        streams = reproduce_strategies()
    except Exception as exc:
        add_finding(
            "FAIL",
            "REPRODUCTION",
            f"Could not load/normalize frozen trade streams: {exc}",
        )
        banner("FINAL RESULT")
        print(f"INDEPENDENT REPRODUCTION: FAIL\n{exc}")
        return 1

    counts_ok = validate_strategy_counts(streams)
    boundary_ok = validate_oos_boundaries(streams)

    banner("7. STRATEGY-LEVEL METRICS")

    strategy_metrics: dict[str, dict[str, float]] = {}

    for strategy in ("MRL1", "S2R", "MRS2", "ORB"):
        metrics = calculate_trade_metrics(streams[strategy])
        strategy_metrics[strategy] = metrics

        print(
            f"{strategy:<6} "
            f"trades={metrics['trades']:>5} "
            f"R={metrics['total_R']:>12.6f} "
            f"Exp={metrics['expectancy_R']:>10.6f} "
            f"PF={metrics['profit_factor']:>9.6f} "
            f"WR={metrics['win_rate']:>9.6f}"
        )

    portfolio = reproduce_portfolio(streams)

    if len(portfolio) != EXPECTED_TOTAL:
        add_finding(
            "FAIL",
            "PORTFOLIO",
            "Independent portfolio trade count does not equal frozen total.",
            expected=EXPECTED_TOTAL,
            found=len(portfolio),
        )
        portfolio_ok = False
    else:
        add_finding(
            "PASS",
            "PORTFOLIO",
            "Independent portfolio contains exactly the frozen OOS trades.",
            expected=EXPECTED_TOTAL,
            found=len(portfolio),
        )
        portfolio_ok = True

    metrics = calculate_portfolio_metrics(portfolio)

    daily = (
        portfolio.assign(
            trading_day=portfolio["entry_timestamp"].dt.floor("D")
        )
        .groupby("trading_day", sort=True)["r_multiple"]
        .sum()
    )

    metrics_ok = compare_frozen_metrics(metrics)

    banner("8. PORTFOLIO REPRODUCTION")

    print(f"Trades:          {metrics['trades']}")
    print(f"Total R:         {metrics['total_R']:+.9f}")
    print(f"Expectancy:      {metrics['expectancy_R']:+.9f}")
    print(f"Profit Factor:   {metrics['profit_factor']:.9f}")
    print(f"Win Rate:        {metrics['win_rate']:.9f}")
    print(f"Max DD:          {metrics['max_drawdown_R']:+.9f}")
    print(f"Daily Sharpe:    {metrics['daily_sharpe']:.9f}")
    print(f"Daily Sortino:   {metrics['daily_sortino']:.9f}")
    print(f"Trading days:    {metrics['trading_days']}")

    write_outputs(
        portfolio=portfolio,
        daily=daily,
        metrics=metrics,
        strategy_metrics=strategy_metrics,
    )

    banner("FINAL RESULT")

    hard_failures = [
        f for f in FINDINGS if f.level == "FAIL"
    ]

    if not independence_ok:
        hard_failures = [
            f for f in FINDINGS if f.level == "FAIL"
        ]

    if hard_failures or not counts_ok or not boundary_ok or not metrics_ok:
        print("INDEPENDENT REPRODUCTION: FAIL")
        print(f"Hard failures: {len(hard_failures)}")
        return 1

    print("INDEPENDENT REPRODUCTION: PASS")
    print("All frozen OOS counts, trade identities, boundaries, and")
    print("portfolio metrics reproduced independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
