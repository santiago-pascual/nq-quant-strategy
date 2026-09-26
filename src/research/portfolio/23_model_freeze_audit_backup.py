from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT

RESULTS_DIR = ROOT / "src" / "research" / "results" / "portfolio"
MANIFEST_PATH = RESULTS_DIR / "model_freeze_manifest.json"


# =============================================================================
# MODEL FREEZE
# =============================================================================

MODEL_VERSION = "v1.3-full-system-validation"

OOS_START = pd.Timestamp("2020-06-23", tz="UTC")
OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="UTC")
POST_OOS_START = pd.Timestamp("2026-06-20", tz="UTC")

STRATEGIES = ["MRL1", "S2R", "MRS2", "ORB"]

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
    "PF": 1.215771,
    "win_rate": 0.506605,
    "max_DD": -18.093472,
    "daily_Sharpe": 2.047973,
    "daily_Sortino": 3.924309,
}


# =============================================================================
# REQUIRED ARTIFACTS
# =============================================================================

REQUIRED_ARTIFACTS = [
    ROOT / "src/research/mean_reversion/research/08aa_modular_reproduction.py",
    ROOT
    / "src/research/mean_reversion/results/research_08aa_modular_reproduction_trades.csv",
    ROOT
    / "src/research/results/s2_extended/s2r_modular_authoritative_reproduction.csv",
    ROOT / "src/research/results/orb/orb_reconciliation_trades.csv",
    ROOT / "src/research/portfolio/21_mr_orb_portfolio_analysis.py",
    ROOT / "src/research/portfolio/22_mr_orb_portfolio_robustness.py",
    ROOT / "src/research/portfolio/22_full_system_funded_simulation.py",
    ROOT
    / "src/research/results/portfolio/funded/full_system_funded_combine_results.csv",
    ROOT / "src/research/results/portfolio/funded/full_system_funded_xfa_results.csv",
]


# =============================================================================
# EXPECTED FINAL SYSTEM
# =============================================================================

STRATEGY_SOURCE_MAP = {
    "MRL1": ROOT
    / "src/research/mean_reversion/results/research_08aa_modular_reproduction_trades.csv",
    "MRS2": ROOT
    / "src/research/mean_reversion/results/research_08aa_modular_reproduction_trades.csv",
    "S2R": ROOT
    / "src/research/results/s2_extended/s2r_modular_authoritative_reproduction.csv",
    "ORB": ROOT / "src/research/results/orb/orb_reconciliation_trades.csv",
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

findings: list[dict] = []


def add_finding(
    level: str,
    category: str,
    message: str,
    path: str = "",
    line: int | None = None,
):
    findings.append(
        {
            "level": level,
            "category": category,
            "message": message,
            "path": path,
            "line": line,
        }
    )


def banner(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# =============================================================================
# ARTIFACT AUDIT
# =============================================================================


def audit_artifacts():
    banner("1. ARTIFACT AUDIT")

    for artifact in REQUIRED_ARTIFACTS:
        rel = artifact.relative_to(ROOT)

        if artifact.exists():
            add_finding(
                "PASS",
                "ARTIFACT",
                "Required file exists.",
                str(rel),
            )
        else:
            add_finding(
                "FAIL",
                "ARTIFACT",
                "Required file is missing.",
                str(rel),
            )


# =============================================================================
# TRADE LOADING
# =============================================================================


def load_trade_csv(path: Path, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "entry_timestamp" not in df.columns:
        raise ValueError("missing entry_timestamp")

    if "r_multiple" not in df.columns:
        if "net_R" in df.columns:
            df["r_multiple"] = pd.to_numeric(df["net_R"], errors="coerce")
        elif "net_r" in df.columns:
            df["r_multiple"] = pd.to_numeric(df["net_r"], errors="coerce")
        else:
            raise ValueError("missing r_multiple/net_R")

    df["entry_timestamp"] = pd.to_datetime(
        df["entry_timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["entry_timestamp"].isna().any():
        raise ValueError("invalid entry_timestamp values")

    # MRL1/MRS2 share the same canonical MR reproduction file.
    if strategy in {"MRL1", "MRS2"}:
        if "strategy_name" not in df.columns:
            raise ValueError("missing strategy_name")

        df = df[df["strategy_name"].astype(str).str.upper() == strategy].copy()

    # S2R and ORB authoritative files do not need strategy_name
    # because the filename itself identifies the strategy.
    else:
        df = df.copy()
        df["strategy_name"] = strategy

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    )

    df = df.dropna(subset=["r_multiple"]).copy()

    return df


# =============================================================================
# OOS AUDIT
# =============================================================================


def audit_oos():
    banner("2. FINAL OOS / POST-OOS AUDIT")

    loaded: dict[str, pd.DataFrame] = {}

    for strategy, path in STRATEGY_SOURCE_MAP.items():
        try:
            df = load_trade_csv(path, strategy)
            loaded[strategy] = df

        except Exception as exc:
            add_finding(
                "FAIL",
                "OOS",
                f"Could not read trade file: {path}: {exc}",
                str(path),
            )
            continue

    # -------------------------------------------------------------------------
    # Strategy counts
    # -------------------------------------------------------------------------

    for strategy in STRATEGIES:
        if strategy not in loaded:
            continue

        df = loaded[strategy]

        oos = df[
            (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
        ].copy()

        expected = EXPECTED_COUNTS[strategy]
        actual = len(oos)

        if actual == expected:
            add_finding(
                "PASS",
                "OOS",
                f"{strategy}: expected OOS trade count {expected:,}, found {actual:,}.",
                str(STRATEGY_SOURCE_MAP[strategy]),
            )
        else:
            add_finding(
                "FAIL",
                "OOS",
                f"{strategy}: expected {expected:,} OOS trades, found {actual:,}.",
                str(STRATEGY_SOURCE_MAP[strategy]),
            )

        # Post-OOS is explicitly identified but never used for model metrics.
        post_oos = df[df["entry_timestamp"] >= POST_OOS_START].copy()

        if len(post_oos) > 0:
            add_finding(
                "PASS",
                "POST_OOS",
                (
                    f"{strategy}: {len(post_oos):,} post-OOS trades identified "
                    "and excluded from official model metrics."
                ),
                str(STRATEGY_SOURCE_MAP[strategy]),
            )

    # -------------------------------------------------------------------------
    # Combined OOS
    # -------------------------------------------------------------------------

    pieces = []

    for strategy, df in loaded.items():
        oos = df[
            (df["entry_timestamp"] >= OOS_START) & (df["entry_timestamp"] <= OOS_END)
        ].copy()

        oos["strategy_name"] = strategy
        pieces.append(oos)

    if not pieces:
        add_finding(
            "FAIL",
            "OOS",
            "No strategy OOS data could be loaded.",
        )
        return

    combined = pd.concat(
        pieces,
        ignore_index=True,
    )

    actual_total = len(combined)

    if actual_total == EXPECTED_TOTAL:
        add_finding(
            "PASS",
            "OOS",
            f"Expected {EXPECTED_TOTAL:,} combined OOS trades, found {actual_total:,}.",
        )
    else:
        add_finding(
            "FAIL",
            "OOS",
            f"Expected {EXPECTED_TOTAL:,} combined OOS trades, found {actual_total:,}.",
        )

    # -------------------------------------------------------------------------
    # Combined strategy counts
    # -------------------------------------------------------------------------

    counts = combined["strategy_name"].value_counts().to_dict()

    for strategy, expected in EXPECTED_COUNTS.items():
        actual = int(counts.get(strategy, 0))

        if actual == expected:
            add_finding(
                "PASS",
                "OOS",
                f"Combined OOS {strategy}: {actual:,} trades.",
            )
        else:
            add_finding(
                "FAIL",
                "OOS",
                f"Combined OOS {strategy}: expected {expected:,}, found {actual:,}.",
            )

    # -------------------------------------------------------------------------
    # Basic metric reconciliation
    # -------------------------------------------------------------------------

    total_R = float(combined["r_multiple"].sum())
    expectancy = float(combined["r_multiple"].mean())

    wins = int((combined["r_multiple"] > 0).sum())
    win_rate = wins / len(combined) if len(combined) else float("nan")

    gross_profit = float(combined.loc[combined["r_multiple"] > 0, "r_multiple"].sum())
    gross_loss = float(-combined.loc[combined["r_multiple"] < 0, "r_multiple"].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily equity curve.
    daily = (
        combined.assign(date=combined["entry_timestamp"].dt.floor("D"))
        .groupby("date")["r_multiple"]
        .sum()
        .sort_index()
    )

    equity = daily.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    # Daily Sharpe / Sortino.
    daily_std = float(daily.std(ddof=1))

    if daily_std > 0:
        daily_sharpe = float(daily.mean() / daily_std * (252**0.5))
    else:
        daily_sharpe = float("nan")

    downside = daily[daily < 0]

    if len(downside) > 1:
        downside_std = float(downside.std(ddof=1))

        daily_sortino = (
            float(daily.mean() / downside_std * (252**0.5))
            if downside_std > 0
            else float("nan")
        )
    else:
        daily_sortino = float("nan")

    actual_metrics = {
        "total_R": total_R,
        "expectancy_R": expectancy,
        "PF": pf,
        "win_rate": win_rate,
        "max_DD": max_dd,
        "daily_Sharpe": daily_sharpe,
        "daily_Sortino": daily_sortino,
    }

    tolerances = {
        "total_R": 1e-5,
        "expectancy_R": 1e-6,
        "PF": 1e-6,
        "win_rate": 1e-6,
        "max_DD": 1e-6,
        "daily_Sharpe": 1e-6,
        "daily_Sortino": 1e-6,
    }

    for metric, expected in EXPECTED_METRICS.items():
        actual = actual_metrics[metric]
        tolerance = tolerances[metric]

        if pd.notna(actual) and abs(actual - expected) <= tolerance:
            add_finding(
                "PASS",
                "OOS_METRIC",
                f"{metric}: {actual:.9f} matches frozen value {expected:.9f}.",
            )
        else:
            add_finding(
                "FAIL",
                "OOS_METRIC",
                (f"{metric}: expected {expected:.9f}, found {actual:.9f}."),
            )


# =============================================================================
# STATIC LEAKAGE AUDIT
# =============================================================================

# Files that are not part of the final frozen execution system.
#
# Session Momentum was explicitly abandoned.
# Direction research and old exploratory research are not final strategies.
EXCLUDED_PATTERNS = [
    r"session_momentum",
    r"direction_",
    r"xgboost_direction",
    r"01_mean_reversion_exploration",
    r"04_feature_interactions",
]


def is_excluded_research_file(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()

    return any(re.search(pattern, normalized) for pattern in EXCLUDED_PATTERNS)


def classify_known_safe_pattern(
    path: Path,
    line_number: int,
    line_text: str,
) -> tuple[str, str] | None:

    normalized = str(path).replace("\\", "/").lower()

    # -------------------------------------------------------------------------
    # Model freeze audit scanning itself
    # -------------------------------------------------------------------------

    if normalized.endswith("src/research/portfolio/23_model_freeze_audit.py"):
        if "rolling" in line_text and "center=True" in line_text:
            return (
                "SAFE",
                "Scanner self-reference: centered rolling pattern exists only in the audit implementation.",
            )

    # -------------------------------------------------------------------------
    # Target construction
    # -------------------------------------------------------------------------

    if normalized.endswith("src/targets.py"):
        if "shift(-" in line_text:
            return (
                "SAFE",
                "Future target construction. Negative shift is used to define supervised future targets, not execution features.",
            )

    # -------------------------------------------------------------------------
    # HMM transition analysis
    # -------------------------------------------------------------------------

    if normalized.endswith("src/research/regime/hmm_transition_analysis.py"):
        if "shift(-1)" in line_text:
            return (
                "SAFE",
                "Descriptive next-state analysis. The shifted state is not used to generate trading entries.",
            )

    # -------------------------------------------------------------------------
    # ORB reconciliation
    # -------------------------------------------------------------------------

    if normalized.endswith("src/research/orb/15_orb_reconciliation.py"):
        if "session_df.iloc[-1]" in line_text:
            return (
                "SAFE",
                "Uses the final bar of the current session for the defined end-of-session exit rule.",
            )

    # -------------------------------------------------------------------------
    # Pure reporting / metadata
    # -------------------------------------------------------------------------

    reporting_patterns = [
        (
            "src/research/orb/20_orb_modular_reproduction.py",
            "trades['entry_timestamp'].iloc[-1]",
        ),
        (
            "src/research/portfolio/21_mr_orb_portfolio_analysis.py",
            "market['timestamp'].iloc[-1]",
        ),
        (
            "src/research/portfolio/22_full_system_funded_simulation.py",
            "df['entry_timestamp'].iloc[-1]",
        ),
    ]

    for suffix, token in reporting_patterns:
        if normalized.endswith(suffix) and token in line_text:
            return (
                "SAFE",
                "Last-element access is used only for reporting/metadata and does not affect trading decisions.",
            )

    return None


def static_leakage_audit():
    banner("3. STATIC LEAKAGE AUDIT")

    python_files = list((ROOT / "src").rglob("*.py"))

    audit_path = Path(__file__).resolve()

    python_files = [path for path in python_files if path.resolve() != audit_path]

    patterns = [
        (
            "shift(-",
            re.compile(r"\.shift\(\s*-\s*\d+"),
            "Negative shift can introduce future information.",
        ),
        (
            "rolling(center=True)",
            re.compile(
                r"\.rolling\([^)]*center\s*=\s*True",
                re.IGNORECASE,
            ),
            "Centered rolling windows can include future observations.",
        ),
        (
            "bfill",
            re.compile(r"\.bfill\s*\("),
            "Backward fill can propagate future observations backward.",
        ),
        (
            "backfill",
            re.compile(r"\.backfill\s*\("),
            "Backward fill can propagate future observations backward.",
        ),
        (
            "expanding",
            re.compile(r"\.expanding\s*\("),
            "Expanding windows require manual review for temporal causality.",
        ),
        (
            "iloc[-1]",
            re.compile(r"\.iloc\[\s*-\s*1\s*\]"),
            "Negative indexing requires manual review in time-series logic.",
        ),
        (
            "merge_asof(forward)",
            re.compile(
                r"merge_asof\s*\([^)]*direction\s*=\s*[\"']forward[\"']",
                re.IGNORECASE,
            ),
            "Forward asof merge can introduce future information.",
        ),
    ]

    review_count = 0
    safe_count = 0
    excluded_count = 0

    for path in python_files:
        if is_excluded_research_file(path):
            excluded_count += 1

            # We do not emit one finding per file because that would flood
            # the manifest. The exclusion is documented in the summary.
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception as exc:
            add_finding(
                "REVIEW",
                "LEAKAGE",
                f"Could not read source file for static scan: {exc}",
                str(path.relative_to(ROOT)),
            )
            continue

        lines = text.splitlines()

        for line_number, line_text in enumerate(lines, start=1):
            for pattern_name, pattern, description in patterns:
                if not pattern.search(line_text):
                    continue

                classified = classify_known_safe_pattern(
                    path,
                    line_number,
                    line_text,
                )

                if classified is not None:
                    level, reason = classified

                    add_finding(
                        level,
                        "LEAKAGE",
                        (f"{pattern_name} classified SAFE: {reason}"),
                        str(path.relative_to(ROOT)),
                        line_number,
                    )

                    safe_count += 1
                    continue

                add_finding(
                    "REVIEW",
                    "LEAKAGE",
                    (f"{pattern_name}: {description}"),
                    str(path.relative_to(ROOT)),
                    line_number,
                )

                review_count += 1

    add_finding(
        "PASS",
        "LEAKAGE_SUMMARY",
        (
            f"Static scan completed. "
            f"SAFE findings: {safe_count:,}; "
            f"REVIEW findings: {review_count:,}; "
            f"excluded legacy research files: {excluded_count:,}."
        ),
    )


# =============================================================================
# REPRODUCIBILITY INVENTORY
# =============================================================================


def reproducibility_inventory():
    banner("4. REPRODUCIBILITY INVENTORY")

    research_files = []

    for path in (ROOT / "src").rglob("*.py"):
        name = path.name.lower()

        if any(
            token in name
            for token in [
                "reproduction",
                "reconcile",
                "validation",
                "portfolio",
                "robustness",
                "funded",
            ]
        ):
            research_files.append(path)

    research_files = sorted(research_files)

    if not research_files:
        add_finding(
            "REVIEW",
            "REPRODUCIBILITY",
            "No reproducibility-related Python files were found.",
        )
        return

    add_finding(
        "PASS",
        "REPRODUCIBILITY",
        (
            f"Reproducibility inventory found "
            f"{len(research_files):,} relevant Python files."
        ),
    )

    for path in research_files:
        print(f"  - {path.relative_to(ROOT)}")


# =============================================================================
# MANIFEST
# =============================================================================


def write_manifest():
    banner("5. WRITING MODEL FREEZE MANIFEST")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pass_count = sum(1 for item in findings if item["level"] == "PASS")

    review_count = sum(1 for item in findings if item["level"] == "REVIEW")

    fail_count = sum(1 for item in findings if item["level"] == "FAIL")

    manifest = {
        "model_version": MODEL_VERSION,
        "official_oos": {
            "start": str(OOS_START),
            "end": str(OOS_END),
        },
        "post_oos_start": str(POST_OOS_START),
        "strategies": STRATEGIES,
        "expected_oos_counts": EXPECTED_COUNTS,
        "expected_total_oos_trades": EXPECTED_TOTAL,
        "expected_metrics": EXPECTED_METRICS,
        "audit_summary": {
            "pass": pass_count,
            "review": review_count,
            "fail": fail_count,
        },
        "gate": (
            "PASS"
            if fail_count == 0 and review_count == 0
            else ("PASS WITH MANUAL REVIEW ITEMS" if fail_count == 0 else "FAIL")
        ),
        "findings": findings,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Manifest written to: {MANIFEST_PATH.relative_to(ROOT)}")


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("\n" + "=" * 88)
    print("MODEL FREEZE / PRE-PAPER AUDIT")
    print("=" * 88)

    print(f"Project root: {ROOT}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Official OOS: {OOS_START.date()} -> {OOS_END.date()}")
    print(f"Strategies: {', '.join(STRATEGIES)}")

    audit_artifacts()
    audit_oos()
    static_leakage_audit()
    reproducibility_inventory()
    write_manifest()

    pass_count = sum(1 for item in findings if item["level"] == "PASS")

    review_count = sum(1 for item in findings if item["level"] == "REVIEW")

    fail_count = sum(1 for item in findings if item["level"] == "FAIL")

    banner("FINAL AUDIT SUMMARY")

    print(f"{'PASS':>8}: {pass_count}")
    print(f"{'REVIEW':>8}: {review_count}")
    print(f"{'FAIL':>8}: {fail_count}")

    if fail_count > 0:
        print("\nMODEL FREEZE GATE: FAIL")
        print("Hard artifact/OOS failures remain.")
    elif review_count > 0:
        print("\nMODEL FREEZE GATE: PASS WITH MANUAL REVIEW ITEMS")
        print("Remaining REVIEW findings require human inspection.")
    else:
        print("\nMODEL FREEZE GATE: PASS")
        print("No hard failures or unresolved review findings remain.")

    print(f"Freeze manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
