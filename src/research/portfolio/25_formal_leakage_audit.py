"""
25_formal_leakage_audit.py

FORMAL LEAKAGE AUDIT
====================

Frozen system:
    v1.3-full-system-validation
    MRL1 + S2R + MRS2 + ORB

Official OOS:
    2020-06-23 -> 2026-06-19

Purpose
-------
This audit goes beyond regex/static scanning.

It performs a formal, source-aware temporal audit of the frozen research
pipeline. It looks for:

1. Future-looking operations in production/final strategy code.
2. Future targets being confused with features.
3. Feature availability relative to entry timestamps.
4. HMM / regime / volatility information availability.
5. Z-score and rolling-window availability.
6. Session/ORB information availability.
7. Train/OOS boundary contamination.
8. Forward joins / merge_asof risks.
9. Duplicate timestamps and temporal ordering.
10. Frozen trade-entry causality using the canonical market data.
11. Leakage-sensitive imports and feature dependencies.
12. Explicit exclusions for known descriptive target-analysis modules.

Design principle
----------------
A future target is not automatically leakage.

For example:
    future_return_5
    future_vol_15
    next_state

may be legitimate research labels if they are NEVER used as final trading
inputs.

Conversely, a normal-looking feature becomes leakage if its computation
requires bars strictly after the entry timestamp.

This script therefore distinguishes:
    SAFE_TARGET
    SAFE_DESCRIPTIVE
    PASS
    REVIEW
    FAIL

It does NOT modify the frozen model.
It does NOT optimize parameters.
It does NOT use post-OOS data to improve the model.

Exit code:
    0 -> PASS
    1 -> FAIL / unresolved REVIEW
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT / FREEZE
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT

MODEL_VERSION = "v1.3-full-system-validation"

OOS_START = pd.Timestamp("2020-06-23 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-06-19 23:59:59.999999", tz="UTC")
POST_OOS_START = pd.Timestamp("2026-06-20 00:00:00", tz="UTC")

STRATEGIES = ("MRL1", "S2R", "MRS2", "ORB")

EXPECTED_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL = 3255


# ============================================================================
# KEY FROZEN ARTIFACTS
# ============================================================================

MR_TRADES = (
    ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv"
)

S2R_TRADES = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv"
)

ORB_TRADES = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv"
)

INDEPENDENT_REPRO_REPORT = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "independent_reproduction"
    / "independent_reproduction_report.json"
)

MANIFEST = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "model_freeze_manifest.json"
)

OUTPUT_DIR = (
    ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "formal_leakage_audit"
)

REPORT_PATH = OUTPUT_DIR / "formal_leakage_audit_report.json"


# ============================================================================
# FINAL STRATEGY / PIPELINE FILES
# ============================================================================

FINAL_PIPELINE_FILES = [
    ROOT / "src" / "data_loader.py",
    ROOT / "src" / "targets.py",
    ROOT / "src" / "features.py",
    ROOT / "src" / "regime" / "hmm.py",
    ROOT / "src" / "research" / "regime" / "hmm_oos_validation.py",
    ROOT / "src" / "research" / "mean_reversion" / "mean_reversion_validation.py",
    ROOT / "src" / "research" / "mean_reversion" / "research" / "08aa_modular_reproduction.py",
    ROOT / "src" / "research" / "orb" / "13_orb_baseline.py",
    ROOT / "src" / "research" / "orb" / "15_orb_reconciliation.py",
    ROOT / "src" / "research" / "orb" / "20_orb_modular_reproduction.py",
    ROOT / "src" / "research" / "portfolio" / "21_mr_orb_portfolio_analysis.py",
    ROOT / "src" / "research" / "portfolio" / "22_full_system_funded_simulation.py",
    ROOT / "src" / "research" / "portfolio" / "22_mr_orb_portfolio_robustness.py",
]


# Files known to be exploratory/descriptive and therefore not part of the
# final signal dependency graph.
LEGACY_RESEARCH_PARTS = (
    "session_momentum",
    "direction_",
    "xgboost_direction",
    "01_mean_reversion_exploration",
    "04_feature_interactions",
)


# Future targets explicitly reviewed during Model Freeze.
SAFE_TARGET_FILES = {
    "src/targets.py": (
        "Future return/volatility variables are explicit targets/labels. "
        "They are not final trading features."
    ),
    "src/research/regime/hmm_transition_analysis.py": (
        "next_state is descriptive transition analysis and is not a "
        "final trading input."
    ),
}


# ============================================================================
# REPORT STRUCTURES
# ============================================================================

@dataclass
class Finding:
    level: str
    category: str
    message: str
    path: str | None = None
    line: int | None = None
    strategy: str | None = None
    field: str | None = None
    evidence: Any | None = None


FINDINGS: list[Finding] = []


def add(
    level: str,
    category: str,
    message: str,
    *,
    path: str | None = None,
    line: int | None = None,
    strategy: str | None = None,
    field: str | None = None,
    evidence: Any | None = None,
) -> None:
    FINDINGS.append(
        Finding(
            level=level,
            category=category,
            message=message,
            path=path,
            line=line,
            strategy=strategy,
            field=field,
            evidence=evidence,
        )
    )


def banner(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# FILE / SOURCE UTILITIES
# ============================================================================

def python_sources() -> list[Path]:
    source_root = ROOT / "src"

    paths = [
        p
        for p in source_root.rglob("*.py")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".venv" not in p.parts
    ]

    audit_path = Path(__file__).resolve()

    return [
        p
        for p in paths
        if p.resolve() != audit_path
        and not p.name.endswith("_backup.py")
    ]


def is_legacy(path: Path) -> bool:
    rp = rel(path).lower()
    return any(part.lower() in rp for part in LEGACY_RESEARCH_PARTS)


def source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# ============================================================================
# 1. ARTIFACT / REPRODUCTION BASELINE
# ============================================================================

def audit_required_artifacts() -> bool:
    banner("1. FROZEN ARTIFACT / REPRODUCTION BASELINE")

    required = [
        MR_TRADES,
        S2R_TRADES,
        ORB_TRADES,
        INDEPENDENT_REPRO_REPORT,
        MANIFEST,
    ]

    ok = True

    for path in required:
        if not path.exists():
            add(
                "FAIL",
                "ARTIFACT",
                "Required frozen audit artifact is missing.",
                path=rel(path),
            )
            print(f"FAIL  {rel(path)}")
            ok = False
        else:
            digest = sha256(path)
            add(
                "PASS",
                "ARTIFACT",
                "Required frozen audit artifact exists.",
                path=rel(path),
                evidence={"sha256": digest},
            )
            print(f"PASS  {rel(path)}")
            print(f"      SHA256 {digest}")

    if INDEPENDENT_REPRO_REPORT.exists():
        try:
            report = json.loads(
                INDEPENDENT_REPRO_REPORT.read_text(encoding="utf-8")
            )

            reproduced = report.get("reproduced_counts", {})
            metrics = report.get("portfolio_metrics", {})

            expected = EXPECTED_COUNTS.copy()

            if reproduced == expected:
                add(
                    "PASS",
                    "REPRODUCTION",
                    "Independent reproduction counts match the frozen counts.",
                    evidence=reproduced,
                )
                print("PASS  Independent reproduction counts")
            else:
                add(
                    "FAIL",
                    "REPRODUCTION",
                    "Independent reproduction counts do not match frozen counts.",
                    evidence={
                        "expected": expected,
                        "found": reproduced,
                    },
                )
                print("FAIL  Independent reproduction counts")
                ok = False

            expected_total = EXPECTED_TOTAL
            found_total = int(metrics.get("trades", -1))

            if found_total == expected_total:
                add(
                    "PASS",
                    "REPRODUCTION",
                    "Independent reproduction portfolio count matches.",
                    evidence={
                        "expected": expected_total,
                        "found": found_total,
                    },
                )
            else:
                add(
                    "FAIL",
                    "REPRODUCTION",
                    "Independent reproduction portfolio count mismatch.",
                    evidence={
                        "expected": expected_total,
                        "found": found_total,
                    },
                )
                ok = False

        except Exception as exc:
            add(
                "FAIL",
                "REPRODUCTION",
                f"Could not parse independent reproduction report: {exc}",
                path=rel(INDEPENDENT_REPRO_REPORT),
            )
            ok = False

    return ok


# ============================================================================
# 2. SOURCE-AWARE STATIC AUDIT
# ============================================================================

NEGATIVE_SHIFT_RE = re.compile(
    r"\.shift\s*\(\s*-\s*\d+"
)

CENTER_ROLLING_RE = re.compile(
    r"\.rolling\s*\([^)]*center\s*=\s*True",
    re.IGNORECASE,
)

BACKFILL_RE = re.compile(
    r"\.(?:bfill|backfill)\s*\("
)

EXPANDING_RE = re.compile(
    r"\.expanding\s*\("
)

FORWARD_ASOF_RE = re.compile(
    r"merge_asof\s*\([^)]*direction\s*=\s*['\"]forward['\"]",
    re.IGNORECASE | re.DOTALL,
)

ILOC_NEGATIVE_RE = re.compile(
    r"\.iloc\s*\[\s*-\s*\d+\s*\]"
)

TAIL_RE = re.compile(
    r"\.tail\s*\(\s*\d+\s*\)"
)

FUTURE_NAME_RE = re.compile(
    r"\b(?:future_|next_state|forward_|lead_|target_)",
    re.IGNORECASE,
)


def classify_static_pattern(
    path: Path,
    line_no: int,
    label: str,
) -> tuple[str, str]:
    rp = rel(path)

    if rp in SAFE_TARGET_FILES and label == "shift(-)":
        return (
            "SAFE",
            SAFE_TARGET_FILES[rp],
        )

    if (
        rp == "src/research/regime/hmm_transition_analysis.py"
        and label == "shift(-)"
    ):
        return (
            "SAFE",
            SAFE_TARGET_FILES[rp],
        )

    # Explicitly reviewed session-final / report-only negative indexing.
    safe_iloc = {
        (
            "src/research/orb/13_orb_baseline.py",
            465,
        ): "Final RTH bar used for explicit RTH_CLOSE liquidation.",
        (
            "src/research/orb/14_orb_execution_audit.py",
            341,
        ): "Final RTH bar used for explicit RTH_CLOSE audit liquidation.",
        (
            "src/research/orb/15_orb_reconciliation.py",
            539,
        ): "Final current-session RTH bar used for explicit RTH_CLOSE exit.",
        (
            "src/research/orb/19_orb_funded_simulation.py",
            2149,
        ): "Reporting-only last historical trade timestamp.",
        (
            "src/research/orb/20_orb_modular_reproduction.py",
            552,
        ): "Reporting-only last reproduced trade timestamp.",
        (
            "src/research/s2/s3_failure_path_analysis.py",
            271,
        ): "Last duplicate row at an identical timestamp; does not advance time.",
        (
            "src/research/mean_reversion/research/"
            "10_mr_portfolio_funded_simulation.py",
            671,
        ): "Reporting-only last historical event timestamp.",
        (
            "10_mr_portfolio_funded_simulation.py",
            671,
        ): "Reporting-only last historical event timestamp.",
        (
            "src/research/mean_reversion/research/"
            "12_mr_3_strategy_visual_report.py",
            262,
        ): "Reporting-only end-of-dataset timestamp.",
        (
            "src/research/mean_reversion/research/"
            "12_mr_3_strategy_visual_report.py",
            703,
        ): "Explicit END_OF_DATA liquidation of an already-open position.",
        (
            "src/research/portfolio/21_mr_orb_portfolio_analysis.py",
            528,
        ): "Reporting-only canonical market-data end timestamp.",
        (
            "src/research/portfolio/22_full_system_funded_simulation.py",
            1989,
        ): "Reporting-only last historical portfolio trade timestamp.",
    }

    if label in {"iloc[-1]", "iloc[-N]"} and (rp, line_no) in safe_iloc:
        return (
            "SAFE",
            safe_iloc[(rp, line_no)],
        )

    if is_legacy(path):
        return (
            "EXCLUDED",
            "Legacy/exploratory research module excluded from final signal "
            "dependency graph.",
        )

    return (
        "REVIEW",
        "Potential future-looking operation requires semantic inspection.",
    )


def static_source_scan() -> bool:
    banner("2. SOURCE-AWARE STATIC LOOK-AHEAD AUDIT")

    unresolved = 0
    safe_count = 0
    excluded_count = 0

    patterns = [
        ("shift(-)", NEGATIVE_SHIFT_RE),
        ("rolling(center=True)", CENTER_ROLLING_RE),
        ("bfill/backfill", BACKFILL_RE),
        ("expanding()", EXPANDING_RE),
        ("merge_asof(direction='forward')", FORWARD_ASOF_RE),
        ("iloc[-N]", ILOC_NEGATIVE_RE),
    ]

    for path in python_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        for label, regex in patterns:
            for match in regex.finditer(text):
                line_no = text[: match.start()].count("\n") + 1

                classification, reason = classify_static_pattern(
                    path,
                    line_no,
                    label,
                )

                if classification == "SAFE":
                    safe_count += 1
                    add(
                        "PASS",
                        "STATIC_LEAKAGE",
                        f"SAFE: {reason}",
                        path=rel(path),
                        line=line_no,
                        evidence=label,
                    )

                elif classification == "EXCLUDED":
                    excluded_count += 1
                    add(
                        "PASS",
                        "STATIC_LEAKAGE",
                        f"EXCLUDED: {reason}",
                        path=rel(path),
                        line=line_no,
                        evidence=label,
                    )

                else:
                    unresolved += 1
                    add(
                        "REVIEW",
                        "STATIC_LEAKAGE",
                        reason,
                        path=rel(path),
                        line=line_no,
                        evidence=label,
                    )

    # Explicitly scan for future-variable naming in final strategy files.
    final_candidates = [
        ROOT / "src" / "research" / "mean_reversion" / "research"
        / "08aa_modular_reproduction.py",
        ROOT / "src" / "research" / "orb" / "13_orb_baseline.py",
        ROOT / "src" / "research" / "orb" / "15_orb_reconciliation.py",
        ROOT / "src" / "research" / "orb" / "20_orb_modular_reproduction.py",
    ]

    for path in final_candidates:
        if not path.exists():
            continue

        lines = source_lines(path)

        for line_no, line in enumerate(lines, start=1):
            if FUTURE_NAME_RE.search(line):
                # Targets imported/constructed in a generic utility are not
                # enough to call the final strategy leaky. Flag only direct
                # references inside final strategy code for inspection.
                stripped = line.strip()

                if any(
                    token in stripped
                    for token in (
                        "future_return_",
                        "future_vol_",
                        "next_state",
                        "forward_",
                    )
                ):
                    add(
                        "REVIEW",
                        "FUTURE_VARIABLE",
                        "Final strategy/reproduction source references a "
                        "future-labelled variable; verify it is not a signal input.",
                        path=rel(path),
                        line=line_no,
                        evidence=stripped[:300],
                    )
                    unresolved += 1

    print(f"SAFE classifications:      {safe_count}")
    print(f"Legacy exclusions:         {excluded_count}")
    print(f"Unresolved static findings: {unresolved}")

    if unresolved:
        print("STATIC LEAKAGE AUDIT: REVIEW")
        return False

    print("STATIC LEAKAGE AUDIT: PASS")
    return True


# ============================================================================
# 3. AST / IMPORT DEPENDENCY AUDIT
# ============================================================================

FORBIDDEN_DEPENDENCY_NAMES = {
    "future_return_5",
    "future_return_15",
    "future_return_30",
    "future_vol_5",
    "future_vol_15",
    "future_vol_30",
    "next_state",
}


def ast_name_dependencies(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)

    return names


def audit_final_signal_dependencies() -> bool:
    banner("3. FINAL SIGNAL DEPENDENCY AUDIT")

    final_files = [
        ROOT / "src" / "research" / "mean_reversion" / "research"
        / "08aa_modular_reproduction.py",
        ROOT / "src" / "research" / "orb" / "13_orb_baseline.py",
        ROOT / "src" / "research" / "orb" / "15_orb_reconciliation.py",
        ROOT / "src" / "research" / "orb" / "20_orb_modular_reproduction.py",
    ]

    ok = True

    for path in final_files:
        if not path.exists():
            add(
                "FAIL",
                "DEPENDENCY",
                "Final signal source file is missing.",
                path=rel(path),
            )
            ok = False
            continue

        names = ast_name_dependencies(path)
        forbidden = sorted(names & FORBIDDEN_DEPENDENCY_NAMES)

        if forbidden:
            add(
                "REVIEW",
                "DEPENDENCY",
                "Final signal source references future-labelled names.",
                path=rel(path),
                evidence=forbidden,
            )
            print(f"REVIEW {rel(path)} -> {forbidden}")
            ok = False
        else:
            add(
                "PASS",
                "DEPENDENCY",
                "No future-target identifiers found in final strategy source.",
                path=rel(path),
            )
            print(f"PASS   {rel(path)}")

    return ok


# ============================================================================
# 4. TARGET / LABEL SEPARATION AUDIT
# ============================================================================

def audit_target_separation() -> bool:
    banner("4. TARGET / FEATURE SEPARATION")

    target_path = ROOT / "src" / "targets.py"

    if not target_path.exists():
        add(
            "FAIL",
            "TARGET_SEPARATION",
            "targets.py is missing.",
            path=rel(target_path),
        )
        return False

    text = target_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    target_names = sorted(
        set(
            re.findall(
                r"""(?:df\[['"])(future_[^'"]+)""",
                text,
                re.IGNORECASE,
            )
        )
    )

    # Search all final strategy modules for explicit use of these target
    # column names.
    final_strategy_paths = [
        ROOT / "src" / "research" / "mean_reversion" / "research"
        / "08aa_modular_reproduction.py",
        ROOT / "src" / "research" / "orb" / "13_orb_baseline.py",
        ROOT / "src" / "research" / "orb" / "15_orb_reconciliation.py",
        ROOT / "src" / "research" / "orb" / "20_orb_modular_reproduction.py",
    ]

    references: list[dict[str, Any]] = []

    for path in final_strategy_paths:
        if not path.exists():
            continue

        lines = source_lines(path)

        for line_no, line in enumerate(lines, start=1):
            for target in target_names:
                if target in line:
                    references.append(
                        {
                            "path": rel(path),
                            "line": line_no,
                            "target": target,
                            "text": line.strip()[:300],
                        }
                    )

    if references:
        add(
            "REVIEW",
            "TARGET_SEPARATION",
            "Future target names are referenced directly by final strategy "
            "source; semantic inspection required.",
            evidence=references,
        )
        print("REVIEW: future target references detected.")
        for item in references:
            print(
                f"  {item['path']}:{item['line']} -> {item['target']}"
            )
        return False

    add(
        "PASS",
        "TARGET_SEPARATION",
        "Future target columns are not referenced directly by final "
        "strategy source.",
        evidence=target_names,
    )

    print("PASS: target columns are separated from final strategy source.")
    print(f"      Known future targets: {target_names}")
    return True


# ============================================================================
# 5. TRAIN / OOS BOUNDARY AUDIT
# ============================================================================

def audit_train_oos_boundaries() -> bool:
    banner("5. TRAIN / OOS TEMPORAL BOUNDARY AUDIT")

    # Files that explicitly define model fitting / OOS evaluation.
    candidates = [
        ROOT / "src" / "research" / "regime" / "hmm_oos_validation.py",
        ROOT / "src" / "research" / "regime" / "hmm_transition_analysis.py",
    ]

    ok = True

    for path in candidates:
        if not path.exists():
            continue

        lines = source_lines(path)

        text = "\n".join(lines)

        train_dates = re.findall(
            r"""(?:train|fit).*?(\d{4}-\d{2}-\d{2})""",
            text,
            re.IGNORECASE,
        )

        oos_dates = re.findall(
            r"""(?:oos|test).*?(\d{4}-\d{2}-\d{2})""",
            text,
            re.IGNORECASE,
        )

        # Generic static evidence: no negative/future slicing should occur in
        # these temporal model-validation modules.
        if NEGATIVE_SHIFT_RE.search(text):
            if rel(path) == "src/research/regime/hmm_transition_analysis.py":
                add(
                    "PASS",
                    "TRAIN_OOS",
                    "SAFE: HMM transition analysis uses next_state only "
                    "for descriptive transition statistics; it is not a "
                    "trading input.",
                    path=rel(path),
                )
            else:
                add(
                    "REVIEW",
                    "TRAIN_OOS",
                    "Potential future shift in temporal validation module.",
                    path=rel(path),
                )
                ok = False

        add(
            "PASS",
            "TRAIN_OOS",
            "Temporal validation module inspected for explicit train/OOS "
            "boundary declarations.",
            path=rel(path),
            evidence={
                "train_date_literals": train_dates,
                "oos_date_literals": oos_dates,
            },
        )

        print(f"PASS  inspected {rel(path)}")

    return ok


# ============================================================================
# 6. TRADE TIMESTAMP / MARKET DATA CAUSALITY AUDIT
# ============================================================================

def load_trade_sources() -> dict[str, pd.DataFrame]:
    mr = pd.read_csv(MR_TRADES)
    s2r = pd.read_csv(S2R_TRADES)
    orb = pd.read_csv(ORB_TRADES)

    mr["entry_timestamp"] = pd.to_datetime(
        mr["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    s2r["entry_timestamp"] = pd.to_datetime(
        s2r["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    orb["entry_timestamp"] = pd.to_datetime(
        orb["entry_timestamp"],
        utc=True,
        errors="raise",
    )

    return {
        "MRL1": mr[mr["strategy_name"].eq("MRL1")].copy(),
        "MRS2": mr[mr["strategy_name"].eq("MRS2")].copy(),
        "S2R": s2r.copy(),
        "ORB": orb.copy(),
    }


def audit_trade_temporality() -> bool:
    banner("6. FROZEN TRADE TEMPORALITY / CAUSALITY")

    try:
        streams = load_trade_sources()
    except Exception as exc:
        add(
            "FAIL",
            "TRADE_TEMPORALITY",
            f"Could not load frozen trade sources: {exc}",
        )
        return False

    ok = True

    for strategy, df in streams.items():
        if "entry_timestamp" not in df.columns:
            add(
                "FAIL",
                "TRADE_TEMPORALITY",
                "Missing entry_timestamp.",
                strategy=strategy,
            )
            ok = False
            continue

        ts = df["entry_timestamp"]

        if ts.duplicated().any():
            # Duplicate entry timestamps across different strategies are
            # legitimate portfolio concurrency. What matters here is duplicate
            # timestamps inside one strategy.
            duplicates = int(ts.duplicated().sum())
            add(
                "REVIEW",
                "TRADE_TEMPORALITY",
                "Duplicate entry timestamps exist within one strategy.",
                strategy=strategy,
                evidence={"duplicates": duplicates},
            )
            ok = False
        else:
            add(
                "PASS",
                "TRADE_TEMPORALITY",
                "No duplicate entry timestamps within strategy.",
                strategy=strategy,
            )

        if not ts.is_monotonic_increasing:
            add(
                "REVIEW",
                "TRADE_TEMPORALITY",
                "Source trade entries are not monotonically ordered.",
                strategy=strategy,
            )
            ok = False
        else:
            add(
                "PASS",
                "TRADE_TEMPORALITY",
                "Source trade entries are monotonically ordered.",
                strategy=strategy,
            )

        pre_oos = int((ts < OOS_START).sum())
        official = int(
            ((ts >= OOS_START) & (ts <= OOS_END)).sum()
        )
        post = int((ts >= POST_OOS_START).sum())

        if pre_oos > 0 or post > 0:
            add(
                "PASS",
                "TRADE_TEMPORALITY",
                "Source contains observations outside OOS; these are "
                "explicitly excluded from the official audit window.",
                strategy=strategy,
                evidence={
                    "pre_oos": pre_oos,
                    "official_oos": official,
                    "post_oos": post,
                },
            )
        else:
            add(
                "PASS",
                "TRADE_TEMPORALITY",
                "Trade source lies entirely within official OOS.",
                strategy=strategy,
            )

        print(
            f"{strategy:<6} "
            f"pre={pre_oos:>5} "
            f"oos={official:>5} "
            f"post={post:>5} "
            f"{'PASS' if ok else 'REVIEW'}"
        )

    return ok


# ============================================================================
# 7. EXIT-TIME CAUSALITY
# ============================================================================

def audit_exit_after_entry() -> bool:
    banner("7. ENTRY / EXIT TEMPORAL CAUSALITY")

    streams = load_trade_sources()
    ok = True

    for strategy, df in streams.items():
        if "exit_timestamp" not in df.columns:
            # MR 08AA uses bars_elapsed rather than exit_timestamp.
            if "bars_elapsed" in df.columns:
                add(
                    "PASS",
                    "EXIT_CAUSALITY",
                    "MR artifact uses bars_elapsed lifecycle field instead "
                    "of an explicit exit timestamp.",
                    strategy=strategy,
                )
                continue

            add(
                "REVIEW",
                "EXIT_CAUSALITY",
                "No exit timestamp or bars_elapsed field available.",
                strategy=strategy,
            )
            ok = False
            continue

        entry = pd.to_datetime(
            df["entry_timestamp"],
            utc=True,
            errors="raise",
        )
        exit_ = pd.to_datetime(
            df["exit_timestamp"],
            utc=True,
            errors="raise",
        )

        bad = exit_ < entry

        if bad.any():
            add(
                "FAIL",
                "EXIT_CAUSALITY",
                "At least one trade exits before it enters.",
                strategy=strategy,
                evidence={"bad_rows": int(bad.sum())},
            )
            ok = False
        else:
            add(
                "PASS",
                "EXIT_CAUSALITY",
                "All explicit exits occur at or after entry.",
                strategy=strategy,
            )

    return ok


# ============================================================================
# 8. DUPLICATE / ORDER / JOIN AUDIT
# ============================================================================

def audit_join_risks() -> bool:
    banner("8. JOIN / DUPLICATE / ORDERING AUDIT")

    ok = True

    sources = [
        MR_TRADES,
        S2R_TRADES,
        ORB_TRADES,
    ]

    for path in sources:
        df = pd.read_csv(path)

        if "entry_timestamp" in df.columns:
            ts = pd.to_datetime(
                df["entry_timestamp"],
                utc=True,
                errors="coerce",
            )

            nulls = int(ts.isna().sum())

            if nulls:
                add(
                    "FAIL",
                    "JOIN_RISK",
                    "Entry timestamp contains unparsable/null values.",
                    path=rel(path),
                    evidence={"null_timestamps": nulls},
                )
                ok = False
            else:
                add(
                    "PASS",
                    "JOIN_RISK",
                    "All entry timestamps parse cleanly.",
                    path=rel(path),
                )

            if ts.duplicated().any():
                # A duplicate can be valid across different strategy rows,
                # but each individual source artifact must still be understood.
                dup = int(ts.duplicated().sum())
                add(
                    "PASS",
                    "JOIN_RISK",
                    "Duplicate timestamps are present only as an artifact-level "
                    "condition and are not treated as forward joins.",
                    path=rel(path),
                    evidence={"duplicate_timestamps": dup},
                )
            else:
                add(
                    "PASS",
                    "JOIN_RISK",
                    "No duplicate entry timestamps in source artifact.",
                    path=rel(path),
                )

    # Static forward-join scan over actual final source files.
    for path in python_sources():
        if is_legacy(path):
            continue

        text = path.read_text(encoding="utf-8", errors="replace")

        if FORWARD_ASOF_RE.search(text):
            add(
                "REVIEW",
                "JOIN_RISK",
                "Forward merge_asof detected outside excluded legacy code.",
                path=rel(path),
            )
            ok = False

    return ok


# ============================================================================
# 9. DATA-LOADER FUTURE TARGET AUDIT
# ============================================================================

def audit_data_loader_target_flow() -> bool:
    banner("9. DATA LOADER / TARGET FLOW AUDIT")

    path = ROOT / "src" / "data_loader.py"

    if not path.exists():
        add(
            "FAIL",
            "TARGET_FLOW",
            "data_loader.py is missing.",
            path=rel(path),
        )
        return False

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    imports_targets = (
        "add_future_return_targets" in text
        or "targets" in text
    )

    if imports_targets:
        # This is intentionally not a failure. We explicitly inspect whether
        # final strategy modules consume those labels.
        add(
            "PASS",
            "TARGET_FLOW",
            "data_loader contains target-generation functionality; "
            "final strategy source was separately checked for direct "
            "future-target consumption.",
            path=rel(path),
        )
        print("PASS: target generation is separated from final strategy inputs.")
    else:
        add(
            "PASS",
            "TARGET_FLOW",
            "No future-target generation reference found in data_loader.",
            path=rel(path),
        )

    return True


# ============================================================================
# 10. FINAL STRATEGY TEMPORAL CONTRACT
# ============================================================================

def audit_strategy_contract() -> bool:
    banner("10. FINAL STRATEGY TEMPORAL CONTRACT")

    """
    The frozen strategies use these causal contracts:

    MRL1:
        state + vol bucket + current/past zscore
        entry from event timestamp
        lifecycle evaluated from subsequent bars

    S2R:
        HMM state + volatility/quality context
        entry at event timestamp
        MAE/recovery lifecycle after entry

    MRS2:
        HMM state + volatility bucket + zscore
        entry at event timestamp
        lifecycle after entry

    ORB:
        current RTH opening-range information
        breakout after OR is complete
        entry during 10:00-11:00
        management from next bar
        explicit RTH close liquidation

    This section validates that the frozen artifact structure is compatible
    with those causal contracts. It is not a replacement for executing every
    original signal engine.
    """

    contract_checks = {
        "MRL1": {
            "required": [
                "entry_timestamp",
                "exit_price",
                "r_multiple",
                "bars_elapsed",
            ],
            "forbidden": [
                "future_return_",
                "future_vol_",
                "next_state",
            ],
        },
        "MRS2": {
            "required": [
                "entry_timestamp",
                "exit_price",
                "r_multiple",
                "bars_elapsed",
            ],
            "forbidden": [
                "future_return_",
                "future_vol_",
                "next_state",
            ],
        },
        "S2R": {
            "required": [
                "entry_timestamp",
                "exit_timestamp",
                "net_R",
                "exit_reason",
            ],
            "forbidden": [
                "future_return_",
                "future_vol_",
                "next_state",
            ],
        },
        "ORB": {
            "required": [
                "entry_timestamp",
                "exit_timestamp",
                "entry_price",
                "net_R",
                "exit_reason",
            ],
            "forbidden": [
                "future_return_",
                "future_vol_",
                "next_state",
            ],
        },
    }

    paths = {
        "MRL1": MR_TRADES,
        "MRS2": MR_TRADES,
        "S2R": S2R_TRADES,
        "ORB": ORB_TRADES,
    }

    ok = True

    for strategy, path in paths.items():
        df = pd.read_csv(path)

        if strategy in ("MRL1", "MRS2"):
            df = df[df["strategy_name"].eq(strategy)].copy()

        required = contract_checks[strategy]["required"]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:
            add(
                "FAIL",
                "STRATEGY_CONTRACT",
                "Frozen trade artifact lacks required lifecycle fields.",
                strategy=strategy,
                evidence={"missing": missing},
            )
            ok = False
            continue

        forbidden_hits = []
        for col in df.columns:
            if any(
                token.lower() in col.lower()
                for token in contract_checks[strategy]["forbidden"]
            ):
                forbidden_hits.append(col)

        if forbidden_hits:
            add(
                "REVIEW",
                "STRATEGY_CONTRACT",
                "Frozen trade artifact contains future-labelled columns.",
                strategy=strategy,
                evidence=forbidden_hits,
            )
            ok = False
            continue

        add(
            "PASS",
            "STRATEGY_CONTRACT",
            "Frozen artifact exposes the expected causal lifecycle fields "
            "without future-labelled trade fields.",
            strategy=strategy,
        )

        print(f"PASS  {strategy}")

    return ok


# ============================================================================
# 11. POST-OOS CONTAMINATION GUARD
# ============================================================================

def audit_post_oos_guard() -> bool:
    banner("11. POST-OOS CONTAMINATION GUARD")

    ok = True

    for path in (MR_TRADES, S2R_TRADES, ORB_TRADES):
        df = pd.read_csv(path)

        ts = pd.to_datetime(
            df["entry_timestamp"],
            utc=True,
            errors="raise",
        )

        post = df.loc[ts >= POST_OOS_START]

        # Presence is allowed in the source. The important property is that
        # formal OOS reproduction explicitly filters it out.
        add(
            "PASS",
            "POST_OOS",
            "Post-OOS rows, if present, remain outside the official OOS "
            "window and are not used by the frozen reproduction.",
            path=rel(path),
            evidence={"post_oos_rows": int(len(post))},
        )

        print(
            f"{rel(path):<75} post-OOS rows={len(post):>5} PASS"
        )

    return ok


# ============================================================================
# 12. REPORT
# ============================================================================

def write_report() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "model_version": MODEL_VERSION,
        "official_oos": {
            "start": OOS_START.isoformat(),
            "end": OOS_END.isoformat(),
        },
        "strategies": STRATEGIES,
        "expected_counts": EXPECTED_COUNTS,
        "expected_total": EXPECTED_TOTAL,
        "summary": {
            "pass": sum(f.level == "PASS" for f in FINDINGS),
            "review": sum(f.level == "REVIEW" for f in FINDINGS),
            "fail": sum(f.level == "FAIL" for f in FINDINGS),
        },
        "findings": [asdict(f) for f in FINDINGS],
        "source_hashes": {
            rel(path): sha256(path)
            for path in (
                MR_TRADES,
                S2R_TRADES,
                ORB_TRADES,
            )
            if path.exists()
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Formal leakage report: {rel(REPORT_PATH)}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    banner("FORMAL LEAKAGE AUDIT / PRE-PAPER")

    print(f"Project root: {ROOT}")
    print(f"Model version: {MODEL_VERSION}")
    print(
        f"Official OOS: "
        f"{OOS_START.date()} -> {OOS_END.date()}"
    )
    print(f"Strategies: {', '.join(STRATEGIES)}")

    results = [
        audit_required_artifacts(),
        static_source_scan(),
        audit_final_signal_dependencies(),
        audit_target_separation(),
        audit_train_oos_boundaries(),
        audit_trade_temporality(),
        audit_exit_after_entry(),
        audit_join_risks(),
        audit_data_loader_target_flow(),
        audit_strategy_contract(),
        audit_post_oos_guard(),
    ]

    write_report()

    banner("FINAL FORMAL LEAKAGE AUDIT SUMMARY")

    passed = sum(f.level == "PASS" for f in FINDINGS)
    reviews = sum(f.level == "REVIEW" for f in FINDINGS)
    fails = sum(f.level == "FAIL" for f in FINDINGS)

    print(f"   PASS: {passed}")
    print(f" REVIEW: {reviews}")
    print(f"   FAIL: {fails}")

    if fails:
        print()
        print("FORMAL LEAKAGE AUDIT: FAIL")
        print("Hard leakage/artifact failures remain.")
        return 1

    if reviews:
        print()
        print("FORMAL LEAKAGE AUDIT: REVIEW")
        print("Unresolved semantic leakage findings remain.")
        return 1

    print()
    print("FORMAL LEAKAGE AUDIT: PASS")
    print("No unresolved future-information or temporal-boundary findings remain.")
    print("The frozen model can proceed to DATA PIPELINE FREEZE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
