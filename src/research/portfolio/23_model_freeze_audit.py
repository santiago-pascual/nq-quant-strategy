"""
23_model_freeze_audit.py

Final pre-paper validation gate for the frozen NQ/MNQ quantitative system.

This audit:
1. Verifies required frozen artifacts exist and hashes them.
2. Reconstructs the official OOS trade set from the authoritative strategy files.
3. Verifies frozen OOS counts and frozen portfolio metrics.
4. Verifies post-OOS observations are outside the official OOS window.
5. Performs a static leakage scan with explicit, manually-audited SAFE exceptions.
6. Builds a reproducibility inventory.
7. Writes a machine-readable model freeze manifest.

IMPORTANT:
- This script does not optimize the model.
- It does not modify strategy logic.
- REVIEW is used only when a suspicious pattern has not been semantically classified.
- The seven iloc[-1] findings previously inspected manually are explicitly classified
  as SAFE below.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT


# =============================================================================
# FROZEN MODEL DEFINITION
# =============================================================================

MODEL_VERSION = "v1.3-full-system-validation"

OFFICIAL_OOS_START = pd.Timestamp("2020-06-23", tz="UTC")
OFFICIAL_OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="UTC")
POST_OOS_START = pd.Timestamp("2026-06-20", tz="UTC")

FINAL_STRATEGIES = (
    "MRL1",
    "S2R",
    "MRS2",
    "ORB",
)

EXPECTED_OOS_COUNTS = {
    "MRL1": 430,
    "S2R": 520,
    "MRS2": 863,
    "ORB": 1442,
}

EXPECTED_TOTAL_OOS = 3255

# Frozen official portfolio OOS metrics from 21_mr_orb_portfolio_analysis.py.
EXPECTED_OOS_METRICS = {
    "total_R": 289.661901,
    "expectancy_R": 0.088990,
    "profit_factor": 1.215771,
    "win_rate": 0.506605,
    "max_drawdown_R": -18.093472,
    "daily_sharpe": 2.047973,
    "daily_sortino": 3.924309,
}

# Verification tolerances only. These are not optimization targets.
METRIC_TOLERANCE = 1e-5


# =============================================================================
# REQUIRED / KEY ARTIFACTS
# =============================================================================

KEY_FILES = [
    Path("src/research/mean_reversion/research/08aa_modular_reproduction.py"),
    Path(
        "src/research/mean_reversion/results/"
        "research_08aa_modular_reproduction_trades.csv"
    ),
    Path("src/research/results/s2_extended/s2r_modular_authoritative_reproduction.csv"),
    Path("src/research/results/orb/orb_reconciliation_trades.csv"),
    Path("src/research/portfolio/22_full_system_funded_simulation.py"),
    Path("src/research/portfolio/22_mr_orb_portfolio_robustness.py"),
    Path(
        "src/research/results/portfolio/funded/full_system_funded_combine_results.csv"
    ),
    Path("src/research/results/portfolio/funded/full_system_funded_xfa_results.csv"),
]


# =============================================================================
# FINDING MODEL
# =============================================================================


@dataclass
class AuditFinding:
    level: str
    category: str
    message: str
    path: str = ""
    line: int | None = None


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def relpath(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def fail_if_missing(
    path: Path,
    findings: list[AuditFinding],
) -> bool:
    if not path.exists():
        findings.append(
            AuditFinding(
                level="FAIL",
                category="ARTIFACT",
                message="Required file is missing.",
                path=relpath(path),
            )
        )
        return False

    findings.append(
        AuditFinding(
            level="PASS",
            category="ARTIFACT",
            message="Required file exists.",
            path=relpath(path),
        )
    )

    return True


# =============================================================================
# TRADE FILE LOADING
# =============================================================================


def load_trade_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "entry_timestamp" not in df.columns:
        raise ValueError(f"{relpath(path)}: missing entry_timestamp")

    df["entry_timestamp"] = normalize_timestamp_series(df["entry_timestamp"])

    if df["entry_timestamp"].isna().any():
        raise ValueError(f"{relpath(path)}: invalid entry_timestamp values found")

    # Strategy-specific authoritative files may not carry strategy_name.
    if "strategy_name" not in df.columns:
        if "strategy" in df.columns:
            df["strategy_name"] = df["strategy"]
        else:
            df["strategy_name"] = ""

    if "r_multiple" not in df.columns:
        if "r" in df.columns:
            df["r_multiple"] = pd.to_numeric(
                df["r"],
                errors="coerce",
            )
        elif "net_R" in df.columns:
            df["r_multiple"] = pd.to_numeric(
                df["net_R"],
                errors="coerce",
            )
        else:
            raise ValueError(f"{relpath(path)}: missing r_multiple/r/net_R")

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    )

    if df["r_multiple"].isna().any():
        raise ValueError(f"{relpath(path)}: invalid R values found")

    return df


# =============================================================================
# 1. ARTIFACT AUDIT
# =============================================================================


def audit_artifacts(
    findings: list[AuditFinding],
) -> dict:
    banner("1. ARTIFACT AUDIT")

    hashes: dict = {}

    for relative_path in KEY_FILES:
        path = ROOT / relative_path

        if fail_if_missing(path, findings):
            hashes[relpath(path)] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }

    return hashes


# =============================================================================
# OOS SOURCE DEFINITIONS
# =============================================================================

OOS_SOURCES = {
    "MRL1": (
        ROOT / "src/research/mean_reversion/results/"
        "research_08aa_modular_reproduction_trades.csv"
    ),
    "MRS2": (
        ROOT / "src/research/mean_reversion/results/"
        "research_08aa_modular_reproduction_trades.csv"
    ),
    "S2R": (
        ROOT / "src/research/results/s2_extended/"
        "s2r_modular_authoritative_reproduction.csv"
    ),
    "ORB": (ROOT / "src/research/results/orb/orb_reconciliation_trades.csv"),
}


# =============================================================================
# 2. OOS / POST-OOS AUDIT
# =============================================================================


def audit_final_oos(
    findings: list[AuditFinding],
) -> tuple[dict, pd.DataFrame]:
    banner("2. FINAL OOS / POST-OOS AUDIT")

    loaded: dict[str, pd.DataFrame] = {}

    for strategy, path in OOS_SOURCES.items():
        if not path.exists():
            findings.append(
                AuditFinding(
                    level="FAIL",
                    category="OOS",
                    message="Trade source is missing.",
                    path=relpath(path),
                )
            )
            continue

        try:
            df = load_trade_csv(path)
        except Exception as exc:
            findings.append(
                AuditFinding(
                    level="FAIL",
                    category="OOS",
                    message=f"Could not read trade file: {exc}",
                    path=relpath(path),
                )
            )
            continue

        if strategy in {"MRL1", "MRS2"}:
            strategy_mask = df["strategy_name"].astype(str).str.upper().eq(strategy)
            df = df.loc[strategy_mask].copy()
        else:
            # S2R and ORB authoritative files are strategy-specific.
            df = df.copy()
            df["strategy_name"] = strategy

        loaded[strategy] = df

        oos_mask = (df["entry_timestamp"] >= OFFICIAL_OOS_START) & (
            df["entry_timestamp"] <= OFFICIAL_OOS_END
        )

        post_mask = df["entry_timestamp"] >= POST_OOS_START

        oos = df.loc[oos_mask].copy()
        post = df.loc[post_mask].copy()

        expected = EXPECTED_OOS_COUNTS[strategy]
        actual = len(oos)

        if actual == expected:
            findings.append(
                AuditFinding(
                    level="PASS",
                    category="OOS",
                    message=(f"{strategy}: official OOS count = {actual}."),
                    path=relpath(path),
                )
            )
        else:
            findings.append(
                AuditFinding(
                    level="FAIL",
                    category="OOS",
                    message=(
                        f"{strategy}: expected {expected} official "
                        f"OOS trades, found {actual}."
                    ),
                    path=relpath(path),
                )
            )

        # Verify the official OOS window has no timestamps outside it.
        if not oos.empty:
            if (
                oos["entry_timestamp"].min() >= OFFICIAL_OOS_START
                and oos["entry_timestamp"].max() <= OFFICIAL_OOS_END
            ):
                findings.append(
                    AuditFinding(
                        level="PASS",
                        category="OOS",
                        message=(
                            f"{strategy}: all official OOS entries are "
                            "inside the frozen OOS window."
                        ),
                        path=relpath(path),
                    )
                )
            else:
                findings.append(
                    AuditFinding(
                        level="FAIL",
                        category="OOS",
                        message=(
                            f"{strategy}: OOS entries cross the frozen OOS boundary."
                        ),
                        path=relpath(path),
                    )
                )

        if not post.empty:
            findings.append(
                AuditFinding(
                    level="PASS",
                    category="POST-OOS",
                    message=(
                        f"{strategy}: {len(post)} post-OOS observations "
                        "exist and are outside official OOS."
                    ),
                    path=relpath(path),
                )
            )

    if not loaded:
        return {}, pd.DataFrame()

    oos_frames = []

    for strategy, df in loaded.items():
        mask = (df["entry_timestamp"] >= OFFICIAL_OOS_START) & (
            df["entry_timestamp"] <= OFFICIAL_OOS_END
        )

        oos_frames.append(df.loc[mask].copy())

    combined = pd.concat(
        oos_frames,
        ignore_index=True,
    )

    total = len(combined)

    if total == EXPECTED_TOTAL_OOS:
        findings.append(
            AuditFinding(
                level="PASS",
                category="OOS",
                message=(f"Combined official OOS count = {total}."),
            )
        )
    else:
        findings.append(
            AuditFinding(
                level="FAIL",
                category="OOS",
                message=(
                    f"Expected {EXPECTED_TOTAL_OOS} combined OOS trades, found {total}."
                ),
            )
        )

    counts = {
        strategy: int(
            (combined["strategy_name"].astype(str).str.upper().eq(strategy)).sum()
        )
        for strategy in FINAL_STRATEGIES
    }

    for strategy, expected in EXPECTED_OOS_COUNTS.items():
        if counts[strategy] == expected:
            findings.append(
                AuditFinding(
                    level="PASS",
                    category="OOS",
                    message=(
                        f"Combined OOS attribution: {strategy} = {counts[strategy]}."
                    ),
                )
            )
        else:
            findings.append(
                AuditFinding(
                    level="FAIL",
                    category="OOS",
                    message=(
                        f"Combined OOS attribution: {strategy} expected "
                        f"{expected}, found {counts[strategy]}."
                    ),
                )
            )

    combined = combined.sort_values(
        ["entry_timestamp", "strategy_name"],
        kind="mergesort",
    ).reset_index(drop=True)

    return (
        {
            "total_oos": total,
            "strategy_counts": counts,
            "oos_start_actual": (
                combined["entry_timestamp"].min().isoformat()
                if not combined.empty
                else None
            ),
            "oos_end_actual": (
                combined["entry_timestamp"].max().isoformat()
                if not combined.empty
                else None
            ),
        },
        combined,
    )


# =============================================================================
# 2B. FROZEN OOS METRIC REPRODUCTION
# =============================================================================


def calculate_frozen_metrics(
    combined: pd.DataFrame,
) -> dict:
    if combined.empty:
        return {}

    r = combined["r_multiple"].astype(float)

    total_R = float(r.sum())
    expectancy_R = float(r.mean())

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = float((r > 0).mean())

    daily = (
        combined.assign(date=combined["entry_timestamp"].dt.floor("D"))
        .groupby("date")["r_multiple"]
        .sum()
        .sort_index()
    )

    equity = daily.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown_R = float(drawdown.min())

    daily_std = float(daily.std(ddof=1))

    if daily_std > 0:
        daily_sharpe = float(daily.mean() / daily_std * (252**0.5))
    else:
        daily_sharpe = float("nan")

    # IMPORTANT:
    # This exactly matches the frozen methodology used by
    # 21_mr_orb_portfolio_analysis.py:
    # sample standard deviation of negative daily observations only.
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

    return {
        "total_R": total_R,
        "expectancy_R": expectancy_R,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_drawdown_R": max_drawdown_R,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
    }


def audit_frozen_metrics(
    combined: pd.DataFrame,
    findings: list[AuditFinding],
) -> dict:
    if combined.empty:
        findings.append(
            AuditFinding(
                level="FAIL",
                category="OOS_METRIC",
                message="Cannot calculate frozen metrics: OOS is empty.",
            )
        )
        return {}

    actual = calculate_frozen_metrics(combined)

    for metric, expected in EXPECTED_OOS_METRICS.items():
        found = actual.get(metric, float("nan"))

        if pd.notna(found) and abs(found - expected) <= METRIC_TOLERANCE:
            findings.append(
                AuditFinding(
                    level="PASS",
                    category="OOS_METRIC",
                    message=(f"{metric}: expected {expected:.9f}, found {found:.9f}."),
                )
            )
        else:
            findings.append(
                AuditFinding(
                    level="FAIL",
                    category="OOS_METRIC",
                    message=(f"{metric}: expected {expected:.9f}, found {found:.9f}."),
                )
            )

    return actual


# =============================================================================
# 3. STATIC LEAKAGE AUDIT
# =============================================================================

LOOKAHEAD_PATTERNS = [
    (
        "shift(-)",
        re.compile(
            r"\.shift\s*\(\s*-\d+",
            re.IGNORECASE,
        ),
        "Negative shift can introduce future information.",
    ),
    (
        "rolling(center=True)",
        re.compile(
            r"rolling\s*\([^)]*center\s*=\s*True",
            re.IGNORECASE,
        ),
        "Centered rolling windows can include future observations.",
    ),
    (
        "bfill",
        re.compile(
            r"\.bfill\s*\(",
            re.IGNORECASE,
        ),
        "Backward fill can propagate future observations backward.",
    ),
    (
        "backfill",
        re.compile(
            r"\.backfill\s*\(",
            re.IGNORECASE,
        ),
        "Backfill can propagate future observations backward.",
    ),
    (
        "expanding()",
        re.compile(
            r"\.expanding\s*\(\s*\)",
            re.IGNORECASE,
        ),
        "Expanding calculations require causal-review.",
    ),
    (
        "iloc[-1]",
        re.compile(
            r"\.iloc\s*\[\s*-\d+\s*\]",
            re.IGNORECASE,
        ),
        "Negative indexing requires semantic time-series review.",
    ),
    (
        "forward merge_asof",
        re.compile(
            r"merge_asof\s*\("
            r"[^)]*direction\s*=\s*[\"']forward[\"']",
            re.IGNORECASE,
        ),
        "Forward merge_asof can match future observations.",
    ),
]


# Files/modules that are historical exploratory research rather than
# components of the frozen final signal-generation pipeline.
LEGACY_RESEARCH_EXCLUSIONS = (
    "session_momentum",
    "direction_",
    "xgboost_direction",
    "01_mean_reversion_exploration",
    "04_feature_interactions",
)


# Manually inspected iloc[-1] findings.
# These are explicit semantic classifications, not blanket suppression.
SAFE_ILOC_LAST_PATTERNS = {
    "src/research/orb/13_orb_baseline.py": {
        465: ("SAFE: final RTH bar used for explicit RTH_CLOSE liquidation."),
    },
    "src/research/orb/14_orb_execution_audit.py": {
        341: ("SAFE: final RTH bar used for explicit RTH_CLOSE audit liquidation."),
    },
    "src/research/orb/15_orb_reconciliation.py": {
        539: (
            "SAFE: final bar of the current RTH session used for "
            "the explicit RTH_CLOSE reconciliation exit."
        ),
    },
    "src/research/orb/19_orb_funded_simulation.py": {
        2149: ("SAFE: reporting-only last historical trade timestamp."),
    },
    "src/research/orb/20_orb_modular_reproduction.py": {
        552: ("SAFE: reporting-only last reproduced trade timestamp."),
    },
    "src/research/s2/s3_failure_path_analysis.py": {
        271: (
            "SAFE: selects the last row among duplicate rows at "
            "the same timestamp; does not advance time."
        ),
    },
    "src/research/mean_reversion/research/10_mr_portfolio_funded_simulation.py": {
        671: ("SAFE: reporting-only last historical event timestamp."),
    },
    "10_mr_portfolio_funded_simulation.py": {
        671: ("SAFE: reporting-only last historical event timestamp."),
    },
    "src/research/mean_reversion/research/12_mr_3_strategy_visual_report.py": {
        262: ("SAFE: reporting-only end-of-dataset timestamp."),
        703: (
            "SAFE: explicit END_OF_DATA liquidation of an "
            "already-open position in a visual/research report."
        ),
    },
    "src/research/portfolio/21_mr_orb_portfolio_analysis.py": {
        528: ("SAFE: reporting-only canonical market-data end timestamp."),
    },
    "src/research/portfolio/22_full_system_funded_simulation.py": {
        1989: (
            "SAFE: reporting-only last historical portfolio trade "
            "timestamp before the funded replay."
        ),
    },
}


def should_exclude_legacy(
    relative_path: str,
) -> bool:
    lower = relative_path.lower()

    return any(token in lower for token in LEGACY_RESEARCH_EXCLUSIONS)


def audit_static_leakage(
    findings: list[AuditFinding],
) -> dict:
    banner("3. STATIC LEAKAGE AUDIT")

    audit_path = Path(__file__).resolve()

    matches = []
    safe_matches = []

    # Scan the actual project source tree only. This deliberately excludes
    # root-level downloaded/copy artifacts such as 23_model_freeze_audit_FINAL.py.
    source_root = ROOT / "src"

    for path in source_root.rglob("*.py"):
        if path.resolve() == audit_path:
            # Do not scan the auditor's own regex definitions.
            continue

        if path.name.endswith("_backup.py"):
            # Local backup copies are not model source artifacts and can
            # otherwise duplicate already-classified findings.
            continue

        if any(
            part
            in {
                ".venv",
                ".git",
                "__pycache__",
                "site-packages",
            }
            for part in path.parts
        ):
            continue

        relative_path = relpath(path)

        if should_exclude_legacy(relative_path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        lines = text.splitlines()

        for label, pattern, explanation in LOOKAHEAD_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1

                # ---------------------------------------------------------
                # Explicitly audited safe iloc[-1] cases.
                # ---------------------------------------------------------
                if label == "iloc[-1]":
                    safe_reason = SAFE_ILOC_LAST_PATTERNS.get(relative_path, {}).get(
                        line_no
                    )

                    if safe_reason:
                        safe_matches.append(
                            {
                                "path": relative_path,
                                "line": line_no,
                                "pattern": label,
                                "reason": safe_reason,
                                "snippet": (
                                    lines[line_no - 1].strip()[:240]
                                    if 0 < line_no <= len(lines)
                                    else ""
                                ),
                            }
                        )
                        continue

                # ---------------------------------------------------------
                # Explicitly audited future-target / transition-analysis
                # cases. These are labels/descriptive diagnostics, not
                # final signal inputs.
                # ---------------------------------------------------------
                safe_reason = None

                if relative_path == "src/targets.py":
                    safe_reason = (
                        "SAFE: future target construction for supervised/"
                        "diagnostic labeling; not used as a final signal."
                    )

                elif (
                    relative_path == "src/research/regime/hmm_transition_analysis.py"
                    and label == "shift(-)"
                ):
                    safe_reason = (
                        "SAFE: next-state transition analysis only; the "
                        "future state is descriptive and not a trading input."
                    )

                if safe_reason:
                    safe_matches.append(
                        {
                            "path": relative_path,
                            "line": line_no,
                            "pattern": label,
                            "reason": safe_reason,
                            "snippet": (
                                lines[line_no - 1].strip()[:240]
                                if 0 < line_no <= len(lines)
                                else ""
                            ),
                        }
                    )
                    continue

                matches.append(
                    {
                        "path": relative_path,
                        "line": line_no,
                        "pattern": label,
                        "explanation": explanation,
                        "snippet": (
                            lines[line_no - 1].strip()[:240]
                            if 0 < line_no <= len(lines)
                            else ""
                        ),
                    }
                )

    # -------------------------------------------------------------
    # Emit explicit SAFE classifications.
    # -------------------------------------------------------------

    for item in safe_matches:
        findings.append(
            AuditFinding(
                level="PASS",
                category="LEAKAGE",
                message=item["reason"],
                path=item["path"],
                line=item["line"],
            )
        )

    # -------------------------------------------------------------
    # Emit unresolved static findings.
    # -------------------------------------------------------------

    for item in matches:
        findings.append(
            AuditFinding(
                level="REVIEW",
                category="LEAKAGE",
                message=(
                    f"{item['pattern']} at line {item['line']}: {item['explanation']}"
                ),
                path=item["path"],
                line=item["line"],
            )
        )

    if not matches:
        print("\nNo unresolved static look-ahead patterns detected.")
    else:
        print(
            f"\nFound {len(matches)} unresolved static patterns "
            "requiring manual review."
        )

    print(f"Explicitly classified SAFE static cases: {len(safe_matches)}")

    return {
        "unresolved_matches": matches,
        "safe_matches": safe_matches,
        "legacy_exclusions": list(LEGACY_RESEARCH_EXCLUSIONS),
    }


# =============================================================================
# 4. REPRODUCIBILITY INVENTORY
# =============================================================================


def audit_reproducibility_inventory(
    findings: list[AuditFinding],
) -> dict:
    banner("4. REPRODUCIBILITY INVENTORY")

    candidates = []

    for path in ROOT.rglob("*.py"):
        if any(
            part
            in {
                ".venv",
                ".git",
                "__pycache__",
                "site-packages",
            }
            for part in path.parts
        ):
            continue

        name = path.name.lower()

        if any(
            token in name
            for token in (
                "reproduction",
                "reconcile",
                "validation",
                "portfolio",
                "robustness",
                "funded",
            )
        ):
            candidates.append(relpath(path))

    candidates = sorted(set(candidates))

    if candidates:
        findings.append(
            AuditFinding(
                level="PASS",
                category="REPRO",
                message=(
                    f"Found {len(candidates)} candidate "
                    "research/validation entry points."
                ),
            )
        )
    else:
        findings.append(
            AuditFinding(
                level="FAIL",
                category="REPRO",
                message=("No research/validation entry points were discovered."),
            )
        )

    for candidate in candidates:
        print(f"  - {candidate}")

    return {
        "candidate_scripts": candidates,
    }


# =============================================================================
# 5. MODEL FREEZE MANIFEST
# =============================================================================


def write_manifest(
    hashes: dict,
    oos: dict,
    metrics: dict,
    leakage: dict,
    repro: dict,
    findings: list[AuditFinding],
) -> Path:
    banner("5. WRITING MODEL FREEZE MANIFEST")

    payload = {
        "model_version": MODEL_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "official_oos": {
            "start": OFFICIAL_OOS_START.isoformat(),
            "end": OFFICIAL_OOS_END.isoformat(),
        },
        "post_oos_start": POST_OOS_START.isoformat(),
        "final_strategies": list(FINAL_STRATEGIES),
        "expected_oos_counts": EXPECTED_OOS_COUNTS,
        "expected_total_oos": EXPECTED_TOTAL_OOS,
        "expected_oos_metrics": EXPECTED_OOS_METRICS,
        "metric_tolerance": METRIC_TOLERANCE,
        "key_file_hashes": hashes,
        "oos_audit": oos,
        "reproduced_oos_metrics": metrics,
        "static_leakage_review": leakage,
        "reproducibility_inventory": repro,
        "findings": [asdict(item) for item in findings],
    }

    output = ROOT / "src/research/results/portfolio/model_freeze_manifest.json"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Manifest written to: {relpath(output)}")

    return output


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    banner("MODEL FREEZE / PRE-PAPER AUDIT")

    print(f"Project root: {ROOT}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Official OOS: {OFFICIAL_OOS_START.date()} -> {OFFICIAL_OOS_END.date()}")
    print(f"Strategies: {', '.join(FINAL_STRATEGIES)}")

    findings: list[AuditFinding] = []

    # -------------------------------------------------------------
    # Artifact audit
    # -------------------------------------------------------------

    hashes = audit_artifacts(findings)

    # -------------------------------------------------------------
    # OOS audit
    # -------------------------------------------------------------

    oos, combined = audit_final_oos(findings)

    # -------------------------------------------------------------
    # Frozen metric audit
    # -------------------------------------------------------------

    metrics = audit_frozen_metrics(
        combined,
        findings,
    )

    # -------------------------------------------------------------
    # Static leakage audit
    # -------------------------------------------------------------

    leakage = audit_static_leakage(findings)

    # -------------------------------------------------------------
    # Reproducibility inventory
    # -------------------------------------------------------------

    repro = audit_reproducibility_inventory(findings)

    # -------------------------------------------------------------
    # Write manifest
    # -------------------------------------------------------------

    manifest = write_manifest(
        hashes=hashes,
        oos=oos,
        metrics=metrics,
        leakage=leakage,
        repro=repro,
        findings=findings,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    banner("FINAL AUDIT SUMMARY")

    counts = {
        "PASS": 0,
        "REVIEW": 0,
        "FAIL": 0,
    }

    for finding in findings:
        counts[finding.level] = counts.get(finding.level, 0) + 1

    for level in (
        "PASS",
        "REVIEW",
        "FAIL",
    ):
        print(f"{level:>7}: {counts.get(level, 0)}")

    if counts["FAIL"] > 0:
        print("\nMODEL FREEZE GATE: FAIL")
        print("Resolve all FAIL findings before freezing the model.")
        print(f"Freeze manifest: {relpath(manifest)}")
        return 1

    if counts["REVIEW"] > 0:
        print("\nMODEL FREEZE GATE: PASS WITH MANUAL REVIEW ITEMS")
        print("No hard artifact/OOS/metric failure was detected.")
        print("Remaining REVIEW findings require manual inspection.")
    else:
        print("\nMODEL FREEZE GATE: PASS")
        print("All artifact, OOS, frozen-metric, and static leakage checks passed.")
        print("No unresolved leakage REVIEW items remain.")

    print(f"Freeze manifest: {relpath(manifest)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
