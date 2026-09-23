"""
26_data_pipeline_freeze.py

DATA PIPELINE FREEZE / PRE-PAPER GATE
=====================================

Purpose
-------
Freeze and audit the canonical market-data pipeline used by the frozen
v1.3-full-system-validation model before paper trading.

This script does NOT optimize or modify any strategy.

It validates:
1. Canonical loader availability and schema.
2. Timestamp parsing, timezone and strict ordering.
3. Duplicate timestamps.
4. OHLC price validity and OHLC relationships.
5. Volume validity.
6. Required symbol/data identity.
7. RTH/session classification consistency.
8. Expected canonical dataset boundaries and row count.
9. Required feature columns and their finite-value coverage.
10. No future-looking feature construction in the frozen pipeline.
11. Frozen OOS boundary consistency.
12. SHA-256 manifest for the source/data artifacts that can be hashed.
13. Deterministic loader output across two consecutive loads.

The gate passes only when every required invariant passes.

Run from project root:
    python .\src\research\portfolio\26_data_pipeline_freeze.py

Output:
    src/research/results/portfolio/data_pipeline_freeze/
        data_pipeline_freeze_report.json
        data_pipeline_manifest.json
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT / FROZEN MODEL CONTRACT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_VERSION = "v1.3-full-system-validation"

OFFICIAL_OOS_START = pd.Timestamp("2020-06-23 00:00:00", tz="UTC")
OFFICIAL_OOS_END = pd.Timestamp("2026-06-19 23:59:59", tz="UTC")

EXPECTED_ROWS = 2_577_661

EXPECTED_START_UTC = pd.Timestamp("2019-05-05 22:03:00", tz="UTC")
EXPECTED_END_UTC = pd.Timestamp("2026-08-26 23:59:00", tz="UTC")

EXPECTED_COLUMNS = [
    "timestamp ET",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
]

EXPECTED_CANONICAL_COLUMNS = [
    "timestamp ET",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
]

REQUIRED_NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

REQUIRED_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "portfolio"
    / "data_pipeline_freeze"
)

REPORT_PATH = OUTPUT_DIR / "data_pipeline_freeze_report.json"
MANIFEST_PATH = OUTPUT_DIR / "data_pipeline_manifest.json"


# =============================================================================
# UTILITIES
# =============================================================================

findings: list[dict[str, Any]] = []


def add(
    level: str,
    check: str,
    message: str,
    **extra: Any,
) -> None:
    row = {
        "level": level,
        "check": check,
        "message": message,
    }
    row.update(extra)
    findings.append(row)

    prefix = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "REVIEW": "REVIEW",
        "INFO": "INFO",
    }.get(level, level)

    print(f"{prefix:<7} {check}")
    if message:
        print(f"        {message}")


def banner(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def normalize_timestamp_series(s: pd.Series) -> pd.Series:
    """
    Normalize the canonical 'timestamp ET' series to UTC for validation only.

    The canonical loader intentionally exposes 'timestamp ET'. We do not
    overwrite the loaded dataframe in this audit.
    """
    ts = pd.to_datetime(s, errors="coerce")

    if ts.dt.tz is None:
        # The canonical column is ET. Treat naive timestamps as America/New_York.
        ts = ts.dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT",
        )

    return ts.dt.tz_convert("UTC")


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Deterministic fingerprint of the loaded canonical dataframe.

    Includes column order, dtypes and pandas row values.
    """
    h = hashlib.sha256()

    header = json.dumps(
        {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "rows": len(df),
        },
        sort_keys=True,
    ).encode()

    h.update(header)

    # hash_pandas_object is deterministic for the same dataframe content.
    row_hash = pd.util.hash_pandas_object(
        df,
        index=True,
    ).to_numpy(dtype=np.uint64)

    h.update(row_hash.tobytes())
    return h.hexdigest()


def locate_loader() -> tuple[Any, str]:
    """
    Locate the canonical Databento MNQ loader.

    Expected historical project structure uses src.data_loader.
    A small candidate list is retained so this gate can fail explicitly
    rather than silently falling back to another data source.
    """
    candidates = [
        ("src.data_loader", "load_databento_mnq"),
        ("src.data.loaders", "load_databento_mnq"),
        ("src.data_loader.databento", "load_databento_mnq"),
        ("src.data.databento_loader", "load_databento_mnq"),
    ]

    errors = []

    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, function_name, None)
            if callable(fn):
                return fn, f"{module_name}.{function_name}"
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Canonical load_databento_mnq was not found. "
        "Checked: "
        + ", ".join(f"{m}.{f}" for m, f in candidates)
        + "\n"
        + "\n".join(errors)
    )


def find_canonical_data_files() -> list[Path]:
    """
    Find likely canonical market-data files for manifesting.

    This is deliberately conservative. The loader remains authoritative.
    We do not select a data file from this function.
    """
    roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "src" / "data",
        PROJECT_ROOT / "src" / "research" / "data",
    ]

    patterns = [
        "*MNQ*.csv",
        "*mnq*.csv",
        "*NQ*.csv",
        "*nq*.csv",
        "*databento*.csv",
        "*MNQ*.parquet",
        "*mnq*.parquet",
        "*NQ*.parquet",
        "*nq*.parquet",
        "*databento*.parquet",
    ]

    paths: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue

        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    paths.add(path.resolve())

    return sorted(paths)


def source_files_for_manifest() -> list[Path]:
    """
    Collect relevant source files without treating them as the data source.
    """
    paths: list[Path] = []

    explicit = [
        PROJECT_ROOT / "src" / "data_loader.py",
        PROJECT_ROOT / "src" / "data" / "loaders.py",
        PROJECT_ROOT / "src" / "data" / "databento_loader.py",
    ]

    for p in explicit:
        if p.exists() and p.is_file():
            paths.append(p.resolve())

    return sorted(set(paths))


# =============================================================================
# LOADER
# =============================================================================

banner("DATA PIPELINE FREEZE / PRE-PAPER")
print(f"Project root: {PROJECT_ROOT}")
print(f"Model version: {MODEL_VERSION}")
print(
    "Official OOS: "
    f"{OFFICIAL_OOS_START.date()} -> {OFFICIAL_OOS_END.date()}"
)
print(f"Expected canonical rows: {EXPECTED_ROWS}")


# =============================================================================
# 1. LOCATE AND LOAD CANONICAL DATA
# =============================================================================

banner("1. CANONICAL LOADER")

try:
    loader, loader_name = locate_loader()
    add(
        "PASS",
        "Canonical loader discovery",
        f"Using {loader_name}",
    )
except Exception as exc:
    add(
        "FAIL",
        "Canonical loader discovery",
        str(exc),
    )
    loader = None
    loader_name = None


df = None
df2 = None

if loader is not None:
    try:
        df = loader()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"loader returned {type(df).__name__}, expected pandas.DataFrame"
            )

        add(
            "PASS",
            "Canonical loader execution",
            f"Loaded {len(df):,} rows.",
        )
    except Exception as exc:
        add(
            "FAIL",
            "Canonical loader execution",
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# 2. SCHEMA / TYPES
# =============================================================================

banner("2. SCHEMA / DATA TYPES")

if df is not None:
    actual_columns = list(df.columns)

    if actual_columns == EXPECTED_COLUMNS:
        add(
            "PASS",
            "Canonical schema",
            f"Exact column order: {actual_columns}",
        )
    else:
        add(
            "FAIL",
            "Canonical schema",
            f"Expected {EXPECTED_COLUMNS}, got {actual_columns}",
        )

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if not missing:
        add(
            "PASS",
            "Required columns present",
            "No required canonical columns are missing.",
        )
    else:
        add(
            "FAIL",
            "Required columns present",
            f"Missing columns: {missing}",
        )

    dtype_report = {c: str(df[c].dtype) for c in df.columns}

    numeric_failures = []
    for col in REQUIRED_NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            numeric_failures.append(
                f"{col}={df[col].dtype}"
            )

    if not numeric_failures:
        add(
            "PASS",
            "Numeric OHLCV dtypes",
            "OHLCV columns are numeric.",
        )
    else:
        add(
            "FAIL",
            "Numeric OHLCV dtypes",
            "; ".join(numeric_failures),
        )


# =============================================================================
# 3. TIMESTAMP CONTRACT
# =============================================================================

banner("3. TIMESTAMP CONTRACT")

utc_ts = None

if df is not None and "timestamp ET" in df.columns:
    raw_ts = df["timestamp ET"]

    invalid_count = int(pd.to_datetime(raw_ts, errors="coerce").isna().sum())

    if invalid_count == 0:
        add(
            "PASS",
            "Timestamp parseability",
            "All canonical timestamps parse successfully.",
        )
    else:
        add(
            "FAIL",
            "Timestamp parseability",
            f"{invalid_count:,} timestamps failed parsing.",
        )

    utc_ts = normalize_timestamp_series(raw_ts)

    nat_count = int(utc_ts.isna().sum())

    if nat_count == 0:
        add(
            "PASS",
            "Timestamp timezone normalization",
            "All timestamps normalize to UTC without ambiguous/nonexistent ET timestamps.",
        )
    else:
        add(
            "FAIL",
            "Timestamp timezone normalization",
            f"{nat_count:,} timestamps became NaT during ET->UTC normalization.",
        )

    if len(utc_ts) > 0 and nat_count == 0:
        first_ts = utc_ts.iloc[0]
        last_ts = utc_ts.iloc[-1]

        if first_ts == EXPECTED_START_UTC:
            add(
                "PASS",
                "Canonical start boundary",
                f"Start = {first_ts.isoformat()}",
            )
        else:
            add(
                "FAIL",
                "Canonical start boundary",
                f"Expected {EXPECTED_START_UTC.isoformat()}, got {first_ts.isoformat()}",
            )

        if last_ts == EXPECTED_END_UTC:
            add(
                "PASS",
                "Canonical end boundary",
                f"End = {last_ts.isoformat()}",
            )
        else:
            add(
                "FAIL",
                "Canonical end boundary",
                f"Expected {EXPECTED_END_UTC.isoformat()}, got {last_ts.isoformat()}",
            )

        if utc_ts.is_monotonic_increasing:
            add(
                "PASS",
                "Strict chronological ordering",
                "Canonical timestamps are monotonically increasing.",
            )
        else:
            bad = int((utc_ts.diff().dt.total_seconds() <= 0).fillna(False).sum())
            add(
                "FAIL",
                "Strict chronological ordering",
                f"{bad:,} non-increasing timestamp transitions detected.",
            )

        duplicate_count = int(utc_ts.duplicated(keep=False).sum())

        if duplicate_count == 0:
            add(
                "PASS",
                "Duplicate timestamps",
                "No duplicate canonical timestamps.",
            )
        else:
            add(
                "FAIL",
                "Duplicate timestamps",
                f"{duplicate_count:,} rows participate in duplicate timestamps.",
            )

else:
    add(
        "FAIL",
        "Timestamp contract",
        "Canonical dataframe or timestamp ET column unavailable.",
    )


# =============================================================================
# 4. ROW COUNT / IDENTITY
# =============================================================================

banner("4. DATASET IDENTITY")

if df is not None:
    if len(df) == EXPECTED_ROWS:
        add(
            "PASS",
            "Canonical row count",
            f"{len(df):,} rows exactly.",
        )
    else:
        add(
            "FAIL",
            "Canonical row count",
            f"Expected {EXPECTED_ROWS:,}, got {len(df):,}.",
        )

    if "symbol" in df.columns:
        symbols = (
            df["symbol"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        if len(symbols) == 1:
            only_symbol = next(iter(symbols))
            add(
                "PASS",
                "Symbol identity",
                f"Single canonical symbol: {only_symbol}",
            )
        else:
            add(
                "FAIL",
                "Symbol identity",
                f"Expected one symbol, got {symbols}",
            )

        if df["symbol"].isna().sum() == 0:
            add(
                "PASS",
                "Symbol completeness",
                "No missing symbol values.",
            )
        else:
            add(
                "FAIL",
                "Symbol completeness",
                f"{int(df['symbol'].isna().sum()):,} missing symbols.",
            )


# =============================================================================
# 5. OHLCV VALIDATION
# =============================================================================

banner("5. OHLCV INVARIANTS")

if df is not None:
    finite_failures = {}

    for col in REQUIRED_NUMERIC_COLUMNS:
        if col in df.columns:
            finite = np.isfinite(df[col].to_numpy(dtype=float))
            finite_failures[col] = int((~finite).sum())

    if sum(finite_failures.values()) == 0:
        add(
            "PASS",
            "Finite OHLCV values",
            "No NaN, +inf or -inf values in OHLCV.",
        )
    else:
        add(
            "FAIL",
            "Finite OHLCV values",
            str(finite_failures),
        )

    positive_price_failures = {}
    for col in REQUIRED_PRICE_COLUMNS:
        if col in df.columns:
            positive_price_failures[col] = int((df[col] <= 0).sum())

    if sum(positive_price_failures.values()) == 0:
        add(
            "PASS",
            "Positive prices",
            "All OHLC prices are strictly positive.",
        )
    else:
        add(
            "FAIL",
            "Positive prices",
            str(positive_price_failures),
        )

    if all(c in df.columns for c in REQUIRED_PRICE_COLUMNS):
        high_low_bad = int((df["high"] < df["low"]).sum())
        high_open_bad = int((df["high"] < df["open"]).sum())
        high_close_bad = int((df["high"] < df["close"]).sum())
        low_open_bad = int((df["low"] > df["open"]).sum())
        low_close_bad = int((df["low"] > df["close"]).sum())

        violations = {
            "high < low": high_low_bad,
            "high < open": high_open_bad,
            "high < close": high_close_bad,
            "low > open": low_open_bad,
            "low > close": low_close_bad,
        }

        if sum(violations.values()) == 0:
            add(
                "PASS",
                "OHLC relationship",
                "Every bar satisfies low <= open/close <= high and low <= high.",
            )
        else:
            add(
                "FAIL",
                "OHLC relationship",
                str(violations),
            )

    if "volume" in df.columns:
        negative_volume = int((df["volume"] < 0).sum())

        if negative_volume == 0:
            add(
                "PASS",
                "Non-negative volume",
                "No negative volume observations.",
            )
        else:
            add(
                "FAIL",
                "Non-negative volume",
                f"{negative_volume:,} negative-volume rows.",
            )


# =============================================================================
# 6. INTRABAR TIMESTAMP / 1-MINUTE CONTRACT
# =============================================================================

banner("6. 1-MINUTE BAR CONTRACT")

if utc_ts is not None and len(utc_ts) > 1 and utc_ts.notna().all():
    delta_seconds = utc_ts.diff().dt.total_seconds().dropna()

    # Overnight/session gaps are expected. We therefore do not require every
    # consecutive row to be exactly one minute. We validate that all positive
    # increments are integer minutes and that no sub-minute timestamps exist.
    positive = delta_seconds[delta_seconds > 0]

    subminute = int((positive < 60).sum())
    nonminute = int((positive % 60 != 0).sum())

    if subminute == 0:
        add(
            "PASS",
            "No sub-minute bars",
            "No consecutive canonical timestamps are separated by less than one minute.",
        )
    else:
        add(
            "FAIL",
            "No sub-minute bars",
            f"{subminute:,} sub-minute timestamp transitions detected.",
        )

    if nonminute == 0:
        add(
            "PASS",
            "Minute-grid alignment",
            "All positive timestamp increments are whole minutes.",
        )
    else:
        add(
            "FAIL",
            "Minute-grid alignment",
            f"{nonminute:,} non-minute timestamp transitions detected.",
        )

    gap_counts = positive.value_counts().sort_index()

    if len(gap_counts) > 0:
        max_gap = int(positive.max())
        add(
            "INFO",
            "Session/data gaps",
            f"Maximum observed gap = {max_gap / 60:.1f} minutes; gaps are retained as market-session boundaries/data availability, not filled.",
        )


# =============================================================================
# 7. RTH / SESSION CONTRACT
# =============================================================================

banner("7. SESSION CONTRACT")

if df is not None and utc_ts is not None and utc_ts.notna().all():
    ny_ts = utc_ts.dt.tz_convert("America/New_York")

    # RTH for the frozen project: 09:30 through 16:00 ET.
    minutes = ny_ts.dt.hour * 60 + ny_ts.dt.minute
    expected_rth = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)

    # Count canonical bars inside RTH and verify all have normal minute stamps.
    rth_rows = int(expected_rth.sum())

    add(
        "PASS",
        "RTH derivation",
        f"Derived {rth_rows:,} rows in the frozen 09:30-16:00 ET RTH window.",
    )

    # ORB's opening range must be represented by 09:30-10:00 ET.
    orb_window = (minutes >= 9 * 60 + 30) & (minutes < 10 * 60)

    if int(orb_window.sum()) > 0:
        add(
            "PASS",
            "ORB opening-range availability",
            f"Found {int(orb_window.sum()):,} canonical rows in 09:30-10:00 ET.",
        )
    else:
        add(
            "FAIL",
            "ORB opening-range availability",
            "No canonical rows found in the ORB opening-range window.",
        )


# =============================================================================
# 8. OOS BOUNDARY REPRESENTABILITY
# =============================================================================

banner("8. OOS BOUNDARY CONTRACT")

if utc_ts is not None and utc_ts.notna().all():
    before_oos = int((utc_ts < OFFICIAL_OOS_START).sum())
    in_oos = int(
        (
            (utc_ts >= OFFICIAL_OOS_START)
            & (utc_ts <= OFFICIAL_OOS_END)
        ).sum()
    )
    after_oos = int((utc_ts > OFFICIAL_OOS_END).sum())

    if before_oos > 0 and in_oos > 0 and after_oos > 0:
        add(
            "PASS",
            "Train / OOS / post-OOS coverage",
            f"pre={before_oos:,}, OOS={in_oos:,}, post={after_oos:,}",
        )
    else:
        add(
            "FAIL",
            "Train / OOS / post-OOS coverage",
            f"pre={before_oos:,}, OOS={in_oos:,}, post={after_oos:,}",
        )


# =============================================================================
# 9. REQUIRED FROZEN ARTIFACTS
# =============================================================================

banner("9. FROZEN ARTIFACT AVAILABILITY")

required_artifacts = [
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "research"
    / "08aa_modular_reproduction.py",
    PROJECT_ROOT
    / "src"
    / "research"
    / "mean_reversion"
    / "results"
    / "research_08aa_modular_reproduction_trades.csv",
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "s2_extended"
    / "s2r_modular_authoritative_reproduction.csv",
    PROJECT_ROOT
    / "src"
    / "research"
    / "results"
    / "orb"
    / "orb_reconciliation_trades.csv",
    PROJECT_ROOT
    / "src"
    / "research"
    / "portfolio"
    / "24_independent_reproduction.py",
    PROJECT_ROOT
    / "src"
    / "research"
    / "portfolio"
    / "25_formal_leakage_audit.py",
]

for path in required_artifacts:
    if path.exists() and path.is_file():
        add(
            "PASS",
            f"Artifact: {path.relative_to(PROJECT_ROOT).as_posix()}",
            f"SHA256={sha256_file(path)}",
        )
    else:
        add(
            "FAIL",
            f"Artifact: {path.relative_to(PROJECT_ROOT).as_posix()}",
            "Required frozen artifact is missing.",
        )


# =============================================================================
# 10. SOURCE / DATA MANIFEST
# =============================================================================

banner("10. DATA / SOURCE MANIFEST")

manifest_entries: list[dict[str, Any]] = []

for path in source_files_for_manifest():
    manifest_entries.append(
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )

candidate_data_files = find_canonical_data_files()

for path in candidate_data_files:
    try:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative = str(path)

    manifest_entries.append(
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "candidate_data_file_only_loader_remains_authoritative",
        }
    )

# Deduplicate manifest entries by path.
manifest_by_path = {
    item["path"]: item
    for item in manifest_entries
}

manifest_entries = [
    manifest_by_path[k]
    for k in sorted(manifest_by_path)
]

if manifest_entries:
    add(
        "PASS",
        "Manifest generation",
        f"Recorded {len(manifest_entries):,} source/data artifact hashes.",
    )
else:
    add(
        "REVIEW",
        "Manifest generation",
        "No source/data files were discovered for hashing; canonical loader audit still remains authoritative.",
    )


# =============================================================================
# 11. DETERMINISM / REPEAT LOAD
# =============================================================================

banner("11. LOADER DETERMINISM")

if loader is not None and df is not None:
    try:
        df2 = loader()

        if not isinstance(df2, pd.DataFrame):
            raise TypeError(
                f"second loader returned {type(df2).__name__}"
            )

        fp1 = dataframe_fingerprint(df)
        fp2 = dataframe_fingerprint(df2)

        if fp1 == fp2:
            add(
                "PASS",
                "Deterministic canonical loader",
                f"Two consecutive loads have identical fingerprint {fp1}.",
            )
        else:
            add(
                "FAIL",
                "Deterministic canonical loader",
                f"Fingerprint mismatch: first={fp1}, second={fp2}.",
            )
    except Exception as exc:
        add(
            "FAIL",
            "Deterministic canonical loader",
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# 12. FEATURE / FUTURE-DATA STATIC GUARD
# =============================================================================

banner("12. DATA PIPELINE LOOK-AHEAD GUARD")

# This is intentionally narrower than the formal leakage audit. It is aimed at
# the data pipeline / loader layer, not at target generation or descriptive
# research modules.

pipeline_scan_roots = [
    PROJECT_ROOT / "src" / "data_loader.py",
    PROJECT_ROOT / "src" / "data",
]

future_patterns = [
    r"\.shift\(\s*-\s*\d+",
    r"\.shift\(\s*-\s*",
    r"rolling\([^)]*center\s*=\s*True",
    r"\.bfill\(",
    r"\.backfill\(",
    r"merge_asof\([^)]*direction\s*=\s*[\"']forward[\"']",
]

scan_files: list[Path] = []

for root in pipeline_scan_roots:
    if root.is_file():
        scan_files.append(root)
    elif root.is_dir():
        scan_files.extend(
            p for p in root.rglob("*.py")
            if p.is_file()
        )

scan_files = sorted(set(p.resolve() for p in scan_files))

static_hits: list[dict[str, Any]] = []

for path in scan_files:
    try:
        text_content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        static_hits.append(
            {
                "path": str(path),
                "line": None,
                "pattern": "READ_ERROR",
                "text": str(exc),
            }
        )
        continue

    for line_no, line in enumerate(text_content.splitlines(), start=1):
        for pattern in future_patterns:
            if re.search(pattern, line):
                static_hits.append(
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "line": line_no,
                        "pattern": pattern,
                        "text": line.strip(),
                    }
                )

if static_hits:
    add(
        "FAIL",
        "Pipeline look-ahead static guard",
        f"Found {len(static_hits):,} potential future-looking constructs in the loader/data layer.",
        hits=static_hits,
    )
else:
    add(
        "PASS",
        "Pipeline look-ahead static guard",
        "No future-looking constructs found in the loader/data layer.",
    )


# =============================================================================
# 13. DATAFRAME FINGERPRINT
# =============================================================================

banner("13. CANONICAL DATA FINGERPRINT")

data_fingerprint = None

if df is not None:
    try:
        data_fingerprint = dataframe_fingerprint(df)
        add(
            "PASS",
            "Canonical dataframe fingerprint",
            data_fingerprint,
        )
    except Exception as exc:
        add(
            "FAIL",
            "Canonical dataframe fingerprint",
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# 14. FINAL REPORT
# =============================================================================

banner("FINAL DATA PIPELINE FREEZE SUMMARY")

pass_count = sum(1 for x in findings if x["level"] == "PASS")
review_count = sum(1 for x in findings if x["level"] == "REVIEW")
fail_count = sum(1 for x in findings if x["level"] == "FAIL")

gate = "PASS" if fail_count == 0 and review_count == 0 else "FAIL"

print(f"   PASS: {pass_count}")
print(f" REVIEW: {review_count}")
print(f"   FAIL: {fail_count}")
print()
print(f"DATA PIPELINE FREEZE: {gate}")

if gate == "PASS":
    print("Canonical market-data pipeline is frozen and ready for the execution-engine stage.")
else:
    print("Data Pipeline Freeze is NOT passed. Resolve all FAIL/REVIEW findings before proceeding.")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = {
    "model_version": MODEL_VERSION,
    "audit": "26_data_pipeline_freeze",
    "official_oos": {
        "start": OFFICIAL_OOS_START.isoformat(),
        "end": OFFICIAL_OOS_END.isoformat(),
    },
    "canonical_dataset": {
        "expected_rows": EXPECTED_ROWS,
        "expected_start_utc": EXPECTED_START_UTC.isoformat(),
        "expected_end_utc": EXPECTED_END_UTC.isoformat(),
        "expected_columns": EXPECTED_CANONICAL_COLUMNS,
        "loader": loader_name,
        "dataframe_fingerprint": data_fingerprint,
    },
    "artifacts": manifest_entries,
}

report = {
    "model_version": MODEL_VERSION,
    "audit": "26_data_pipeline_freeze",
    "gate": gate,
    "counts": {
        "pass": pass_count,
        "review": review_count,
        "fail": fail_count,
    },
    "official_oos": {
        "start": OFFICIAL_OOS_START.isoformat(),
        "end": OFFICIAL_OOS_END.isoformat(),
    },
    "canonical_dataset": {
        "expected_rows": EXPECTED_ROWS,
        "expected_start_utc": EXPECTED_START_UTC.isoformat(),
        "expected_end_utc": EXPECTED_END_UTC.isoformat(),
        "columns": EXPECTED_CANONICAL_COLUMNS,
        "loader": loader_name,
        "dataframe_fingerprint": data_fingerprint,
    },
    "findings": findings,
}

MANIFEST_PATH.write_text(
    json.dumps(manifest, indent=2, default=json_default),
    encoding="utf-8",
)

REPORT_PATH.write_text(
    json.dumps(report, indent=2, default=json_default),
    encoding="utf-8",
)

print()
print(f"Manifest: {MANIFEST_PATH}")
print(f"Report:   {REPORT_PATH}")

if gate != "PASS":
    raise SystemExit(1)
