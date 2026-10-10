"""Read-only, provenance-checked research/Paper analytics for the dashboard."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


STRATEGIES = ("MRL1", "MRS2", "S2R", "ORB")
EXPECTED_COUNTS = {"MRL1": 430, "MRS2": 863, "S2R": 520, "ORB": 1442}
OOS_START = "2020-06-23T00:00:00+00:00"
OOS_END = "2026-06-19T23:59:59.999999+00:00"
EXPECTED_TOTAL_R = 289.6619012127312
EXPECTED_TRADES_SHA256 = "950f95bba2ed74bb11fc5840a85d720b4f197aac5c72debcd0f6624f401c81ea"
EXPECTED_DAILY_SHA256 = "f8b986015d85e993c81d1f0152bea5a5241bda7ce1e9602361ceb8c5db1a99a9"
DEFAULT_RESEARCH_DIR = Path("src/research/results/portfolio/independent_reproduction")


class ResearchIntegrityError(ValueError):
    """Raised when local research inputs do not match the validated benchmark."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    # csv.DictReader tolerates a final partial row; required-field parsing below
    # rejects it from quantitative use while leaving the source file untouched.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_validated_research(root: str | Path, research_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the independently reproduced OOS portfolio after integrity checks.

    No other similarly named replay output is accepted as the benchmark.
    """
    project = Path(root).resolve()
    directory = (project / research_dir if research_dir else project / DEFAULT_RESEARCH_DIR).resolve()
    report_path = directory / "independent_reproduction_report.json"
    trades_path = directory / "independent_reproduction_oos_trades.csv"
    daily_path = directory / "independent_reproduction_daily.csv"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trades_rows = _read_csv(trades_path)
        daily_rows = _read_csv(daily_path)
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ResearchIntegrityError(f"validated Research source unavailable or unreadable ({type(exc).__name__})") from exc

    official = report.get("official_oos") or {}
    if (report.get("model_version") != "v1.3-full-system-validation"
            or official.get("start") != OOS_START
            or not str(official.get("end", "")).startswith("2026-06-19T23:59:59")
            or report.get("expected_total") != 3255
            or report.get("reproduced_counts") != EXPECTED_COUNTS
            or any(row.get("level") != "PASS" for row in report.get("findings", []))):
        raise ResearchIntegrityError("research reproduction report does not match the frozen OOS specification")

    actual_hashes = {"trades_sha256": _sha256(trades_path), "daily_sha256": _sha256(daily_path)}
    if (actual_hashes["trades_sha256"] != EXPECTED_TRADES_SHA256
            or actual_hashes["daily_sha256"] != EXPECTED_DAILY_SHA256):
        raise ResearchIntegrityError("frozen Research output artifact hash mismatch")
    for item in (report.get("source_artifacts") or {}).values():
        relative = str(item.get("path", "")).replace("\\", "/")
        source = project / relative
        if not source.is_file() or _sha256(source) != str(item.get("sha256", "")).lower():
            raise ResearchIntegrityError(f"validated component source hash mismatch: {relative}")

    trades: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    counts = {name: 0 for name in STRATEGIES}
    for index, row in enumerate(trades_rows):
        strategy = row.get("strategy_name")
        entry = _parse_utc(row.get("entry_timestamp"))
        try:
            r_value = float(row["r_multiple"])
        except (KeyError, TypeError, ValueError):
            continue
        if strategy not in EXPECTED_COUNTS or entry is None or not math.isfinite(r_value):
            continue
        if not (_parse_utc(OOS_START) <= entry <= _parse_utc(OOS_END)):
            raise ResearchIntegrityError("research trade lies outside the validated OOS bounds")
        key = (strategy, entry.isoformat())
        if key in keys:
            raise ResearchIntegrityError(f"duplicate Research trade identity: {strategy} {entry.isoformat()}")
        keys.add(key)
        exit_time = _parse_utc(row.get("exit_timestamp"))
        trades.append({**row, "strategy": strategy, "entry_timestamp_utc": entry.isoformat(),
                       "exit_timestamp_utc": exit_time.isoformat() if exit_time else None,
                       "r_multiple": r_value, "source_row": index})
        counts[strategy] += 1
    if len(trades) != 3255 or counts != EXPECTED_COUNTS:
        raise ResearchIntegrityError(f"Research ledger count mismatch: total={len(trades)}, strategies={counts}")
    trades.sort(key=lambda item: (item["entry_timestamp_utc"], item["strategy"], item["source_row"]))

    daily: list[dict[str, Any]] = []
    for row in daily_rows:
        stamp = _parse_utc(row.get("trading_day"))
        try:
            r_value = float(row["daily_R"])
        except (KeyError, TypeError, ValueError):
            continue
        if stamp is None or not math.isfinite(r_value):
            continue
        daily.append({"timestamp_utc": stamp.isoformat(), "daily_r": r_value})
    daily.sort(key=lambda item: item["timestamp_utc"])
    if len(daily) != 1486 or not math.isclose(sum(item["daily_r"] for item in daily), EXPECTED_TOTAL_R, abs_tol=1e-6):
        raise ResearchIntegrityError("validated daily-R artifact does not reconcile to the frozen portfolio total")

    cumulative = 0.0
    peak = 0.0
    for row in daily:
        cumulative += row["daily_r"]
        peak = max(peak, cumulative)
        row["cumulative_r"] = cumulative
        row["drawdown_r"] = cumulative - peak
    return {
        "available": True, "trades": trades, "daily": daily, "report": report,
        "provenance": {"report_path": str(report_path), "trade_path": str(trades_path),
                       "daily_path": str(daily_path), **actual_hashes,
                       "model_version": report["model_version"],
                       "findings_passed": len(report.get("findings", [])),
                       "trade_rows": len(trades), "daily_rows": len(daily)},
    }


def reconstruct_paper_outcomes(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Normalize persisted closed-trade rows without inventing or deduplicating facts."""
    if rows is None:
        return {"available": False, "reason": "Paper trade outcomes unavailable", "trades": []}
    output: list[dict[str, Any]] = []
    identities: set[str] = set()
    invalid = 0
    for row in rows:
        if not isinstance(row, Mapping):
            invalid += 1
            continue
        trade_id = str(row.get("trade_id") or "")
        stamp = _parse_utc(row.get("exit_timestamp_utc"))
        try:
            r_value = float(row["realized_r"])
            pnl = float(row["net_pnl"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if (not trade_id or trade_id in identities or stamp is None
                or not math.isfinite(r_value) or not math.isfinite(pnl)):
            if trade_id in identities:
                return {"available": False, "reason": "Duplicate Paper trade ID in analytics response", "trades": []}
            invalid += 1
            continue
        identities.add(trade_id)
        output.append({**dict(row), "trade_id": trade_id, "strategy": row.get("strategy") or "UNAVAILABLE",
                       "exit_timestamp_utc": stamp.isoformat(), "realized_r": r_value, "net_pnl": pnl})
    output.sort(key=lambda item: (item["exit_timestamp_utc"], item["trade_id"]))
    return {"available": True, "reason": None, "trades": output, "invalid_rows": invalid,
            "trade_count": len(output)}


def summarize_r(values: Iterable[float]) -> dict[str, Any]:
    data = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed):
            data.append(parsed)
    gains = sum(value for value in data if value > 0)
    losses = abs(sum(value for value in data if value < 0))
    return {"trades": len(data), "total_r": sum(data),
            "expectancy_r": sum(data) / len(data) if data else None,
            "profit_factor": gains / losses if losses else None,
            "win_rate": sum(value > 0 for value in data) / len(data) if data else None}


def rolling_metrics(trades: Iterable[Mapping[str, Any]], *, value_key: str, time_key: str,
                    window: int = 25, minimum: int | None = None) -> list[dict[str, Any]]:
    """Trade-ordered rolling expectancy, PF and WR; no partial windows by default."""
    clean = []
    for row in trades:
        stamp = _parse_utc(row.get(time_key))
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if stamp and math.isfinite(value):
            clean.append((stamp, value))
    clean.sort(key=lambda item: item[0])
    n_min = window if minimum is None else max(1, int(minimum))
    if window < 1 or n_min > window:
        raise ValueError("rolling window must be positive and minimum cannot exceed window")
    result = []
    for end in range(len(clean)):
        start = max(0, end - window + 1)
        sample = clean[start:end + 1]
        if len(sample) < n_min:
            continue
        metrics = summarize_r(value for _, value in sample)
        result.append({"timestamp_utc": sample[-1][0].isoformat(), "sample_size": len(sample), **metrics})
    return result


def moving_block_expectancy_band(values: Iterable[float], *, window: int = 25,
                                 block_length: int = 5, samples: int = 2000,
                                 seed: int = 20261010) -> dict[str, Any]:
    """Percentile band for a fixed-window mean using a moving-block bootstrap.

    Blocks are sampled from the chronological research R sequence to retain
    short-range serial dependence. This is descriptive, not a hypothesis test
    or proof of alpha decay.
    """
    data = [float(value) for value in values if math.isfinite(float(value))]
    if window < 1 or block_length < 1 or samples < 1:
        raise ValueError("window, block_length, and samples must be positive")
    if len(data) < window:
        return {"available": False, "reason": f"Need at least {window} Research trades; found {len(data)}",
                "sample_size": len(data), "window": window, "block_length": block_length}
    block = min(block_length, len(data))
    rng = random.Random(seed)
    means = []
    max_start = len(data) - block
    for _ in range(samples):
        draw: list[float] = []
        while len(draw) < window:
            start = rng.randrange(max_start + 1)
            draw.extend(data[start:start + block])
        means.append(sum(draw[:window]) / window)
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return {"available": True, "lower": lower, "upper": upper,
            "sample_size": len(data), "window": window, "block_length": block,
            "bootstrap_samples": samples, "seed": seed,
            "method": "moving-block bootstrap of chronological trade R; 2.5th–97.5th percentiles"}


def cumulative_r_series(trades: Iterable[Mapping[str, Any]], *, value_key: str,
                        time_key: str, strategy: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in trades:
        if strategy and row.get("strategy") != strategy:
            continue
        stamp = _parse_utc(row.get(time_key))
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if stamp and math.isfinite(value):
            rows.append((stamp, value, row))
    rows.sort(key=lambda item: item[0])
    total = 0.0
    output = []
    for stamp, value, row in rows:
        total += value
        output.append({"timestamp_utc": stamp.isoformat(), "cumulative_r": total,
                       "trade_r": value, "strategy": row.get("strategy")})
    return output


def equity_drawdown_r_series(daily: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = peak = 0.0
    result = []
    for row in sorted(daily, key=lambda item: str(item.get("timestamp_utc", ""))):
        try:
            value = float(row["daily_r"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or not _parse_utc(row.get("timestamp_utc")):
            continue
        total += value
        peak = max(peak, total)
        result.append({"timestamp_utc": row["timestamp_utc"], "cumulative_r": total,
                       "drawdown_r": total - peak, "daily_r": value})
    return result
