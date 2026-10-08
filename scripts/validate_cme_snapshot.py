"""Validate an already reviewed CME snapshot and its review record.

This command never downloads or infers trading hours. Human verification of
the product-specific CME schedule remains a required input.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.paper.cme_calendar import CMECalendarSnapshot, CME_HOLIDAY_SOURCE


REQUIRED_START = "2026-10-07"
REQUIRED_END = "2026-12-31"


def validate(snapshot_path: Path, review_path: Path) -> dict:
    snapshot = CMECalendarSnapshot.from_json(snapshot_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    errors = []
    if snapshot.source != CME_HOLIDAY_SOURCE:
        errors.append("snapshot source must be the official CME trading-hours page")
    if snapshot.coverage_start.isoformat() > REQUIRED_START or snapshot.coverage_end.isoformat() < REQUIRED_END:
        errors.append(f"snapshot must cover {REQUIRED_START} through {REQUIRED_END}")
    if review.get("source_url") != CME_HOLIDAY_SOURCE:
        errors.append("review source_url must identify the official CME schedule")
    if review.get("product") != "MNQ":
        errors.append("review record must specify product MNQ")
    if review.get("timezone") != "America/New_York":
        errors.append("review record timezone must be America/New_York")
    if review.get("coverage_start") != REQUIRED_START or review.get("coverage_end") != REQUIRED_END:
        errors.append(f"review record must cover {REQUIRED_START} through {REQUIRED_END}")
    if not review.get("reviewed_at_utc"):
        errors.append("reviewed_at_utc is required")
    else:
        try:
            reviewed_at = datetime.fromisoformat(str(review["reviewed_at_utc"]).replace("Z", "+00:00"))
            if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
                errors.append("reviewed_at_utc must be timezone-aware UTC")
        except ValueError:
            errors.append("reviewed_at_utc must be an ISO-8601 timestamp")
    if review.get("snapshot_identity") != snapshot.identity:
        errors.append("review record snapshot_identity does not match the parsed snapshot")
    out_of_coverage = [
        key for key in snapshot.exceptions
        if not snapshot.coverage_start <= date.fromisoformat(key) <= snapshot.coverage_end
    ]
    if out_of_coverage:
        errors.append(f"snapshot exceptions fall outside declared coverage: {out_of_coverage}")
    if errors:
        raise ValueError("; ".join(errors))
    return {"valid": True, "version": snapshot.version,
            "snapshot_identity": snapshot.identity,
            "coverage_start": snapshot.coverage_start.isoformat(),
            "coverage_end": snapshot.coverage_end.isoformat(),
            "product": "MNQ", "timezone": "America/New_York"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--review-record", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.snapshot, args.review_record), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
