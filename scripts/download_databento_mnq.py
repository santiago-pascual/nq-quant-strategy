from __future__ import annotations

import os
from pathlib import Path

import databento as db


DATASET = "GLBX.MDP3"
SYMBOL = "MNQ.v.0"
SCHEMA = "ohlcv-1m"

START = "2010-06-06"
END = "2026-08-27"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "mnq" / "ohlcv_1m"


def main() -> None:
    if not os.environ.get("DATABENTO_API_KEY"):
        raise RuntimeError("DATABENTO_API_KEY is not set.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = db.Historical()

    print("=" * 70)
    print("DATABENTO MNQ HISTORICAL DOWNLOAD")
    print("=" * 70)
    print(f"Dataset : {DATASET}")
    print(f"Symbol  : {SYMBOL}")
    print(f"Schema  : {SCHEMA}")
    print(f"Range   : {START} -> {END}")
    print(f"Output  : {OUTPUT_DIR}")
    print()

    print("Submitting batch request...")

    job = client.batch.submit_job(
        dataset=DATASET,
        symbols=SYMBOL,
        schema=SCHEMA,
        stype_in="continuous",
        start=START,
        end=END,
        encoding="csv",
        compression="zstd",
        map_symbols=True,
        split_duration="year",
    )

    print()
    print("=" * 70)
    print("BATCH SUBMITTED")
    print("=" * 70)

    print(f"Job ID : {job['id']}")
    print(f"State  : {job['state']}")
    print()
    print("Wait for the job to finish in Databento.")
    print("Do NOT run the research pipeline yet.")


if __name__ == "__main__":
    main()
