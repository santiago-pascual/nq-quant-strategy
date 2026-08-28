from __future__ import annotations

import os

import databento as db


DATASET = "GLBX.MDP3"
SYMBOL = "MNQ.v.0"
SCHEMA = "ohlcv-1m"

START = "2010-06-06"
END = "2026-08-27"


def main() -> None:
    if not os.environ.get("DATABENTO_API_KEY"):
        raise RuntimeError("DATABENTO_API_KEY is not set.")

    client = db.Historical()

    print("=" * 70)
    print("DATABENTO MNQ COST CHECK")
    print("=" * 70)

    print(f"Dataset : {DATASET}")
    print(f"Symbol  : {SYMBOL}")
    print(f"Schema  : {SCHEMA}")
    print(f"Range   : {START} -> {END}")
    print()

    cost = client.metadata.get_cost(
        dataset=DATASET,
        symbols=SYMBOL,
        schema=SCHEMA,
        start=START,
        end=END,
        stype_in="continuous",
    )

    print(f"Estimated cost: ${float(cost):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
