"""Record or verify the user's scoped residual-history Paper risk acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paper.paper_coverage_acceptance import (
    validate_acceptance_record,
    write_acceptance_record,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "validate"))
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-user-acceptance", action="store_true",
                        help="required to persist the user's explicit Paper-only risk acceptance")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            if not args.record_user_acceptance:
                raise ValueError("record requires --record-user-acceptance")
            record = write_acceptance_record(args.certificate, args.output)
            print(json.dumps({"created": True, "path": str(args.output),
                              "policy": record["policy"], "decision": record["decision"],
                              "accepted_span_count": len(record["accepted_unresolved_spans"]),
                              "accepted_minutes": record["decision"]["unresolved_absent_minutes"],
                              "strict_readiness_preserved": record["strict_certificate"]["readiness"]}, indent=2))
            return 0
        result = validate_acceptance_record(args.output, args.certificate)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
