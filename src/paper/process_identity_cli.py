"""Register a running Paper PID for the independent Windows watchdog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.paper.process_identity import write_identity_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register an active Paper process identity (read-only metadata).")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid", required=True, type=int)
    args = parser.parse_args(argv)
    run = Path(args.run_dir).resolve()
    try:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        state = str(status.get("system", {}).get("state", "UNAVAILABLE")).upper()
        if state not in {"RUNNING", "RECOVERING", "DEGRADED", "PAUSED_REFIT"}:
            raise ValueError(f"persisted Paper state is {state}")
        record = write_identity_record(run / "paper_process_identity.json", run.name, args.pid)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Paper PID registration failed ({type(exc).__name__}); no engine state was changed", file=sys.stderr)
        return 1
    print(f"Registered Paper process PID {record['pid']} for run {record['run_id']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
