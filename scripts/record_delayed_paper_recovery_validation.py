"""Run offline Paper recovery integration tests and write readiness evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results/diagnostics/delayed_paper_strategy_recovery_validation_20261009.json"
TESTS = [
    "tests/paper/test_ibkr_paper_recovery.py::test_hard_crash_after_real_orb_fill_replays_bar_without_duplicate_execution",
    "tests/paper/test_ibkr_paper_recovery.py::test_real_strategy_pending_entry_order_survives_checkpoint_and_restores_idempotently",
    "tests/paper/test_ibkr_paper_recovery.py::test_acknowledged_paper_replay_matches_uninterrupted_engine_state",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:tmpdir", "-p", "no:cacheprovider",
        *TESTS,
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode:
        print(json.dumps({"status": "NOT_RECORDED", "pytest_exit_code": result.returncode}))
        return result.returncode

    from src.paper.realtime_checkpoint import runtime_identity

    payload = {
        "schema_version": 2,
        "status": "PASSED",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "deterministic_fixture_integration_tests",
        "tests_passed": TESTS,
        "test_sources": {
            str(Path(test.split("::", 1)[0]).as_posix()): sha256(ROOT / test.split("::", 1)[0])
            for test in TESTS
        },
        "test_command": " ".join(command),
        "runtime_fingerprint": runtime_identity(),
        "all_four_strategies": True,
        "strategy_names": ["MRL1", "S2R", "MRS2", "ORB"],
        "fitted_hmm_and_refit_boundary": True,
        "one_hour_catchup": True,
        "uninterrupted_equals_recovered": True,
        "pending_strategy_order_checkpoint_restore": True,
        "no_broker_orders": True,
        "limitations": (
            "Deterministic local fixtures only; not an IBKR/TWS smoke, not a production activation seed, "
            "and not evidence of historical calendar coverage."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASSED", "record": str(OUTPUT),
                      "tests": len(TESTS), "runtime_fingerprint": "recorded"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
