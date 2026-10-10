import json
from pathlib import Path

from src.paper.delayed_paper_cli import (
    DEFAULT_RECOVERY_VALIDATION, _expected_risk_configuration, _parser, main, readiness_report,
)


def test_delayed_paper_readiness_discovers_repository_recovery_record_by_default():
    args = _parser().parse_args(["readiness", "--output-dir", "results/paper/delayed_test"])
    assert args.recovery_validation == DEFAULT_RECOVERY_VALIDATION
    assert args.recovery_validation.is_file()
    evidence = json.loads(args.recovery_validation.read_text(encoding="utf-8"))
    assert evidence.get("status") == "PASSED"
    assert evidence.get("all_four_strategies") is True
    assert evidence.get("fitted_hmm_and_refit_boundary") is True
    assert evidence.get("one_hour_catchup") is True
    assert evidence.get("uninterrupted_equals_recovered") is True
    assert evidence.get("pending_strategy_order_checkpoint_restore") is True
    assert evidence.get("runtime_fingerprint")
    assert evidence.get("test_sources")
    report = readiness_report(
        activation="2026-10-09T00:00:00Z", calendar_path=None, review_path=None,
        bootstrap_path=None, identity_path=None,
        recovery_validation_path=args.recovery_validation, cost_config=None,
        output_dir=Path("results/paper/delayed_test"), host="127.0.0.1", port=7497,
    )
    assert not any("recovery validation" in item for item in report["blockers"])
    assert report["recovery_validation"]["passed"] is True


def test_delayed_paper_readiness_fails_closed_without_seed_and_validation():
    report = readiness_report(
        activation="2026-10-09T00:00:00Z",
        calendar_path=Path("src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json"),
        review_path=Path("src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.review.json"),
        bootstrap_path=None, identity_path=None, recovery_validation_path=None,
        cost_config=Path("src/paper/config/topstepx_mnq_fees_2026-07.json"),
        output_dir=Path("results/paper/delayed_test"), host="127.0.0.1", port=7496,
    )
    assert report["ready"] is False
    assert report["orders_enabled"] is False
    assert any("bootstrap" in item for item in report["blockers"])
    assert any("recovery validation" in item for item in report["blockers"])


def test_delayed_paper_start_risk_configuration_matches_frozen_policy():
    from dataclasses import asdict
    from src.risk.policy import XFA_50K_PRODUCTION_POLICY

    assert _expected_risk_configuration() == asdict(XFA_50K_PRODUCTION_POLICY)


def test_delayed_paper_readiness_rejects_output_outside_dedicated_root():
    report = readiness_report(
        activation="2026-10-09T00:00:00Z", calendar_path=None, review_path=None,
        bootstrap_path=None, identity_path=None, recovery_validation_path=None,
        cost_config=None, output_dir=Path("ordinary-output"),
        host="127.0.0.1", port=7496,
    )
    assert any("dedicated results/paper/delayed_*" in item for item in report["blockers"])


def test_delayed_paper_stop_refuses_non_delayed_replay_directory(capsys):
    result = main(["stop", "--output-dir", "results/paper/full_research_replay_final_v3"])
    assert result == 2
    assert not Path("results/paper/full_research_replay_final_v3/stop.request").exists()
    assert "not a known delayed-Paper run" in capsys.readouterr().out
