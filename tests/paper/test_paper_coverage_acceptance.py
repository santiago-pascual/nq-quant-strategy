from __future__ import annotations

import pytest

from src.paper.paper_coverage_acceptance import _paper_acceptance_decision


def _evidence():
    spans = []
    for index in range(8):
        spans.append({
            "missing_minutes": 900,
            "minute_class_counts": {"degraded_source_day_gap_unresolved": 131 if index < 7 else 134},
        })
    return ({"status": "VALID", "observed_data_integrity": True,
             "coverage": {"demonstrated_missing_required_observations": 0}},
            {"readiness": "NOT_FULLY_CERTIFIED"}, spans)


def test_policy_records_scoped_user_acceptance_without_relabeling_strict_certificate():
    validation, certificate, spans = _evidence()
    decision = _paper_acceptance_decision(validation, certificate, spans)

    assert decision == {
        "policy": "paper_research_quality_accepted",
        "decision": "ACCEPTED_FOR_INTERNAL_SIMULATED_PAPER_ONLY",
        "strict_certificate_readiness": "NOT_FULLY_CERTIFIED",
        "activation_coverage_verified": False,
        "observed_data_integrity_verified": True,
        "demonstrated_missing_required_observations": 0,
        "unresolved_span_count": 8,
        "unresolved_absent_minutes": 1051,
    }


def test_policy_rejects_observed_integrity_failure_or_demonstrated_missing_rows():
    validation, certificate, spans = _evidence()
    with pytest.raises(ValueError, match="integrity"):
        _paper_acceptance_decision({**validation, "observed_data_integrity": False}, certificate, spans)
    with pytest.raises(ValueError, match="demonstrated missing"):
        _paper_acceptance_decision({**validation, "coverage": {
            "demonstrated_missing_required_observations": 1}}, certificate, spans)


def test_policy_rejects_changed_uncertainty_scope():
    validation, certificate, spans = _evidence()
    with pytest.raises(ValueError, match="eight-span"):
        _paper_acceptance_decision(validation, certificate, spans[:-1])
    spans[-1]["minute_class_counts"]["degraded_source_day_gap_unresolved"] = 135
    with pytest.raises(ValueError, match="eight-span"):
        _paper_acceptance_decision(validation, certificate, spans)
