import hashlib
import json
from pathlib import Path
import unittest
import uuid
import shutil

import pandas as pd

from scripts.report_databento_coverage_evidence import build_evidence_report


class DatabentoCoverageEvidenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent
        self.root = scratch / f".tmp_coverage_evidence_{uuid.uuid4().hex}"
        self.root.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        payload = b"fixture zstd placeholder"
        (self.data / "tiny.csv.zst").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (self.data / "metadata.json").write_text(json.dumps({
            "job_id": "fixture-job",
            "query": {"dataset": "GLBX.MDP3", "schema": "ohlcv-1m",
                      "symbols": ["MNQ.v.0"], "stype_in": "continuous",
                      "stype_out": "instrument_id", "start": 1, "end": 2},
            "customizations": {"map_symbols": True},
        }))
        (self.data / "manifest.json").write_text(json.dumps({
            "job_id": "fixture-job", "files": [{"filename": "tiny.csv.zst",
                                                   "hash": f"sha256:{digest}"}]}))
        (self.data / "condition.json").write_text(json.dumps([
            {"date": "2026-10-08", "condition": "degraded"},
            {"date": "2026-10-09", "condition": "available"},
        ]))
        self.cert = self.root / "certificate.json"
        self.cert.write_text(json.dumps({
            "data_first_bar_utc": "2026-10-08T00:00:00Z",
            "data_last_bar_utc": "2026-10-09T00:01:00Z",
        }))
        self.gaps = self.root / "gaps.csv"
        pd.DataFrame([{
            "previous_observed_utc": "2026-10-08T23:58:00Z",
            "next_observed_utc": "2026-10-09T00:01:00Z",
            "missing_minutes": 2,
            "classification": "uncertain_uncertified_calendar",
        }]).to_csv(self.gaps, index=False)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_manifest_condition_and_uncertainty_are_reported_without_certifying(self):
        report = build_evidence_report(self.data, self.cert, self.gaps)
        self.assertTrue(report["batch_integrity"]["all_manifest_files_present_and_matching"])
        self.assertEqual(report["vendor_day_condition_metadata"]["status_counts"],
                         {"available": 1, "degraded": 1})
        self.assertEqual(report["uncertain_gap_overlap_with_degraded_dates"]["absent_minute_count"], 1)
        self.assertEqual(report["uncertain_gap_overlap_with_degraded_dates"]["absent_minutes_by_utc_date"],
                         {"2026-10-08": 1})
        classification = report["historical_absence_evidence"]
        self.assertEqual(classification["gap_spans_by_classification"],
                         {"uncertain_uncertified_calendar": 1})
        self.assertEqual(classification["intervening_wall_clock_minutes_inside_gap_spans_by_classification"],
                         {"uncertain_uncertified_calendar": 2})
        self.assertEqual(classification["demonstrated_missing_required_observations"], 0)
        self.assertFalse(classification["legitimate_sparse_trade_ohlcv"]["distinguishable_from_capture_loss"])
        self.assertTrue(classification["degraded_condition_is_not_a_completeness_finding"])
        self.assertFalse(report["coverage_conclusion"]["per_minute_source_completeness_proven"])
        self.assertEqual(report["coverage_conclusion"]["readiness"], "NOT_FULLY_CERTIFIED")

    def test_manifest_corruption_is_rejected_as_nonmatching(self):
        (self.data / "tiny.csv.zst").write_bytes(b"changed")
        report = build_evidence_report(self.data, self.cert, self.gaps)
        self.assertFalse(report["batch_integrity"]["all_manifest_files_present_and_matching"])
        self.assertEqual(report["batch_integrity"]["hash_mismatches"], ["tiny.csv.zst"])


if __name__ == "__main__":
    unittest.main()
