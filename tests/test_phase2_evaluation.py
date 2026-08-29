from datetime import datetime, timedelta

import pytest

from ml.evaluation.incident_evaluation import (
    confirmed_incident_type_by_window,
    evaluate,
    label_ground_truth,
)
from ml.health.incidents import (
    Incident,
    IncidentStatus,
    IncidentType,
    Severity,
    WindowClassification,
)


def make_classification(ws, incident_type, is_anomaly=False):
    return WindowClassification(
        window_start=ws,
        window_end=ws + timedelta(minutes=15),
        incident_type=incident_type,
        signals=[],
        affected_dimensions=[],
        evidence=[],
        health_score=90.0 if incident_type == IncidentType.NORMAL else 40.0,
        anomaly_score=0.1,
        is_anomaly=is_anomaly,
        severity=None if incident_type == IncidentType.NORMAL else Severity.WARNING,
    )


def make_incident(ws, we, incident_type):
    return Incident(
        incident_id=f"inc_{ws.isoformat()}",
        detected_at=ws,
        window_start=ws,
        window_end=we,
        severity=Severity.WARNING,
        incident_type=incident_type,
        anomaly_score=0.5,
        health_score=40.0,
        affected_dimensions=[],
        signals=[],
        evidence=[],
        status=IncidentStatus.CONFIRMED,
        n_windows=int((we - ws).total_seconds() / 900),
    )


class TestGroundTruthLabeling:
    def test_window_outside_incident_is_normal(self):
        starts = [datetime(2026, 8, 1, 0, 0)]
        ends = [datetime(2026, 8, 1, 0, 15)]
        windows = [
            {"scenario_type": "latency_spike", "start": "2026-08-02T00:00:00", "end": "2026-08-02T01:00:00"}
        ]
        labels = label_ground_truth(starts, ends, windows)
        assert labels == [IncidentType.NORMAL]

    def test_window_inside_incident_gets_scenario_label(self):
        starts = [datetime(2026, 8, 2, 0, 15)]
        ends = [datetime(2026, 8, 2, 0, 30)]
        windows = [
            {"scenario_type": "latency_spike", "start": "2026-08-02T00:00:00", "end": "2026-08-02T01:00:00"}
        ]
        labels = label_ground_truth(starts, ends, windows)
        assert labels == [IncidentType.LATENCY_SPIKE]


class TestConfirmedIncidentLookup:
    def test_window_inside_confirmed_incident_found(self):
        starts = [datetime(2026, 8, 1, 0, 15)]
        incidents = [
            make_incident(datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 1, 0), IncidentType.LATENCY_SPIKE)
        ]
        result = confirmed_incident_type_by_window(starts, incidents)
        assert result == [IncidentType.LATENCY_SPIKE]

    def test_window_outside_any_incident_is_none(self):
        starts = [datetime(2026, 8, 3, 0, 0)]
        incidents = [
            make_incident(datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 1, 0), IncidentType.LATENCY_SPIKE)
        ]
        result = confirmed_incident_type_by_window(starts, incidents)
        assert result == [None]


class TestEvaluateMetrics:
    def test_perfect_detection_gives_perfect_scores(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.NORMAL),
            make_classification(base + timedelta(minutes=15), IncidentType.LATENCY_SPIKE, is_anomaly=True),
            make_classification(base + timedelta(minutes=30), IncidentType.LATENCY_SPIKE, is_anomaly=True),
            make_classification(base + timedelta(minutes=45), IncidentType.NORMAL),
        ]
        incidents = [
            make_incident(base + timedelta(minutes=15), base + timedelta(minutes=45), IncidentType.LATENCY_SPIKE)
        ]
        incident_windows = [
            {
                "scenario_type": "latency_spike",
                "start": (base + timedelta(minutes=15)).isoformat(),
                "end": (base + timedelta(minutes=45)).isoformat(),
            }
        ]
        report = evaluate(classifications, incidents, incident_windows)
        assert report.classification_accuracy == pytest.approx(1.0)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1 == pytest.approx(1.0)
        assert report.false_positive_rate == pytest.approx(0.0)

    def test_missed_detection_lowers_recall(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.NORMAL),
            make_classification(base + timedelta(minutes=15), IncidentType.NORMAL),  # missed!
        ]
        incidents = []  # nothing confirmed
        incident_windows = [
            {
                "scenario_type": "latency_spike",
                "start": (base + timedelta(minutes=15)).isoformat(),
                "end": (base + timedelta(minutes=30)).isoformat(),
            }
        ]
        report = evaluate(classifications, incidents, incident_windows)
        assert report.recall == pytest.approx(0.0)

    def test_false_positive_on_normal_window_detected(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.MERCHANT_SYSTEM_DEGRADATION, is_anomaly=True),
        ]
        incidents = [
            make_incident(base, base + timedelta(minutes=15), IncidentType.MERCHANT_SYSTEM_DEGRADATION)
        ]
        incident_windows = []  # no ground-truth incident at all -> false positive
        report = evaluate(classifications, incidents, incident_windows)
        assert report.false_positive_rate == pytest.approx(1.0)
        assert report.precision == pytest.approx(0.0)

    def test_isolated_failures_excluded_from_systemic_precision_recall(self):
        # ISOLATED_FAILURES is neither ground-truth-systemic nor
        # predicted-systemic, so it should not affect precision/recall.
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.ISOLATED_FAILURES),
        ]
        incidents = []
        incident_windows = [
            {
                "scenario_type": "isolated_failures",
                "start": base.isoformat(),
                "end": (base + timedelta(minutes=15)).isoformat(),
            }
        ]
        report = evaluate(classifications, incidents, incident_windows)
        # No systemic ground truth and no systemic prediction -> precision/recall trivially 0/0 -> 0 by convention.
        assert report.false_positive_rate == pytest.approx(0.0)

    def test_per_scenario_results_present_for_all_five_scenarios(self):
        base = datetime(2026, 8, 1)
        classifications = [make_classification(base, IncidentType.NORMAL)]
        report = evaluate(classifications, [], [])
        scenario_types = {s.scenario_type for s in report.per_scenario}
        assert scenario_types == {
            "bank_rail_degradation", "regional_degradation", "latency_spike",
            "merchant_system_degradation", "isolated_failures",
        }
