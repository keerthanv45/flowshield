"""
End-to-end integration test for the Phase 2 pipeline. Uses a smaller
synthetic dataset (shorter period, fewer events) than the full 20,000-
event Phase 1 dataset so the test suite stays fast — the full-scale
pipeline is exercised for real by `scripts/evaluate_phase2.py`, whose
actual output is what's reported in the Phase 2 summary.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ml.data_generation.generator import GenerationConfig, PaymentEventGenerator
from ml.analysis.report import events_to_dataframe
from ml.evaluation.incident_evaluation import evaluate
from ml.health.incidents import IncidentType
from ml.health.pipeline import run_health_pipeline


@pytest.fixture(scope="module")
def small_pipeline_result():
    config = GenerationConfig(
        n_events=6000, seed=123, period_start=datetime(2026, 8, 1), period_days=7,
    )
    generator = PaymentEventGenerator(config)
    events = generator.generate()
    df = events_to_dataframe(events)
    incident_windows = [w.to_dict() for w in generator.incident_windows]
    result = run_health_pipeline(df, incident_windows)
    return result, incident_windows


class TestPipelineRuns:
    def test_produces_classifications_for_every_window(self, small_pipeline_result):
        result, _ = small_pipeline_result
        assert len(result.classifications) == len(result.overall_scored)

    def test_health_score_and_anomaly_columns_present(self, small_pipeline_result):
        result, _ = small_pipeline_result
        for col in ["health_score", "anomaly_score", "is_anomaly"]:
            assert col in result.overall_scored.columns

    def test_at_least_one_incident_confirmed(self, small_pipeline_result):
        # With 5 injected incident scenarios in the schedule, the pipeline
        # should confirm at least one Incident somewhere.
        result, _ = small_pipeline_result
        assert len(result.incidents) > 0

    def test_latency_spike_window_detected_somewhere(self, small_pipeline_result):
        result, incident_windows = small_pipeline_result
        latency_incidents = [inc for inc in result.incidents if inc.incident_type == IncidentType.LATENCY_SPIKE]
        assert len(latency_incidents) > 0


class TestEvaluationRunsOnPipelineOutput:
    def test_evaluate_produces_bounded_metrics(self, small_pipeline_result):
        result, incident_windows = small_pipeline_result
        report = evaluate(result.classifications, result.incidents, incident_windows)
        assert 0.0 <= report.classification_accuracy <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1 <= 1.0
        assert 0.0 <= report.false_positive_rate <= 1.0

    def test_false_positive_rate_is_low(self, small_pipeline_result):
        # The core false-positive-handling requirement: normal windows
        # should rarely be confirmed as incidents.
        result, incident_windows = small_pipeline_result
        report = evaluate(result.classifications, result.incidents, incident_windows)
        assert report.false_positive_rate < 0.15
