from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.health.incidents import (
    ConcentrationResult,
    IncidentStatus,
    IncidentType,
    Severity,
    apply_persistence,
    classify_window,
    compute_concentration,
    compute_severity,
    rule_signals_for_row,
    WindowClassification,
)


def make_row(**overrides):
    base = dict(
        success_rate=0.9,
        baseline_success_rate=0.9,
        success_rate_delta=0.0,
        success_rate_z=0.0,
        failure_rate=0.1,
        baseline_failure_rate=0.1,
        failure_rate_delta=0.0,
        failure_rate_z=0.0,
        average_latency_ms=1200.0,
        baseline_average_latency_ms=1200.0,
        latency_ratio=1.0,
        transaction_count=15,
        baseline_transaction_count=15,
        volume_ratio=1.0,
        is_anomaly=False,
    )
    base.update(overrides)
    return base


class TestRuleSignals:
    def test_no_signals_on_normal_row(self):
        assert rule_signals_for_row(make_row()) == []

    def test_success_rate_degradation_requires_both_magnitude_and_significance(self):
        # Big magnitude but z not significant -> no signal.
        row = make_row(success_rate=0.75, success_rate_delta=-0.15, success_rate_z=-1.0)
        assert "SUCCESS_RATE_DEGRADATION" not in rule_signals_for_row(row)
        # Big magnitude AND significant z -> signal fires.
        row2 = make_row(success_rate=0.75, success_rate_delta=-0.15, success_rate_z=-3.0)
        assert "SUCCESS_RATE_DEGRADATION" in rule_signals_for_row(row2)

    def test_latency_spike_signal(self):
        row = make_row(latency_ratio=2.0)
        assert "LATENCY_SPIKE" in rule_signals_for_row(row)

    def test_failure_surge_signal(self):
        row = make_row(failure_rate=0.3, failure_rate_delta=0.2, failure_rate_z=3.0)
        assert "FAILURE_SURGE" in rule_signals_for_row(row)

    def test_volume_anomaly_signal_low_and_high(self):
        assert "VOLUME_ANOMALY" in rule_signals_for_row(make_row(volume_ratio=0.2))
        assert "VOLUME_ANOMALY" in rule_signals_for_row(make_row(volume_ratio=3.0))
        assert "VOLUME_ANOMALY" not in rule_signals_for_row(make_row(volume_ratio=1.0))


class TestConcentration:
    def test_empty_group_returns_zero(self):
        result = compute_concentration(pd.DataFrame(), ["bank"])
        assert result.n_considered == 0
        assert result.n_degraded == 0

    def test_low_volume_members_excluded(self):
        df = pd.DataFrame(
            [
                make_row(transaction_count=2, success_rate_delta=-0.5, success_rate_z=-5.0),  # below min volume
            ]
        )
        df["bank"] = ["HDFC"]
        result = compute_concentration(df, ["bank"], min_volume=5)
        assert result.n_considered == 0

    def test_degraded_member_detected(self):
        df = pd.DataFrame(
            [
                make_row(transaction_count=20, success_rate_delta=-0.5, success_rate_z=-5.0),
                make_row(transaction_count=20, success_rate_delta=0.0, success_rate_z=0.0),
            ]
        )
        df["bank"] = ["HDFC", "SBI"]
        result = compute_concentration(df, ["bank"], min_volume=5)
        assert result.n_considered == 2
        assert result.n_degraded == 1
        assert result.degraded_values == ["HDFC"]


class TestClassification:
    def test_no_signal_no_anomaly_is_normal(self):
        result = classify_window(make_row(), {})
        assert result.incident_type == IncidentType.NORMAL

    def test_bank_rail_degradation_classification(self):
        overall = make_row(
            success_rate=0.5, success_rate_delta=-0.4, success_rate_z=-6.0,
            failure_rate=0.5, failure_rate_delta=0.4, failure_rate_z=6.0,
        )
        concentration = {
            "bank": ConcentrationResult(n_considered=5, n_degraded=1, degraded_values=["HDFC"]),
            "payment_method": ConcentrationResult(n_considered=4, n_degraded=1, degraded_values=["UPI"]),
            "region": ConcentrationResult(n_considered=6, n_degraded=0, degraded_values=[]),
            "bank_payment_method": ConcentrationResult(
                n_considered=20, n_degraded=1, degraded_values=[("HDFC", "UPI")]
            ),
        }
        result = classify_window(overall, concentration)
        assert result.incident_type == IncidentType.BANK_RAIL_DEGRADATION
        assert result.affected_dimensions == [{"bank": "HDFC", "payment_method": "UPI"}]

    def test_regional_degradation_classification(self):
        overall = make_row(
            success_rate=0.55, success_rate_delta=-0.35, success_rate_z=-5.0,
            failure_rate=0.45, failure_rate_delta=0.35, failure_rate_z=5.0,
        )
        concentration = {
            "bank": ConcentrationResult(n_considered=5, n_degraded=2, degraded_values=["HDFC", "SBI"]),
            "payment_method": ConcentrationResult(n_considered=4, n_degraded=2, degraded_values=["UPI", "CARD"]),
            "region": ConcentrationResult(n_considered=6, n_degraded=1, degraded_values=["KA"]),
            "bank_payment_method": ConcentrationResult(n_considered=20, n_degraded=4, degraded_values=[]),
        }
        result = classify_window(overall, concentration)
        assert result.incident_type == IncidentType.REGIONAL_DEGRADATION
        assert result.affected_dimensions == [{"region": "KA"}]

    def test_latency_spike_classification(self):
        overall = make_row(
            latency_ratio=3.0,
            success_rate=0.7, success_rate_delta=-0.2, success_rate_z=-4.0,
            failure_rate=0.3, failure_rate_delta=0.2, failure_rate_z=4.0,
        )
        concentration = {
            "bank": ConcentrationResult(n_considered=5, n_degraded=4, degraded_values=["HDFC", "SBI", "ICICI", "AXIS"]),
            "payment_method": ConcentrationResult(n_considered=4, n_degraded=3, degraded_values=["UPI", "CARD", "WALLET"]),
            "region": ConcentrationResult(n_considered=6, n_degraded=5, degraded_values=[]),
            "bank_payment_method": ConcentrationResult(n_considered=20, n_degraded=15, degraded_values=[]),
        }
        result = classify_window(overall, concentration)
        assert result.incident_type == IncidentType.LATENCY_SPIKE

    def test_merchant_system_degradation_classification(self):
        overall = make_row(
            latency_ratio=1.1,  # latency NOT dominant
            success_rate=0.6, success_rate_delta=-0.3, success_rate_z=-5.0,
            failure_rate=0.4, failure_rate_delta=0.3, failure_rate_z=5.0,
        )
        concentration = {
            "bank": ConcentrationResult(n_considered=5, n_degraded=4, degraded_values=["HDFC", "SBI", "ICICI", "AXIS"]),
            "payment_method": ConcentrationResult(n_considered=4, n_degraded=3, degraded_values=["UPI", "CARD", "WALLET"]),
            "region": ConcentrationResult(n_considered=6, n_degraded=5, degraded_values=[]),
            "bank_payment_method": ConcentrationResult(n_considered=20, n_degraded=15, degraded_values=[]),
        }
        result = classify_window(overall, concentration)
        assert result.incident_type == IncidentType.MERCHANT_SYSTEM_DEGRADATION

    def test_isolated_failures_classification(self):
        overall = make_row(
            success_rate=0.75, success_rate_delta=-0.15, success_rate_z=-3.0,
            failure_rate=0.25, failure_rate_delta=0.15, failure_rate_z=3.0,
        )
        concentration = {
            "bank": ConcentrationResult(n_considered=5, n_degraded=0, degraded_values=[]),
            "payment_method": ConcentrationResult(n_considered=4, n_degraded=0, degraded_values=[]),
            "region": ConcentrationResult(n_considered=6, n_degraded=0, degraded_values=[]),
            "bank_payment_method": ConcentrationResult(n_considered=20, n_degraded=0, degraded_values=[]),
        }
        result = classify_window(overall, concentration)
        assert result.incident_type == IncidentType.ISOLATED_FAILURES

    def test_anomaly_only_triggers_classification(self):
        overall = make_row(is_anomaly=True)
        result = classify_window(overall, {})
        assert result.incident_type != IncidentType.NORMAL


class TestSeverity:
    def test_critical_from_low_health_score(self):
        assert compute_severity(20.0, make_row(success_rate_delta=-0.05), []) == Severity.CRITICAL

    def test_critical_from_extreme_success_delta(self):
        assert compute_severity(70.0, make_row(success_rate_delta=-0.35), []) == Severity.CRITICAL

    def test_warning_from_moderate_health_score(self):
        assert compute_severity(55.0, make_row(success_rate_delta=-0.05), []) == Severity.WARNING

    def test_warning_from_multiple_signals(self):
        assert compute_severity(90.0, make_row(success_rate_delta=-0.01), ["A", "B"]) == Severity.WARNING

    def test_info_otherwise(self):
        assert compute_severity(90.0, make_row(success_rate_delta=-0.01), ["A"]) == Severity.INFO

    def test_deterministic_same_input_same_output(self):
        row = make_row(success_rate_delta=-0.2)
        s1 = compute_severity(40.0, row, ["A", "B"])
        s2 = compute_severity(40.0, row, ["A", "B"])
        assert s1 == s2


def make_classification(ws, incident_type, severity):
    return WindowClassification(
        window_start=ws,
        window_end=ws + timedelta(minutes=15),
        incident_type=incident_type,
        signals=["SUCCESS_RATE_DEGRADATION"],
        affected_dimensions=[],
        evidence=["evidence line"],
        health_score=40.0,
        anomaly_score=0.5,
        is_anomaly=True,
        severity=severity,
    )


class TestPersistence:
    def test_single_noisy_window_not_confirmed(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.NORMAL, None),
            make_classification(base + timedelta(minutes=15), IncidentType.ISOLATED_FAILURES, Severity.WARNING),
            make_classification(base + timedelta(minutes=30), IncidentType.NORMAL, None),
        ]
        incidents = apply_persistence(classifications)
        assert len(incidents) == 0

    def test_two_consecutive_windows_confirmed(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.LATENCY_SPIKE, Severity.WARNING),
            make_classification(base + timedelta(minutes=15), IncidentType.LATENCY_SPIKE, Severity.WARNING),
        ]
        incidents = apply_persistence(classifications)
        assert len(incidents) == 1
        assert incidents[0].status == IncidentStatus.CONFIRMED
        assert incidents[0].n_windows == 2

    def test_single_critical_window_bypasses_persistence(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.BANK_RAIL_DEGRADATION, Severity.CRITICAL),
        ]
        incidents = apply_persistence(classifications)
        assert len(incidents) == 1
        assert incidents[0].severity == Severity.CRITICAL

    def test_isolated_failures_does_not_flood_critical_incidents(self):
        # Scattered single-window ISOLATED_FAILURES candidates, none
        # consecutive, none CRITICAL -> should produce zero incidents.
        base = datetime(2026, 8, 1)
        classifications = []
        for i in range(10):
            classifications.append(
                make_classification(
                    base + timedelta(minutes=30 * i), IncidentType.ISOLATED_FAILURES, Severity.WARNING
                )
            )
            classifications.append(
                make_classification(base + timedelta(minutes=30 * i + 15), IncidentType.NORMAL, None)
            )
        incidents = apply_persistence(classifications)
        assert len(incidents) == 0

    def test_episode_breaks_on_type_change(self):
        base = datetime(2026, 8, 1)
        classifications = [
            make_classification(base, IncidentType.LATENCY_SPIKE, Severity.WARNING),
            make_classification(base + timedelta(minutes=15), IncidentType.BANK_RAIL_DEGRADATION, Severity.WARNING),
        ]
        incidents = apply_persistence(classifications)
        # Each type only has 1 window and neither is CRITICAL -> neither confirmed.
        assert len(incidents) == 0
