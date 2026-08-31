from datetime import datetime

from backend.app.services.audit import AuditEventType, build_audit_trail
from backend.app.services.reasoning.schemas import ReasoningSource, RCAResult
from backend.app.services.recovery.outcome import build_outcome
from backend.app.services.recovery.schemas import (
    RecoveryActionType,
    RecoveryCandidate,
    RecoveryDecision,
    RiskLevel,
    SimulatedExecution,
)
from backend.app.services.recovery.revenue_risk import calculate_revenue_risk
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity
import pandas as pd


def make_incident(**overrides) -> Incident:
    base = dict(
        incident_id="inc_test_audit",
        detected_at=datetime(2026, 8, 3, 2, 0, 0),
        window_start=datetime(2026, 8, 3, 2, 0, 0),
        window_end=datetime(2026, 8, 3, 2, 15, 0),
        severity=Severity.CRITICAL,
        incident_type=IncidentType.BANK_RAIL_DEGRADATION,
        anomaly_score=0.8,
        health_score=25.0,
        affected_dimensions=[{"bank": "HDFC", "payment_method": "UPI"}],
        signals=["SUCCESS_RATE_DEGRADATION"],
        evidence=["evidence line"],
        status=IncidentStatus.CONFIRMED,
        n_windows=2,
    )
    base.update(overrides)
    return Incident(**base)


def make_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=500.0,
                failure_reason="timeout", bank="HDFC", payment_method="UPI", region="KA",
            ),
        ]
    )


def make_rca(confidence: float = 0.7) -> RCAResult:
    return RCAResult(
        root_cause="Bank rail degradation", confidence=confidence, explanation="x",
        supporting_evidence=[], affected_scope=[], recommended_actions=[],
        source=ReasoningSource.MOCK,
    )


def build_pipeline_objects(incident: Incident):
    events = make_events()
    revenue_risk = calculate_revenue_risk(incident, events)
    candidate = RecoveryCandidate(
        action=RecoveryActionType.ROUTE_ALTERNATE_RAIL, reason="test", eligible_failure_types=["timeout"],
        estimated_recovery=100.0, risk=RiskLevel.LOW, priority=1,
    )
    decision = RecoveryDecision(
        incident_id=incident.incident_id, recommended_action=RecoveryActionType.ROUTE_ALTERNATE_RAIL,
        decision_score=100.0, expected_recovery_amount=100.0, risk_level=RiskLevel.LOW,
        reasoning=["test reasoning"], candidates_considered=[candidate], policy_notes=[],
    )
    simulated = SimulatedExecution(
        action_id="sim_test", incident_id=incident.incident_id, action=RecoveryActionType.ROUTE_ALTERNATE_RAIL,
        eligible_transactions=1, simulated_successes=1, simulated_recovered_amount=500.0,
        seed=42, status="SIMULATED_COMPLETE",
    )
    outcome = build_outcome(simulated, decision, revenue_risk)
    rca = make_rca(0.7)
    return rca, revenue_risk, decision, simulated, outcome


class TestAuditEventGeneration:
    def test_generates_seven_events(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert len(events) == 7

    def test_event_types_match_expected_stages(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        types = [e.event_type for e in events]
        assert types == [
            AuditEventType.INCIDENT_DETECTED,
            AuditEventType.RCA_COMPLETED,
            AuditEventType.REVENUE_RISK_CALCULATED,
            AuditEventType.RECOVERY_POLICY_EVALUATED,
            AuditEventType.GUARDRAILS_CHECKED,
            AuditEventType.RECOVERY_SIMULATED,
            AuditEventType.OUTCOME_RECORDED,
        ]

    def test_values_come_from_real_objects_not_invented(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert f"{revenue_risk.gross_amount_at_risk:.2f}" in events[2].value
        assert decision.recommended_action.value in events[3].message
        assert simulated.status == events[5].status
        assert outcome.status == events[6].status

    def test_guardrails_status_reflects_policy_notes(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert events[4].status == "PASS"

        decision_with_notes = decision.model_copy(update={"policy_notes": ["Rejected X: reason"]})
        events2 = build_audit_trail(incident, rca, revenue_risk, decision_with_notes, simulated, outcome)
        assert events2[4].status == "ADJUSTED"


class TestEventOrdering:
    def test_order_field_sequential(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert [e.order for e in events] == [1, 2, 3, 4, 5, 6, 7]

    def test_timestamps_strictly_increasing(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_timestamps_derived_from_incident_detected_at(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert events[0].timestamp.startswith(incident.detected_at.isoformat()[:16])


class TestSimulationLabeledSimulated:
    def test_recovery_simulated_status_contains_simulated(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        assert "SIMULATED" in events[5].status
        assert "SIMULATED" in events[6].status
        assert "simulated" in events[6].message.lower()


class TestNoSecrets:
    def test_no_api_key_like_strings_in_events(self):
        incident = make_incident()
        rca, revenue_risk, decision, simulated, outcome = build_pipeline_objects(incident)
        events = build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)
        blob = " ".join(f"{e.message} {e.value or ''} {e.status}" for e in events)
        assert "sk-" not in blob
        assert "nvapi-" not in blob
