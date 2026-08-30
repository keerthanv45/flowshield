from datetime import datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult, ReasoningSource
from backend.app.services.recovery.candidates import generate_candidates
from backend.app.services.recovery.outcome import build_outcome
from backend.app.services.recovery.policy import MIN_EXPECTED_RECOVERY, MIN_RCA_CONFIDENCE, RecoveryPolicyEngine
from backend.app.services.recovery.revenue_risk import ASSUMED_RECOVERY_RATE, calculate_revenue_risk
from backend.app.services.recovery.schemas import (
    FailureBreakdownEntry,
    RecoveryActionType,
    RecoveryCandidate,
    RevenueRisk,
    RiskLevel,
)
from backend.app.services.recovery.simulator import simulate_execution
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity


def make_incident(**overrides) -> Incident:
    base = dict(
        incident_id="inc_test_0001",
        detected_at=datetime(2026, 8, 3, 2, 0),
        window_start=datetime(2026, 8, 3, 2, 0),
        window_end=datetime(2026, 8, 3, 2, 15),
        severity=Severity.CRITICAL,
        incident_type=IncidentType.BANK_RAIL_DEGRADATION,
        anomaly_score=0.87,
        health_score=22.5,
        affected_dimensions=[{"bank": "HDFC", "payment_method": "UPI"}],
        signals=["SUCCESS_RATE_DEGRADATION", "BANK_CONCENTRATION"],
        evidence=["Overall success rate: 0.50 (baseline: 0.90, delta: -0.40)"],
        status=IncidentStatus.CONFIRMED,
        n_windows=1,
    )
    base.update(overrides)
    return Incident(**base)


def make_events(rows: list[dict]) -> pd.DataFrame:
    defaults = dict(bank="HDFC", payment_method="UPI", region="KA")
    full = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        full.append(row)
    return pd.DataFrame(full)


def make_rca(confidence: float = 0.7) -> RCAResult:
    return RCAResult(
        root_cause="Bank rail degradation", confidence=confidence, explanation="x",
        supporting_evidence=[], affected_scope=[], recommended_actions=[],
        source=ReasoningSource.MOCK,
    )


class TestRevenueRisk:
    def test_total_at_risk_calculation(self):
        incident = make_incident()
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=100.0, failure_reason="network_error"),
                dict(timestamp=datetime(2026, 8, 3, 2, 6), status="failed", amount=200.0, failure_reason="bank_declined"),
                dict(timestamp=datetime(2026, 8, 3, 2, 7), status="success", amount=50.0, failure_reason=None),
            ]
        )
        risk = calculate_revenue_risk(incident, events)
        assert risk.transactions_at_risk == 2
        assert risk.gross_amount_at_risk == pytest.approx(300.0)

    def test_recoverable_vs_non_recoverable_split(self):
        incident = make_incident()
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=100.0, failure_reason="timeout"),
                dict(timestamp=datetime(2026, 8, 3, 2, 6), status="failed", amount=200.0, failure_reason="bank_declined"),
            ]
        )
        risk = calculate_revenue_risk(incident, events)
        assert risk.recoverable_transactions == 1
        assert risk.recoverable_amount == pytest.approx(100.0)
        assert risk.gross_amount_at_risk == pytest.approx(300.0)
        # expected_recovered_amount weights ALL reasons, including low-rate ones
        expected = 100.0 * ASSUMED_RECOVERY_RATE["timeout"] + 200.0 * ASSUMED_RECOVERY_RATE["bank_declined"]
        assert risk.expected_recovered_amount == pytest.approx(expected)

    def test_zero_risk_case(self):
        incident = make_incident()
        events = make_events(
            [dict(timestamp=datetime(2026, 8, 3, 2, 5), status="success", amount=100.0, failure_reason=None)]
        )
        risk = calculate_revenue_risk(incident, events)
        assert risk.transactions_at_risk == 0
        assert risk.gross_amount_at_risk == 0.0
        assert risk.expected_recovered_amount == 0.0

    def test_scoped_to_affected_dimensions(self):
        incident = make_incident(affected_dimensions=[{"bank": "HDFC", "payment_method": "UPI"}])
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=100.0, failure_reason="timeout", bank="HDFC", payment_method="UPI"),
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=999.0, failure_reason="timeout", bank="SBI", payment_method="CARD"),
            ]
        )
        risk = calculate_revenue_risk(incident, events)
        assert risk.gross_amount_at_risk == pytest.approx(100.0)

    def test_scoped_to_time_window_only_when_no_dimensions(self):
        incident = make_incident(incident_type=IncidentType.LATENCY_SPIKE, affected_dimensions=[])
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=100.0, failure_reason="timeout", bank="HDFC"),
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=200.0, failure_reason="timeout", bank="SBI"),
                dict(timestamp=datetime(2026, 8, 3, 3, 0), status="failed", amount=999.0, failure_reason="timeout", bank="HDFC"),  # outside window
            ]
        )
        risk = calculate_revenue_risk(incident, events)
        assert risk.gross_amount_at_risk == pytest.approx(300.0)


class TestCandidates:
    def test_timeout_produces_retry_candidate(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        risk = RevenueRisk(
            incident_id="x", transactions_at_risk=1, gross_amount_at_risk=100.0,
            recoverable_transactions=1, recoverable_amount=100.0, expected_recovered_amount=55.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=100.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        assert any(c.action == RecoveryActionType.RETRY for c in candidates)

    def test_network_error_produces_retry_candidate(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        risk = RevenueRisk(
            incident_id="x", transactions_at_risk=1, gross_amount_at_risk=100.0,
            recoverable_transactions=1, recoverable_amount=100.0, expected_recovered_amount=55.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="network_error", count=1, amount=100.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        assert any(c.action == RecoveryActionType.RETRY for c in candidates)

    def test_bank_rail_degradation_produces_alternate_routing(self):
        incident = make_incident(incident_type=IncidentType.BANK_RAIL_DEGRADATION)
        risk = RevenueRisk(
            incident_id="x", transactions_at_risk=1, gross_amount_at_risk=100.0,
            recoverable_transactions=1, recoverable_amount=100.0, expected_recovered_amount=55.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=100.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        assert any(c.action == RecoveryActionType.ROUTE_ALTERNATE_RAIL for c in candidates)
        assert not any(c.action == RecoveryActionType.RETRY for c in candidates)

    def test_hard_decline_does_not_produce_retry(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        risk = RevenueRisk(
            incident_id="x", transactions_at_risk=1, gross_amount_at_risk=100.0,
            recoverable_transactions=0, recoverable_amount=0.0, expected_recovered_amount=3.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="bank_declined", count=1, amount=100.0, assumed_recovery_rate=0.03)],
        )
        candidates = generate_candidates(incident, risk)
        assert not any(c.action == RecoveryActionType.RETRY for c in candidates)
        assert any(c.action == RecoveryActionType.NO_ACTION and "bank_declined" in c.eligible_failure_types for c in candidates)

    def test_authentication_failure_produces_alternate_method_not_retry(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        risk = RevenueRisk(
            incident_id="x", transactions_at_risk=1, gross_amount_at_risk=100.0,
            recoverable_transactions=1, recoverable_amount=100.0, expected_recovered_amount=25.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="authentication_failed", count=1, amount=100.0, assumed_recovery_rate=0.25)],
        )
        candidates = generate_candidates(incident, risk)
        actions = [c.action for c in candidates]
        assert RecoveryActionType.ALTERNATE_PAYMENT_METHOD in actions
        assert RecoveryActionType.RETRY not in actions


class TestPolicyEngine:
    def test_timeout_retry_allowed_and_scored(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[], n_windows=1)
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=200.0,
            recoverable_transactions=1, recoverable_amount=200.0, expected_recovered_amount=110.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=200.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        assert decision.recommended_action == RecoveryActionType.RETRY

    def test_bank_rail_prefers_alternate_routing(self):
        incident = make_incident(incident_type=IncidentType.BANK_RAIL_DEGRADATION, n_windows=2)
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=200.0,
            recoverable_transactions=1, recoverable_amount=200.0, expected_recovered_amount=110.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=200.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        assert decision.recommended_action == RecoveryActionType.ROUTE_ALTERNATE_RAIL

    def test_hard_decline_never_recommends_retry(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=2000.0,
            recoverable_transactions=0, recoverable_amount=0.0, expected_recovered_amount=60.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="bank_declined", count=1, amount=2000.0, assumed_recovery_rate=0.03)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        assert decision.recommended_action != RecoveryActionType.RETRY
        assert decision.recommended_action == RecoveryActionType.NO_ACTION

    def test_authentication_failure_gets_alternate_method_action(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=1000.0,
            recoverable_transactions=1, recoverable_amount=1000.0, expected_recovered_amount=250.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="authentication_failed", count=1, amount=1000.0, assumed_recovery_rate=0.25)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        assert decision.recommended_action == RecoveryActionType.ALTERNATE_PAYMENT_METHOD

    def test_weak_evidence_confidence_forces_wait(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=1000.0,
            recoverable_transactions=1, recoverable_amount=1000.0, expected_recovered_amount=550.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=1000.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        low_confidence_rca = make_rca(MIN_RCA_CONFIDENCE - 0.05)
        decision = RecoveryPolicyEngine().decide(evidence, low_confidence_rca, risk, candidates)
        assert decision.recommended_action == RecoveryActionType.WAIT_AND_MONITOR

    def test_negligible_recovery_forces_no_action(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=5.0,
            recoverable_transactions=1, recoverable_amount=5.0,
            expected_recovered_amount=MIN_EXPECTED_RECOVERY - 1,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=5.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.9), risk, candidates)
        assert decision.recommended_action == RecoveryActionType.NO_ACTION


class TestDecisionOutput:
    def test_deterministic_same_inputs_same_decision(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=200.0,
            recoverable_transactions=1, recoverable_amount=200.0, expected_recovered_amount=110.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=200.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        engine = RecoveryPolicyEngine()
        d1 = engine.decide(evidence, make_rca(0.8), risk, candidates)
        d2 = engine.decide(evidence, make_rca(0.8), risk, candidates)
        assert d1.recommended_action == d2.recommended_action
        assert d1.decision_score == d2.decision_score

    def test_expected_recovery_amount_matches_chosen_candidate(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=200.0,
            recoverable_transactions=1, recoverable_amount=200.0, expected_recovered_amount=110.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=200.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        chosen = next(c for c in decision.candidates_considered if c.action == decision.recommended_action)
        assert decision.expected_recovery_amount == pytest.approx(chosen.estimated_recovery)


class TestSimulator:
    def test_deterministic_seed_reproducible(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=10, gross_amount_at_risk=1000.0,
            recoverable_transactions=10, recoverable_amount=1000.0, expected_recovered_amount=550.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=10, amount=1000.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        sim1 = simulate_execution(decision, risk, seed=42)
        sim2 = simulate_execution(decision, risk, seed=42)
        assert sim1.simulated_successes == sim2.simulated_successes
        assert sim1.simulated_recovered_amount == sim2.simulated_recovered_amount

    def test_status_labeled_simulated(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=5, gross_amount_at_risk=500.0,
            recoverable_transactions=5, recoverable_amount=500.0, expected_recovered_amount=275.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=5, amount=500.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        sim = simulate_execution(decision, risk)
        assert "SIMULATED" in sim.status

    def test_no_action_produces_zero_eligible(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=1, gross_amount_at_risk=5.0,
            recoverable_transactions=1, recoverable_amount=5.0,
            expected_recovered_amount=MIN_EXPECTED_RECOVERY - 1,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=1, amount=5.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.9), risk, candidates)
        sim = simulate_execution(decision, risk)
        assert sim.eligible_transactions == 0
        assert sim.status == "SIMULATED_NO_ACTION"

    def test_outcome_calculation(self):
        incident = make_incident(incident_type=IncidentType.ISOLATED_FAILURES, affected_dimensions=[])
        evidence = IncidentEvidence.from_incident(incident)
        risk = RevenueRisk(
            incident_id=incident.incident_id, transactions_at_risk=10, gross_amount_at_risk=1000.0,
            recoverable_transactions=10, recoverable_amount=1000.0, expected_recovered_amount=550.0,
            failure_breakdown=[FailureBreakdownEntry(failure_reason="timeout", count=10, amount=1000.0, assumed_recovery_rate=0.55)],
        )
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        sim = simulate_execution(decision, risk, seed=42)
        outcome = build_outcome(sim, decision, risk)
        assert outcome.attempted_transactions == 10
        assert outcome.attempted_amount == pytest.approx(1000.0)
        assert outcome.recovery_rate == pytest.approx(
            outcome.recovered_transactions / outcome.attempted_transactions
        )


class TestSchemaValidation:
    def test_revenue_risk_rejects_negative_amount(self):
        with pytest.raises(ValidationError):
            RevenueRisk(
                incident_id="x", transactions_at_risk=1, gross_amount_at_risk=-5.0,
                recoverable_transactions=0, recoverable_amount=0.0, expected_recovered_amount=0.0,
            )

    def test_decision_requires_valid_action_enum(self):
        with pytest.raises(ValidationError):
            RecoveryCandidate(
                action="NOT_A_REAL_ACTION", reason="x", eligible_failure_types=[],
                estimated_recovery=0.0, risk=RiskLevel.LOW, priority=1,
            )


class TestIntegration:
    def test_confirmed_incident_produces_valid_recovery_decision(self):
        incident = make_incident(incident_type=IncidentType.LATENCY_SPIKE, affected_dimensions=[], n_windows=3)
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 3, 2, 5), status="failed", amount=300.0, failure_reason="timeout"),
                dict(timestamp=datetime(2026, 8, 3, 2, 6), status="failed", amount=150.0, failure_reason="network_error"),
                dict(timestamp=datetime(2026, 8, 3, 2, 7), status="success", amount=100.0, failure_reason=None),
            ]
        )
        evidence = IncidentEvidence.from_incident(incident)
        risk = calculate_revenue_risk(incident, events)
        candidates = generate_candidates(incident, risk)
        decision = RecoveryPolicyEngine().decide(evidence, make_rca(0.8), risk, candidates)
        sim = simulate_execution(decision, risk)
        outcome = build_outcome(sim, decision, risk)

        assert decision.incident_id == incident.incident_id
        assert outcome.incident_id == incident.incident_id
        assert 0.0 <= outcome.recovery_rate <= 1.0
