"""
RecoveryPolicyEngine: deterministic decision layer.

Takes IncidentEvidence + RCAResult + RevenueRisk + candidates, and
returns ONE RecoveryDecision. Nemotron's RCA is used only as an input
signal (its `confidence` gates the low-evidence guardrail) — it never
selects or overrides the recommended action. All guardrails below are
fixed, documented rules, not LLM output.

============================================================
GUARDRAILS (applied in order; a triggered guardrail short-circuits scoring)
============================================================
  1. LOW CONFIDENCE: if the RCA's self-reported confidence is below
     `MIN_RCA_CONFIDENCE` (0.3), the evidence is treated as insufficient
     to act on -> WAIT_AND_MONITOR, regardless of candidates.
  2. NEGLIGIBLE RECOVERY: if RevenueRisk.expected_recovered_amount is
     below `MIN_EXPECTED_RECOVERY` (₹50), no action is worth the
     operational cost -> NO_ACTION.
  3. NO ELIGIBLE CANDIDATES: after filtering out any candidate whose
     `eligible_failure_types` are ENTIRELY hard-decline reasons
     (bank_declined/unknown — never retried, per the Phase 4 brief) and
     any RETRY-typed candidate that includes authentication_failed
     (auth failures never get a blind retry, only
     ALTERNATE_PAYMENT_METHOD) -> NO_ACTION if nothing remains.

============================================================
SCORING (only reached if no guardrail fired)
============================================================
  score = expected_recovery_amount        (revenue signal, dominant term)
        + scope_bonus                     (+200 if incident spans >1 window --
                                             larger incidents get modest priority)
        + recoverability_bonus            (+100 per LOW risk_level, +50 MEDIUM, +0 HIGH)
        - priority_penalty                (candidate.priority * 10, lower priority number wins ties)

  ROUTE_ALTERNATE_RAIL gets an additional +150 when the incident_type is
  BANK_RAIL_DEGRADATION or REGIONAL_DEGRADATION (brief: "prefer alternate
  routing during bank/rail degradation" — implemented as a scoring
  preference, not a hard override, so it can still lose to a
  dramatically higher-value candidate).

The highest-scoring remaining candidate is `recommended_action`. This is
a deterministic function of documented inputs — no randomness, no LLM
call, fully reproducible for the same evidence/candidates.
"""

from __future__ import annotations

from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult
from backend.app.services.recovery.schemas import (
    RecoveryActionType,
    RecoveryCandidate,
    RecoveryDecision,
    RevenueRisk,
    RiskLevel,
)
from ml.health.incidents import IncidentType

MIN_RCA_CONFIDENCE = 0.3
MIN_EXPECTED_RECOVERY = 50.0

HARD_DECLINE_REASONS = {"bank_declined", "unknown"}
RAIL_DEGRADATION_TYPES = {IncidentType.BANK_RAIL_DEGRADATION, IncidentType.REGIONAL_DEGRADATION}

RISK_BONUS = {RiskLevel.LOW: 100.0, RiskLevel.MEDIUM: 50.0, RiskLevel.HIGH: 0.0}


def _no_action_decision(
    incident_id: str, candidates: list[RecoveryCandidate], reasoning: list[str], policy_notes: list[str]
) -> RecoveryDecision:
    return RecoveryDecision(
        incident_id=incident_id,
        recommended_action=RecoveryActionType.NO_ACTION,
        decision_score=0.0,
        expected_recovery_amount=0.0,
        risk_level=RiskLevel.LOW,
        reasoning=reasoning,
        candidates_considered=candidates,
        policy_notes=policy_notes,
    )


def _wait_and_monitor_decision(
    incident_id: str, candidates: list[RecoveryCandidate], reasoning: list[str], policy_notes: list[str]
) -> RecoveryDecision:
    return RecoveryDecision(
        incident_id=incident_id,
        recommended_action=RecoveryActionType.WAIT_AND_MONITOR,
        decision_score=0.0,
        expected_recovery_amount=0.0,
        risk_level=RiskLevel.LOW,
        reasoning=reasoning,
        candidates_considered=candidates,
        policy_notes=policy_notes,
    )


class RecoveryPolicyEngine:
    def decide(
        self,
        evidence: IncidentEvidence,
        rca: RCAResult,
        revenue_risk: RevenueRisk,
        candidates: list[RecoveryCandidate],
    ) -> RecoveryDecision:
        policy_notes: list[str] = []

        if rca.confidence < MIN_RCA_CONFIDENCE:
            policy_notes.append(
                f"RCA confidence {rca.confidence:.2f} below threshold {MIN_RCA_CONFIDENCE} "
                "-- insufficient evidence to act"
            )
            return _wait_and_monitor_decision(
                evidence.incident_id, candidates,
                reasoning=["Evidence confidence too low to justify a recovery action; monitoring only."],
                policy_notes=policy_notes,
            )

        if revenue_risk.expected_recovered_amount < MIN_EXPECTED_RECOVERY:
            policy_notes.append(
                f"Expected recovered amount Rs.{revenue_risk.expected_recovered_amount:.2f} below "
                f"threshold Rs.{MIN_EXPECTED_RECOVERY:.2f}"
            )
            return _no_action_decision(
                evidence.incident_id, candidates,
                reasoning=["Expected recovery is negligible relative to the cost of acting."],
                policy_notes=policy_notes,
            )

        allowed: list[RecoveryCandidate] = []
        for c in candidates:
            eligible = set(c.eligible_failure_types)
            if eligible and eligible.issubset(HARD_DECLINE_REASONS):
                policy_notes.append(f"Rejected {c.action.value}: eligible types are all hard-decline reasons")
                continue
            if c.action == RecoveryActionType.RETRY and "authentication_failed" in eligible:
                policy_notes.append(
                    f"Rejected {c.action.value}: authentication failures require "
                    "ALTERNATE_PAYMENT_METHOD, not blind retry"
                )
                continue
            allowed.append(c)

        scoreable = [c for c in allowed if c.action != RecoveryActionType.NO_ACTION] or allowed

        if not scoreable:
            policy_notes.append("No candidates survived guardrail filtering")
            return _no_action_decision(
                evidence.incident_id, candidates,
                reasoning=["No recovery action passed policy guardrails."],
                policy_notes=policy_notes,
            )

        best = max(scoreable, key=lambda c: self._score(c, evidence))
        best_score = self._score(best, evidence)

        reasoning = [
            f"Selected {best.action.value}: {best.reason}",
            f"Decision score {best_score:.2f}, expected recovery Rs.{best.estimated_recovery:.2f}",
        ]
        if best.action == RecoveryActionType.ROUTE_ALTERNATE_RAIL:
            reasoning.append("Alternate routing preferred during bank/rail-level degradation.")

        return RecoveryDecision(
            incident_id=evidence.incident_id,
            recommended_action=best.action,
            decision_score=best_score,
            expected_recovery_amount=best.estimated_recovery,
            risk_level=best.risk,
            reasoning=reasoning,
            candidates_considered=candidates,
            policy_notes=policy_notes,
        )

    def _score(self, candidate: RecoveryCandidate, evidence: IncidentEvidence) -> float:
        score = candidate.estimated_recovery
        if evidence.n_windows > 1:
            score += 200.0
        score += RISK_BONUS[candidate.risk]
        score -= candidate.priority * 10.0
        if candidate.action == RecoveryActionType.ROUTE_ALTERNATE_RAIL and evidence.incident_type in RAIL_DEGRADATION_TYPES:
            score += 150.0
        return score
