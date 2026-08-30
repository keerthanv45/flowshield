"""
Recovery candidate generation.

Turns a RevenueRisk's failure breakdown (+ incident type) into typed
`RecoveryCandidate` options. This module only PROPOSES candidates — the
`RecoveryPolicyEngine` (policy.py) is what actually decides which one,
if any, is allowed and recommended. No guardrail enforcement happens
here; keeping generation and policy separate is deliberate (per the
Phase 4 brief's "Nemotron reasons/recommends, deterministic policy rules
decide").
"""

from __future__ import annotations

from backend.app.services.recovery.schemas import RecoveryActionType, RecoveryCandidate, RevenueRisk, RiskLevel
from ml.health.incidents import Incident, IncidentType

# Which failure reasons map to which action family. A blind RETRY is
# only offered for transient/infrastructure reasons; authentication
# failures get a distinct action (switching method/flow, not repeating
# the same failed attempt); insufficient_funds/bank_declined/unknown are
# deliberately excluded from RETRY entirely (see module docstring in
# revenue_risk.py and the Phase 4 brief's explicit guardrail examples).
RETRY_ELIGIBLE_REASONS = {"timeout", "network_error", "technical_error"}
ALTERNATE_METHOD_ELIGIBLE_REASONS = {"authentication_failed"}
WAIT_ELIGIBLE_REASONS = {"insufficient_funds"}
NO_ACTION_REASONS = {"bank_declined", "unknown"}

RAIL_DEGRADATION_TYPES = {IncidentType.BANK_RAIL_DEGRADATION, IncidentType.REGIONAL_DEGRADATION}


def _amount_for_reasons(revenue_risk: RevenueRisk, reasons: set[str]) -> tuple[int, float, float]:
    """Returns (count, gross_amount, expected_recovered_amount) for the
    breakdown entries matching `reasons`."""
    count = 0
    amount = 0.0
    expected = 0.0
    for entry in revenue_risk.failure_breakdown:
        if entry.failure_reason in reasons:
            count += entry.count
            amount += entry.amount
            expected += entry.amount * entry.assumed_recovery_rate
    return count, amount, expected


def generate_candidates(incident: Incident, revenue_risk: RevenueRisk) -> list[RecoveryCandidate]:
    candidates: list[RecoveryCandidate] = []
    present_reasons = {e.failure_reason for e in revenue_risk.failure_breakdown}

    retry_reasons = RETRY_ELIGIBLE_REASONS & present_reasons
    if retry_reasons:
        count, amount, expected = _amount_for_reasons(revenue_risk, retry_reasons)
        if incident.incident_type in RAIL_DEGRADATION_TYPES:
            candidates.append(
                RecoveryCandidate(
                    action=RecoveryActionType.ROUTE_ALTERNATE_RAIL,
                    reason=(
                        f"Bank/rail-level degradation detected ({incident.incident_type.value}); "
                        f"{count} transient failure(s) (₹{amount:.2f}) may recover via an alternate "
                        "route rather than retrying the same degraded rail"
                    ),
                    eligible_failure_types=sorted(retry_reasons),
                    estimated_recovery=expected,
                    risk=RiskLevel.LOW,
                    priority=1,
                )
            )
        else:
            candidates.append(
                RecoveryCandidate(
                    action=RecoveryActionType.RETRY,
                    reason=(
                        f"{count} transient failure(s) (₹{amount:.2f}) from timeout/network/technical "
                        "reasons are commonly recoverable on retry"
                    ),
                    eligible_failure_types=sorted(retry_reasons),
                    estimated_recovery=expected,
                    risk=RiskLevel.LOW,
                    priority=1,
                )
            )

    alt_method_reasons = ALTERNATE_METHOD_ELIGIBLE_REASONS & present_reasons
    if alt_method_reasons:
        count, amount, expected = _amount_for_reasons(revenue_risk, alt_method_reasons)
        candidates.append(
            RecoveryCandidate(
                action=RecoveryActionType.ALTERNATE_PAYMENT_METHOD,
                reason=(
                    f"{count} authentication failure(s) (₹{amount:.2f}) are unlikely to succeed on a "
                    "blind retry of the same method; offering an alternate payment method may recover them"
                ),
                eligible_failure_types=sorted(alt_method_reasons),
                estimated_recovery=expected,
                risk=RiskLevel.MEDIUM,
                priority=2,
            )
        )

    wait_reasons = WAIT_ELIGIBLE_REASONS & present_reasons
    if wait_reasons:
        count, amount, expected = _amount_for_reasons(revenue_risk, wait_reasons)
        candidates.append(
            RecoveryCandidate(
                action=RecoveryActionType.WAIT_AND_MONITOR,
                reason=(
                    f"{count} insufficient-funds failure(s) (₹{amount:.2f}) are only temporarily "
                    "retryable -- monitor and retry later rather than immediately"
                ),
                eligible_failure_types=sorted(wait_reasons),
                estimated_recovery=expected,
                risk=RiskLevel.LOW,
                priority=3,
            )
        )

    hard_decline_reasons = NO_ACTION_REASONS & present_reasons
    if hard_decline_reasons:
        count, amount, _expected = _amount_for_reasons(revenue_risk, hard_decline_reasons)
        candidates.append(
            RecoveryCandidate(
                action=RecoveryActionType.NO_ACTION,
                reason=(
                    f"{count} hard-decline/unknown failure(s) (₹{amount:.2f}) should not be blindly "
                    "retried -- low genuine recovery likelihood"
                ),
                eligible_failure_types=sorted(hard_decline_reasons),
                estimated_recovery=0.0,
                risk=RiskLevel.HIGH,
                priority=5,
            )
        )

    # Always include an explicit no-action fallback so the policy engine
    # never has an empty candidate list.
    candidates.append(
        RecoveryCandidate(
            action=RecoveryActionType.NO_ACTION,
            reason="Fallback candidate: no recovery action justified by current evidence",
            eligible_failure_types=[],
            estimated_recovery=0.0,
            risk=RiskLevel.LOW,
            priority=99,
        )
    )

    return candidates
