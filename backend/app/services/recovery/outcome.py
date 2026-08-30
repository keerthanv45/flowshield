"""
Outcome tracking: converts a SimulatedExecution into a RecoveryOutcome
record. This is what a future closed-loop optimization phase would read
back (not implemented here — Phase 4 stops at producing the record).
"""

from __future__ import annotations

from backend.app.services.recovery.schemas import RecoveryDecision, RecoveryOutcome, RevenueRisk, SimulatedExecution
from backend.app.services.recovery.simulator import eligible_entries_for_decision


def build_outcome(
    simulated: SimulatedExecution, decision: RecoveryDecision, revenue_risk: RevenueRisk
) -> RecoveryOutcome:
    eligible_entries = eligible_entries_for_decision(decision, revenue_risk)
    attempted_amount = sum(e.amount for e in eligible_entries)

    recovery_rate = (
        simulated.simulated_successes / simulated.eligible_transactions
        if simulated.eligible_transactions
        else 0.0
    )
    return RecoveryOutcome(
        action_id=simulated.action_id,
        incident_id=simulated.incident_id,
        attempted_transactions=simulated.eligible_transactions,
        recovered_transactions=simulated.simulated_successes,
        attempted_amount=attempted_amount,
        recovered_amount=simulated.simulated_recovered_amount,
        recovery_rate=recovery_rate,
        status=simulated.status,
    )
