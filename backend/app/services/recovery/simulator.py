"""
Simulated recovery execution.

SIMULATED ONLY. No real payment gateway is called anywhere in this
module. Given a RecoveryDecision + RevenueRisk, deterministically
"executes" (simulates) the recommended action against the eligible
transaction pool using a fixed random seed, so results are reproducible.

WAIT_AND_MONITOR and NO_ACTION never simulate any transaction attempts
(there is nothing to execute) — they report zero eligible/attempted
transactions and a distinct `status`, so a caller can never mistake
"decided not to act" for "attempted and failed".
"""

from __future__ import annotations

import random

from backend.app.services.recovery.schemas import RecoveryActionType, RecoveryDecision, RevenueRisk, SimulatedExecution

DEFAULT_SEED = 42


def eligible_entries_for_decision(decision: RecoveryDecision, revenue_risk: RevenueRisk):
    """Failure-breakdown entries eligible under the decision's chosen
    candidate. Shared by the simulator and outcome-building so both stay
    consistent about which transactions "attempted_amount" refers to."""
    if decision.recommended_action in (RecoveryActionType.NO_ACTION, RecoveryActionType.WAIT_AND_MONITOR):
        return []
    best_candidate = next(
        (c for c in decision.candidates_considered if c.action == decision.recommended_action),
        None,
    )
    eligible_types = set(best_candidate.eligible_failure_types) if best_candidate else set()
    return [e for e in revenue_risk.failure_breakdown if e.failure_reason in eligible_types]


def simulate_execution(
    decision: RecoveryDecision,
    revenue_risk: RevenueRisk,
    seed: int = DEFAULT_SEED,
) -> SimulatedExecution:
    action_id = f"sim_{revenue_risk.incident_id}_{decision.recommended_action.value}"

    if decision.recommended_action in (RecoveryActionType.NO_ACTION, RecoveryActionType.WAIT_AND_MONITOR):
        return SimulatedExecution(
            action_id=action_id,
            incident_id=revenue_risk.incident_id,
            action=decision.recommended_action,
            eligible_transactions=0,
            simulated_successes=0,
            simulated_recovered_amount=0.0,
            seed=seed,
            status="SIMULATED_NO_ACTION",
        )

    eligible_entries = eligible_entries_for_decision(decision, revenue_risk)
    eligible_transactions = sum(e.count for e in eligible_entries)

    rng = random.Random(seed)
    simulated_successes = 0
    simulated_recovered_amount = 0.0

    for entry in eligible_entries:
        avg_amount = entry.amount / entry.count if entry.count else 0.0
        for _ in range(entry.count):
            if rng.random() < entry.assumed_recovery_rate:
                simulated_successes += 1
                simulated_recovered_amount += avg_amount

    return SimulatedExecution(
        action_id=action_id,
        incident_id=revenue_risk.incident_id,
        action=decision.recommended_action,
        eligible_transactions=eligible_transactions,
        simulated_successes=simulated_successes,
        simulated_recovered_amount=round(simulated_recovered_amount, 2),
        seed=seed,
        status="SIMULATED_COMPLETE",
    )
