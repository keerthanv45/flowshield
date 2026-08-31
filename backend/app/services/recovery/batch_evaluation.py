"""
Phase 6: batch recovery evaluation across the WHOLE synthetic dataset
(all 20,000 events), not scoped to any single incident.

Reuses, unchanged:
  - RevenueRisk / FailureBreakdownEntry schemas (backend.app.services.recovery.schemas)
  - ASSUMED_RECOVERY_RATE / RECOVERABLE_THRESHOLD and the breakdown
    computation (backend.app.services.recovery.revenue_risk)
  - RecoveryCandidate / RecoveryDecision / SimulatedExecution schemas
  - simulate_execution() (backend.app.services.recovery.simulator) --
    called completely unmodified.
  - the failure-reason -> action-family guardrail sets already defined
    in candidates.py / policy.py (RETRY_ELIGIBLE_REASONS,
    ALTERNATE_METHOD_ELIGIBLE_REASONS, WAIT_ELIGIBLE_REASONS,
    NO_ACTION_REASONS / HARD_DECLINE_REASONS) -- imported directly, not
    redefined.

============================================================
WHY THIS ISN'T "JUST CALL RecoveryPolicyEngine.decide() ONCE"
============================================================
`RecoveryPolicyEngine.decide()` is shaped around ONE confirmed incident
with ONE RCA and ONE recommended action. A dataset-wide batch mixes many
failure reasons across many (or no) incidents at once -- there is no
single "the incident" to reason about. So instead of forcing the batch
through the single-decision API, this module partitions the batch's
failed transactions by failure_reason into the SAME action families
Phase 4 already defines, and simulates only the family that Phase 4's
own guardrails always allow to be attempted automatically:

  RETRY / ROUTE (timeout, network_error, technical_error)
      -- system-side, no customer action needed -> SIMULATED (this is
         the "actions_selected" / "retry/routing actions selected" bucket)
  ALTERNATE_PAYMENT_METHOD (authentication_failed)
      -- requires the CUSTOMER to act (switch method / re-auth) -- per
         the Phase 4 brief ("authentication failures may require
         customer action") this is not something a batch job can
         execute automatically, so it is reported as EXCLUDED from this
         batch's automated simulation, not simulated as an attempt.
  WAIT_AND_MONITOR (insufficient_funds)
      -- "temporarily retryable", not immediately -- never simulated as
         an attempt here either (matches Phase 4's simulator, which
         also reports zero eligible transactions for WAIT_AND_MONITOR).
  NO_ACTION / hard declines (bank_declined) and unsupported (unknown)
      -- NEVER retried, per guardrail -- reported as excluded, split
         into two separate counts as the Phase 6 brief requests.

The guardrail RULES (which reasons fall in which bucket) are the exact
same sets Phase 4 already uses -- nothing new is invented here, only
applied across the whole dataset instead of one incident's scope.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.recovery.candidates import (
    ALTERNATE_METHOD_ELIGIBLE_REASONS,
    NO_ACTION_REASONS,
    RETRY_ELIGIBLE_REASONS,
    WAIT_ELIGIBLE_REASONS,
)
from backend.app.services.recovery.revenue_risk import compute_revenue_risk_for_failed
from backend.app.services.recovery.schemas import (
    RecoveryActionType,
    RecoveryCandidate,
    RecoveryDecision,
    RiskLevel,
    SimulatedExecution,
)
from backend.app.services.recovery.simulator import DEFAULT_SEED, simulate_execution

BATCH_SCOPE_ID = "BATCH_ALL_TRANSACTIONS"

HARD_DECLINE_REASON = "bank_declined"
UNSUPPORTED_REASON = "unknown"


class GuardrailSummary(BaseModel):
    hard_declines_excluded_count: int = Field(..., ge=0)
    hard_declines_excluded_amount: float = Field(..., ge=0.0)
    auth_failures_excluded_count: int = Field(..., ge=0)
    auth_failures_excluded_amount: float = Field(..., ge=0.0)
    unsupported_failures_excluded_count: int = Field(..., ge=0)
    unsupported_failures_excluded_amount: float = Field(..., ge=0.0)
    insufficient_funds_deferred_count: int = Field(
        ..., ge=0, description="WAIT_AND_MONITOR bucket -- not excluded, not simulated as an attempt"
    )
    insufficient_funds_deferred_amount: float = Field(..., ge=0.0)
    retry_routing_actions_selected_count: int = Field(..., ge=0)
    retry_routing_actions_selected_amount: float = Field(..., ge=0.0)


class BatchEvaluationResult(BaseModel):
    total_transactions: int = Field(..., ge=0)
    failed_transactions: int = Field(..., ge=0)
    gross_failed_amount: float = Field(..., ge=0.0)
    revenue_at_risk: float = Field(..., ge=0.0, description="Alias of gross_failed_amount for report clarity")
    recoverable_transactions: int = Field(..., ge=0)
    recoverable_amount: float = Field(..., ge=0.0)
    expected_recovery_amount: float = Field(..., ge=0.0)

    actions_selected: int = Field(..., ge=0, description="Count of failed transactions given an automated action")
    simulated_attempts: int = Field(..., ge=0)
    simulated_recovered_transactions: int = Field(..., ge=0)
    simulated_recovered_amount: float = Field(..., ge=0.0)
    recovery_rate: float = Field(..., ge=0.0, le=1.0, description="simulated_recovered / simulated_attempts")
    revenue_recovery_rate: float = Field(
        ..., ge=0.0, le=1.0, description="simulated_recovered_amount / revenue_at_risk"
    )

    guardrails: GuardrailSummary
    seed: int
    status: str

    model_config = ConfigDict(use_enum_values=False)


def _amount_count(df: pd.DataFrame, reasons: set[str]) -> tuple[int, float]:
    sub = df.loc[df["failure_reason"].isin(reasons)]
    return int(len(sub)), float(sub["amount"].sum())


def run_batch_evaluation(events_df: pd.DataFrame, seed: int = DEFAULT_SEED) -> BatchEvaluationResult:
    """Evaluate recovery across every failed transaction in `events_df`
    (the full synthetic dataset by default -- see module docstring for
    exactly how the batch is selected: it is NOT incident-scoped, it is
    every row with status == 'failed' in the dataset passed in)."""
    total_transactions = int(len(events_df))

    if total_transactions == 0 or "status" not in events_df.columns:
        empty_guardrails = GuardrailSummary(
            hard_declines_excluded_count=0, hard_declines_excluded_amount=0.0,
            auth_failures_excluded_count=0, auth_failures_excluded_amount=0.0,
            unsupported_failures_excluded_count=0, unsupported_failures_excluded_amount=0.0,
            insufficient_funds_deferred_count=0, insufficient_funds_deferred_amount=0.0,
            retry_routing_actions_selected_count=0, retry_routing_actions_selected_amount=0.0,
        )
        return BatchEvaluationResult(
            total_transactions=total_transactions, failed_transactions=0, gross_failed_amount=0.0,
            revenue_at_risk=0.0, recoverable_transactions=0, recoverable_amount=0.0,
            expected_recovery_amount=0.0, actions_selected=0, simulated_attempts=0,
            simulated_recovered_transactions=0, simulated_recovered_amount=0.0,
            recovery_rate=0.0, revenue_recovery_rate=0.0, guardrails=empty_guardrails,
            seed=seed, status="SIMULATED_NO_ACTION",
        )

    failed = events_df.loc[events_df["status"] == "failed"]

    revenue_risk = compute_revenue_risk_for_failed(BATCH_SCOPE_ID, failed)

    hard_count, hard_amount = _amount_count(failed, {HARD_DECLINE_REASON})
    unsupported_count, unsupported_amount = _amount_count(failed, {UNSUPPORTED_REASON})
    auth_count, auth_amount = _amount_count(failed, ALTERNATE_METHOD_ELIGIBLE_REASONS)
    wait_count, wait_amount = _amount_count(failed, WAIT_ELIGIBLE_REASONS)
    retry_count, retry_amount = _amount_count(failed, RETRY_ELIGIBLE_REASONS)

    # Sanity: every failed transaction falls into exactly one of the
    # five buckets above (RETRY_ELIGIBLE_REASONS | ALTERNATE_METHOD... |
    # WAIT_ELIGIBLE_REASONS | NO_ACTION_REASONS covers every
    # ASSUMED_RECOVERY_RATE key from Phase 1's schema, and
    # NO_ACTION_REASONS == {bank_declined, unknown} is split into the
    # two separate counts above).
    assert hard_count + unsupported_count == len(failed.loc[failed["failure_reason"].isin(NO_ACTION_REASONS)])

    retry_candidate = RecoveryCandidate(
        action=RecoveryActionType.RETRY,
        reason="Batch-wide automated retry/route for transient failure reasons (timeout, network_error, technical_error)",
        eligible_failure_types=sorted(RETRY_ELIGIBLE_REASONS),
        estimated_recovery=sum(
            e.amount * e.assumed_recovery_rate
            for e in revenue_risk.failure_breakdown
            if e.failure_reason in RETRY_ELIGIBLE_REASONS
        ),
        risk=RiskLevel.LOW,
        priority=1,
    )
    batch_decision = RecoveryDecision(
        incident_id=BATCH_SCOPE_ID,
        recommended_action=RecoveryActionType.RETRY,
        decision_score=retry_candidate.estimated_recovery,
        expected_recovery_amount=retry_candidate.estimated_recovery,
        risk_level=RiskLevel.LOW,
        reasoning=["Batch evaluation: automated retry/route applied only to system-side-recoverable failure reasons."],
        candidates_considered=[retry_candidate],
        policy_notes=[
            f"Excluded {hard_count} hard-decline transaction(s) (bank_declined)",
            f"Excluded {auth_count} authentication-failure transaction(s) (requires customer action)",
            f"Excluded {unsupported_count} unsupported/unknown-reason transaction(s)",
            f"Deferred {wait_count} insufficient-funds transaction(s) to WAIT_AND_MONITOR (not simulated)",
        ],
    )

    simulated: SimulatedExecution = simulate_execution(batch_decision, revenue_risk, seed=seed)

    recovery_rate = (
        simulated.simulated_successes / simulated.eligible_transactions
        if simulated.eligible_transactions
        else 0.0
    )
    revenue_recovery_rate = (
        simulated.simulated_recovered_amount / revenue_risk.gross_amount_at_risk
        if revenue_risk.gross_amount_at_risk
        else 0.0
    )

    guardrails = GuardrailSummary(
        hard_declines_excluded_count=hard_count,
        hard_declines_excluded_amount=hard_amount,
        auth_failures_excluded_count=auth_count,
        auth_failures_excluded_amount=auth_amount,
        unsupported_failures_excluded_count=unsupported_count,
        unsupported_failures_excluded_amount=unsupported_amount,
        insufficient_funds_deferred_count=wait_count,
        insufficient_funds_deferred_amount=wait_amount,
        retry_routing_actions_selected_count=retry_count,
        retry_routing_actions_selected_amount=retry_amount,
    )

    return BatchEvaluationResult(
        total_transactions=total_transactions,
        failed_transactions=revenue_risk.transactions_at_risk,
        gross_failed_amount=revenue_risk.gross_amount_at_risk,
        revenue_at_risk=revenue_risk.gross_amount_at_risk,
        recoverable_transactions=revenue_risk.recoverable_transactions,
        recoverable_amount=revenue_risk.recoverable_amount,
        expected_recovery_amount=revenue_risk.expected_recovered_amount,
        actions_selected=retry_count,
        simulated_attempts=simulated.eligible_transactions,
        simulated_recovered_transactions=simulated.simulated_successes,
        simulated_recovered_amount=simulated.simulated_recovered_amount,
        recovery_rate=recovery_rate,
        revenue_recovery_rate=revenue_recovery_rate,
        guardrails=guardrails,
        seed=seed,
        status=simulated.status,
    )
