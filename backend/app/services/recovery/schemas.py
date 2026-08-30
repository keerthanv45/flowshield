"""
Structured schemas for FlowShield's Phase 4 revenue-at-risk and recovery
decision layer.

Pipeline: Confirmed Incident -> RCA -> RevenueRisk -> RecoveryCandidates
-> RecoveryPolicyEngine -> RecoveryDecision -> simulated execution ->
RecoveryOutcome.

CRITICAL: everything downstream of `RecoveryDecision` in this phase is
SIMULATED ONLY. No real payment action, no live Razorpay call, no real
money movement anywhere in this module tree.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RecoveryActionType(str, Enum):
    RETRY = "RETRY"
    ALTERNATE_PAYMENT_METHOD = "ALTERNATE_PAYMENT_METHOD"
    ROUTE_ALTERNATE_RAIL = "ROUTE_ALTERNATE_RAIL"
    WAIT_AND_MONITOR = "WAIT_AND_MONITOR"
    NO_ACTION = "NO_ACTION"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FailureBreakdownEntry(BaseModel):
    failure_reason: str
    count: int = Field(..., ge=0)
    amount: float = Field(..., ge=0.0)
    assumed_recovery_rate: float = Field(..., ge=0.0, le=1.0)


class RevenueRisk(BaseModel):
    """Revenue-at-risk estimate for ONE confirmed incident's scope
    (its time window, and its affected dimensions if any). Built from
    ACTUAL synthetic events -- nothing here is invented."""

    model_config = ConfigDict(use_enum_values=False)

    incident_id: str
    transactions_at_risk: int = Field(..., ge=0, description="Count of failed transactions in incident scope")
    gross_amount_at_risk: float = Field(..., ge=0.0, description="Total amount of ALL failed transactions in scope")
    recoverable_transactions: int = Field(
        ..., ge=0, description="Count of failed transactions whose failure reason is genuinely recoverable"
    )
    recoverable_amount: float = Field(
        ..., ge=0.0, description="Amount pool available to recovery actions (genuinely-recoverable reasons only)"
    )
    expected_recovered_amount: float = Field(
        ..., ge=0.0,
        description=(
            "Probability-weighted expectation across ALL failed transactions "
            "using assumed per-reason recovery rates -- an estimate, not a guarantee"
        ),
    )
    failure_breakdown: list[FailureBreakdownEntry] = Field(default_factory=list)


class RecoveryCandidate(BaseModel):
    action: RecoveryActionType
    reason: str
    eligible_failure_types: list[str] = Field(default_factory=list)
    estimated_recovery: float = Field(..., ge=0.0, description="Expected recovered amount if this action is taken")
    risk: RiskLevel
    priority: int = Field(..., description="Lower = higher priority")

    model_config = ConfigDict(use_enum_values=False)


class RecoveryDecision(BaseModel):
    """Deterministic policy output. Nemotron/RCA inform scoring inputs
    but never override the guardrails that produce this."""

    incident_id: str
    recommended_action: RecoveryActionType
    decision_score: float
    expected_recovery_amount: float = Field(..., ge=0.0)
    risk_level: RiskLevel
    reasoning: list[str] = Field(default_factory=list)
    candidates_considered: list[RecoveryCandidate] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list, description="Guardrail rejections/adjustments applied")

    model_config = ConfigDict(use_enum_values=False)


class SimulatedExecution(BaseModel):
    """SIMULATED ONLY. No real payment gateway call is ever made here."""

    action_id: str
    incident_id: str
    action: RecoveryActionType
    eligible_transactions: int = Field(..., ge=0)
    simulated_successes: int = Field(..., ge=0)
    simulated_recovered_amount: float = Field(..., ge=0.0)
    seed: int
    status: str = Field(..., description='Always one of "SIMULATED_COMPLETE" or "SIMULATED_NO_ACTION"')

    model_config = ConfigDict(use_enum_values=False)


class RecoveryOutcome(BaseModel):
    action_id: str
    incident_id: str
    attempted_transactions: int = Field(..., ge=0)
    recovered_transactions: int = Field(..., ge=0)
    attempted_amount: float = Field(..., ge=0.0)
    recovered_amount: float = Field(..., ge=0.0)
    recovery_rate: float = Field(..., ge=0.0, le=1.0)
    status: str
