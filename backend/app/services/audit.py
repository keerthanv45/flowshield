"""
Phase 7: audit trail — a structured, human-readable record of the 7
pipeline stages already computed by Phases 2-4. This module invents NO
new business logic and NO new facts: every value in an `AuditEvent`
comes directly from an object the pipeline already produced (Incident,
RCAResult, RevenueRisk, RecoveryDecision, SimulatedExecution,
RecoveryOutcome). Its only job is formatting/sequencing those existing
outputs into a readable trail.

TIMESTAMPS: this project has no real event-processing telemetry (each
pipeline run is a synchronous, in-process function call chain, not a
distributed system with per-step wall-clock logging). Per the Phase 7
brief ("use deterministic/generated timestamps if the system has no
real event timestamps"), each event's timestamp is
`incident.detected_at + timedelta(seconds=order)` — deterministic,
ordered, and clearly derived from the one real timestamp Phase 2
recorded (`detected_at`), not fabricated wall-clock data. This is
explicitly a display ordering aid, not a claim about real processing
time.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.reasoning.schemas import RCAResult
from backend.app.services.recovery.schemas import RecoveryDecision, RecoveryOutcome, RevenueRisk, SimulatedExecution
from ml.health.incidents import Incident


class AuditEventType(str, Enum):
    INCIDENT_DETECTED = "INCIDENT_DETECTED"
    RCA_COMPLETED = "RCA_COMPLETED"
    REVENUE_RISK_CALCULATED = "REVENUE_RISK_CALCULATED"
    RECOVERY_POLICY_EVALUATED = "RECOVERY_POLICY_EVALUATED"
    GUARDRAILS_CHECKED = "GUARDRAILS_CHECKED"
    RECOVERY_SIMULATED = "RECOVERY_SIMULATED"
    OUTCOME_RECORDED = "OUTCOME_RECORDED"


class AuditEvent(BaseModel):
    order: int = Field(..., ge=1, le=7)
    event_type: AuditEventType
    timestamp: str
    message: str
    value: str | None = None
    status: str

    model_config = ConfigDict(use_enum_values=False)


def build_audit_trail(
    incident: Incident,
    rca: RCAResult,
    revenue_risk: RevenueRisk,
    decision: RecoveryDecision,
    simulated: SimulatedExecution,
    outcome: RecoveryOutcome,
) -> list[AuditEvent]:
    """Every field used below is read directly from an object the
    pipeline already computed — no new calculation happens here."""

    def ts(step: int) -> str:
        return (incident.detected_at + timedelta(seconds=step)).isoformat()

    events = [
        AuditEvent(
            order=1,
            event_type=AuditEventType.INCIDENT_DETECTED,
            timestamp=ts(1),
            message=f"{incident.incident_type.value} detected, {incident.severity.value} severity",
            value=f"health_score={incident.health_score:.1f}",
            status=incident.status.value,
        ),
        AuditEvent(
            order=2,
            event_type=AuditEventType.RCA_COMPLETED,
            timestamp=ts(2),
            message=rca.root_cause,
            value=f"confidence={rca.confidence:.2f}, source={rca.source.value}",
            status="COMPLETED",
        ),
        AuditEvent(
            order=3,
            event_type=AuditEventType.REVENUE_RISK_CALCULATED,
            timestamp=ts(3),
            message=f"{revenue_risk.transactions_at_risk} transaction(s) at risk",
            value=f"gross_amount_at_risk={revenue_risk.gross_amount_at_risk:.2f}",
            status="COMPLETED",
        ),
        AuditEvent(
            order=4,
            event_type=AuditEventType.RECOVERY_POLICY_EVALUATED,
            timestamp=ts(4),
            message=f"Recommended action: {decision.recommended_action.value}",
            value=f"decision_score={decision.decision_score:.2f}",
            status=decision.risk_level.value,
        ),
        AuditEvent(
            order=5,
            event_type=AuditEventType.GUARDRAILS_CHECKED,
            timestamp=ts(5),
            message=(
                f"{len(decision.policy_notes)} guardrail note(s) applied"
                if decision.policy_notes
                else "No guardrail adjustments needed"
            ),
            value="; ".join(decision.policy_notes) if decision.policy_notes else None,
            status="ADJUSTED" if decision.policy_notes else "PASS",
        ),
        AuditEvent(
            order=6,
            event_type=AuditEventType.RECOVERY_SIMULATED,
            timestamp=ts(6),
            message=f"Simulated {simulated.eligible_transactions} eligible transaction(s)",
            value=f"simulated_recovered_amount={simulated.simulated_recovered_amount:.2f}",
            status=simulated.status,
        ),
        AuditEvent(
            order=7,
            event_type=AuditEventType.OUTCOME_RECORDED,
            timestamp=ts(7),
            message=(
                f"{outcome.recovered_transactions}/{outcome.attempted_transactions} "
                "transaction(s) recovered (simulated)"
            ),
            value=f"recovery_rate={outcome.recovery_rate:.2%}",
            status=outcome.status,
        ),
    ]
    return events
