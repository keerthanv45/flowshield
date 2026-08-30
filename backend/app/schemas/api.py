"""
Phase 5A API schemas.

Reuses Phase 3/4 pydantic models directly as response types (no field
duplication) — `RCAResponse`/`RevenueRiskResponse`/
`RecoveryDecisionResponse`/`SimulationResponse` are aliases. Only
`IncidentResponse` (wrapping the Phase 2 `Incident` dataclass, which
isn't itself pydantic) and the dashboard/summary types are new.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.reasoning.schemas import RCAResult
from backend.app.services.recovery.schemas import RecoveryDecision, RevenueRisk, SimulatedExecution
from ml.health.incidents import Incident, IncidentType, Severity

# Reuse existing models directly -- see module docstring.
RCAResponse = RCAResult
RevenueRiskResponse = RevenueRisk
RecoveryDecisionResponse = RecoveryDecision
SimulationResponse = SimulatedExecution


class IncidentResponse(BaseModel):
    """Pydantic wrapper around the Phase 2 `Incident` dataclass (JSON-API
    boundary only — Phase 2 itself remains a plain dataclass)."""

    model_config = ConfigDict(use_enum_values=False)

    incident_id: str
    detected_at: datetime
    window_start: datetime
    window_end: datetime
    severity: Severity
    incident_type: IncidentType
    anomaly_score: float
    health_score: float
    affected_dimensions: list[dict]
    signals: list[str]
    evidence: list[str]
    status: str
    n_windows: int

    @classmethod
    def from_incident(cls, incident: Incident) -> "IncidentResponse":
        return cls(
            incident_id=incident.incident_id,
            detected_at=incident.detected_at,
            window_start=incident.window_start,
            window_end=incident.window_end,
            severity=incident.severity,
            incident_type=incident.incident_type,
            anomaly_score=incident.anomaly_score,
            health_score=incident.health_score,
            affected_dimensions=incident.affected_dimensions,
            signals=incident.signals,
            evidence=incident.evidence,
            status=incident.status.value,
            n_windows=incident.n_windows,
        )


class FlowShieldAnalysisResponse(BaseModel):
    incident: IncidentResponse
    rca: RCAResponse
    revenue_risk: RevenueRiskResponse
    recovery_decision: RecoveryDecisionResponse
    simulation: SimulationResponse | None = None

    model_config = ConfigDict(use_enum_values=False)


class DashboardSummary(BaseModel):
    total_transactions: int
    overall_success_rate: float
    current_health_score: float
    confirmed_incident_count: int
    active_incident_count: int = Field(..., description="Confirmed incidents with severity WARNING or CRITICAL")
    total_revenue_at_risk: float
    total_recoverable_revenue: float
    recommended_action_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of confirmed incidents by their top-scoring recommended action"
    )


class ConfigStatus(BaseModel):
    provider: str
    model: str | None
    nemotron_configured: bool
