"""
Structured schemas for FlowShield's Phase 3 reasoning layer.

`IncidentEvidence` is built ONLY from Phase 2's already-confirmed
`ml.health.incidents.Incident` — the reasoning layer never sees raw
events or unconfirmed candidates, and never decides whether an incident
exists. `RCAResult` is what any `ReasoningProvider` must return; it is
validated (Pydantic) regardless of which provider produced it, so a
malformed LLM response cannot silently pass through.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ml.health.incidents import Incident, IncidentType, Severity


class ReasoningSource(str, Enum):
    """Where an RCAResult actually came from — never hidden from the caller."""

    MOCK = "mock"
    NEMOTRON = "nemotron"
    FALLBACK = "fallback"  # nemotron was configured but failed; mock output used instead


class IncidentEvidence(BaseModel):
    """Structured input to a ReasoningProvider, built from ONE confirmed
    Phase 2 Incident. This is the entire universe of facts the provider
    is allowed to reason over — nothing else is supplied."""

    model_config = ConfigDict(use_enum_values=False)

    incident_id: str
    incident_type: IncidentType
    severity: Severity
    detection_confidence: float = Field(
        ..., ge=0.0,
        description=(
            "The Phase 2 anomaly_score for this incident (0=typical of "
            "training data, can exceed 1). NOT a calibrated probability — "
            "carried through as-is so the reasoning layer doesn't mistake "
            "it for one."
        ),
    )
    affected_scope: list[dict] = Field(default_factory=list, description="Incident.affected_dimensions")
    health_score: float = Field(..., description="Phase 2 0-100 health score for this incident")
    signals: list[str] = Field(default_factory=list, description="Rule-based signals that fired")
    evidence: list[str] = Field(default_factory=list, description="Phase 2's own human-readable evidence lines")
    window_start: datetime
    window_end: datetime
    n_windows: int = 1

    @classmethod
    def from_incident(cls, incident: Incident) -> "IncidentEvidence":
        return cls(
            incident_id=incident.incident_id,
            incident_type=incident.incident_type,
            severity=incident.severity,
            detection_confidence=incident.anomaly_score,
            affected_scope=incident.affected_dimensions,
            health_score=incident.health_score,
            signals=incident.signals,
            evidence=incident.evidence,
            window_start=incident.window_start,
            window_end=incident.window_end,
            n_windows=incident.n_windows,
        )


class RCAResult(BaseModel):
    """Structured root-cause-analysis output. Required from every
    provider, mock or real — validated before use either way."""

    root_cause: str = Field(..., min_length=1, description="Inferred root cause, in plain language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Provider's self-reported confidence, 0-1")
    explanation: str = Field(..., min_length=1, description="Reasoning connecting evidence to root_cause")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Subset of the input evidence cited as support"
    )
    affected_scope: list[dict] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    source: ReasoningSource

    model_config = ConfigDict(use_enum_values=False)
