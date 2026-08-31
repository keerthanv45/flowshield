"""
FlowShield API v1 routes. Thin HTTP layer only — all business logic
lives in `backend.app.services.orchestrator.FlowShieldOrchestrator` and
the Phase 2/3/4 services it wires together.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import (
    AuditTrailResponse,
    ConfigStatus,
    DashboardSummary,
    FlowShieldAnalysisResponse,
    IncidentResponse,
)
from backend.app.services.orchestrator import IncidentNotFoundError, get_orchestrator
from backend.app.services.recovery.batch_evaluation import BatchEvaluationResult

router = APIRouter(prefix="/api/v1")


@router.get("/summary", response_model=DashboardSummary)
def get_summary() -> DashboardSummary:
    orchestrator = get_orchestrator()
    return DashboardSummary(**orchestrator.summary())


@router.get("/incidents", response_model=list[IncidentResponse])
def list_incidents(
    severity: str | None = Query(default=None, description="Filter by severity: INFO|WARNING|CRITICAL"),
    incident_type: str | None = Query(default=None, description="Filter by incident_type"),
    limit: int | None = Query(default=None, ge=1, description="Cap the number of results returned"),
) -> list[IncidentResponse]:
    orchestrator = get_orchestrator()
    incidents = orchestrator.list_incidents(severity=severity, incident_type=incident_type)
    if limit is not None:
        incidents = incidents[:limit]
    return [IncidentResponse.from_incident(i) for i in incidents]


@router.get("/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str) -> IncidentResponse:
    orchestrator = get_orchestrator()
    try:
        incident = orchestrator.get_incident(incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")
    return IncidentResponse.from_incident(incident)


@router.post("/incidents/{incident_id}/analyze", response_model=FlowShieldAnalysisResponse)
def analyze_incident(incident_id: str) -> FlowShieldAnalysisResponse:
    """incident -> RCA -> revenue risk -> candidates -> policy decision.
    Does NOT execute or simulate recovery."""
    orchestrator = get_orchestrator()
    try:
        incident, rca, revenue_risk, decision = orchestrator.analyze(incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")

    return FlowShieldAnalysisResponse(
        incident=IncidentResponse.from_incident(incident),
        rca=rca,
        revenue_risk=revenue_risk,
        recovery_decision=decision,
        simulation=None,
    )


@router.post("/incidents/{incident_id}/simulate", response_model=FlowShieldAnalysisResponse)
def simulate_incident(incident_id: str) -> FlowShieldAnalysisResponse:
    """incident -> RCA -> revenue risk -> policy decision -> SIMULATED
    execution. No real payment action is ever taken; never calls Razorpay."""
    orchestrator = get_orchestrator()
    try:
        incident, rca, revenue_risk, decision, simulated = orchestrator.simulate(incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")

    return FlowShieldAnalysisResponse(
        incident=IncidentResponse.from_incident(incident),
        rca=rca,
        revenue_risk=revenue_risk,
        recovery_decision=decision,
        simulation=simulated,
    )


@router.get("/incidents/{incident_id}/audit", response_model=AuditTrailResponse)
def incident_audit_trail(incident_id: str) -> AuditTrailResponse:
    """Structured audit trail across the 7 pipeline stages. Reuses the
    full analyze+simulate pipeline; formats existing outputs only."""
    orchestrator = get_orchestrator()
    try:
        events = orchestrator.audit_trail(incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")
    return AuditTrailResponse(incident_id=incident_id, events=events)


@router.get("/config/status", response_model=ConfigStatus)
def config_status() -> ConfigStatus:
    """Safe configuration info only -- NEVER returns API keys/secrets."""
    provider = os.environ.get("REASONING_PROVIDER", "mock").strip().lower()
    model = os.environ.get("NEMOTRON_MODEL") or None
    nemotron_configured = bool(
        os.environ.get("NEMOTRON_API_KEY") and os.environ.get("NEMOTRON_MODEL") and os.environ.get("NEMOTRON_BASE_URL")
    )
    return ConfigStatus(provider=provider, model=model, nemotron_configured=nemotron_configured)


@router.get("/recovery/evaluation", response_model=BatchEvaluationResult)
def recovery_evaluation() -> BatchEvaluationResult:
    """Phase 6: batch recovery evaluation across the whole synthetic
    dataset. SIMULATED ONLY -- reuses backend.app.services.recovery.*
    unchanged; no duplicate recovery logic lives in this route."""
    orchestrator = get_orchestrator()
    return orchestrator.batch_recovery_evaluation()
