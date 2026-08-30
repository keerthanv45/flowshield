"""
FlowShieldOrchestrator: the single service the API layer talks to.

Wires together (never reimplements):
  - Phase 2 confirmed incidents (data/synthetic/phase2_incidents.json)
  - Phase 3 reasoning provider (backend.app.services.reasoning.factory)
  - Phase 4 revenue risk / candidates / policy / simulator
    (backend.app.services.recovery.*)

Default reasoning provider remains MOCK (via `get_reasoning_provider()`,
which itself defaults to mock and only uses Nemotron if
REASONING_PROVIDER=nemotron is explicitly set) -- the API works with zero
NVIDIA credentials.

In-memory caching only (no database, per Phase 5A scope): incidents and
events are loaded once per process; a full analysis (RCA + revenue risk
+ decision) is computed once per incident_id and reused. This is safe
because all Phase 2 artifacts are static synthetic files for this phase.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backend.app.services.reasoning.factory import get_reasoning_provider
from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult
from backend.app.services.recovery.candidates import generate_candidates
from backend.app.services.recovery.policy import RecoveryPolicyEngine
from backend.app.services.recovery.revenue_risk import calculate_revenue_risk
from backend.app.services.recovery.schemas import RecoveryDecision, RevenueRisk, SimulatedExecution
from backend.app.services.recovery.simulator import simulate_execution
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity

DEFAULT_DATA_DIR = Path("data/synthetic")


def _incident_from_dict(d: dict) -> Incident:
    return Incident(
        incident_id=d["incident_id"],
        detected_at=datetime.fromisoformat(d["detected_at"]),
        window_start=datetime.fromisoformat(d["window_start"]),
        window_end=datetime.fromisoformat(d["window_end"]),
        severity=Severity(d["severity"]),
        incident_type=IncidentType(d["incident_type"]),
        anomaly_score=d["anomaly_score"],
        health_score=d["health_score"],
        affected_dimensions=d["affected_dimensions"],
        signals=d["signals"],
        evidence=d["evidence"],
        status=IncidentStatus(d["status"]),
        n_windows=d.get("n_windows", 1),
    )


class IncidentNotFoundError(Exception):
    pass


class FlowShieldOrchestrator:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._data_dir = data_dir
        self._incidents: list[Incident] | None = None
        self._events_df: pd.DataFrame | None = None
        self._analysis_report: dict | None = None
        self._analysis_cache: dict[str, tuple[RCAResult, RevenueRisk, RecoveryDecision]] = {}
        self._policy_engine = RecoveryPolicyEngine()

    @property
    def incidents(self) -> list[Incident]:
        if self._incidents is None:
            path = self._data_dir / "phase2_incidents.json"
            payload = json.loads(path.read_text()) if path.exists() else []
            self._incidents = [_incident_from_dict(d) for d in payload]
        return self._incidents

    @property
    def events_df(self) -> pd.DataFrame:
        if self._events_df is None:
            path = self._data_dir / "events.csv"
            self._events_df = pd.read_csv(path, parse_dates=["timestamp"])
        return self._events_df

    @property
    def analysis_report(self) -> dict:
        if self._analysis_report is None:
            path = self._data_dir / "analysis_report.json"
            self._analysis_report = json.loads(path.read_text()) if path.exists() else {}
        return self._analysis_report

    def _latest_health_score(self) -> float:
        path = self._data_dir / "phase2_health_snapshots.csv"
        if not path.exists():
            return 0.0
        snapshots = pd.read_csv(path, parse_dates=["window_start"])
        if snapshots.empty:
            return 0.0
        return float(snapshots.sort_values("window_start").iloc[-1]["health_score"])

    def list_incidents(
        self, severity: str | None = None, incident_type: str | None = None
    ) -> list[Incident]:
        result = self.incidents
        if severity:
            result = [i for i in result if i.severity.value == severity]
        if incident_type:
            result = [i for i in result if i.incident_type.value == incident_type]
        return result

    def get_incident(self, incident_id: str) -> Incident:
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                return incident
        raise IncidentNotFoundError(incident_id)

    def _run_analysis(self, incident: Incident) -> tuple[RCAResult, RevenueRisk, RecoveryDecision]:
        if incident.incident_id in self._analysis_cache:
            return self._analysis_cache[incident.incident_id]

        evidence = IncidentEvidence.from_incident(incident)
        rca = get_reasoning_provider().analyze_incident(evidence)
        revenue_risk = calculate_revenue_risk(incident, self.events_df)
        candidates = generate_candidates(incident, revenue_risk)
        decision = self._policy_engine.decide(evidence, rca, revenue_risk, candidates)

        result = (rca, revenue_risk, decision)
        self._analysis_cache[incident.incident_id] = result
        return result

    def analyze(self, incident_id: str) -> tuple[Incident, RCAResult, RevenueRisk, RecoveryDecision]:
        incident = self.get_incident(incident_id)
        rca, revenue_risk, decision = self._run_analysis(incident)
        return incident, rca, revenue_risk, decision

    def simulate(
        self, incident_id: str
    ) -> tuple[Incident, RCAResult, RevenueRisk, RecoveryDecision, SimulatedExecution]:
        incident, rca, revenue_risk, decision = self.analyze(incident_id)
        simulated = simulate_execution(decision, revenue_risk)
        return incident, rca, revenue_risk, decision, simulated

    def summary(self) -> dict:
        report = self.analysis_report
        total_transactions = int(report.get("total_events", len(self.events_df)))
        overall_success_rate = float(report.get("success_rate", 0.0))

        confirmed = self.incidents
        active = [i for i in confirmed if i.severity in (Severity.WARNING, Severity.CRITICAL)]

        total_revenue_at_risk = 0.0
        total_recoverable_revenue = 0.0
        action_counts: dict[str, int] = {}

        for incident in confirmed:
            _rca, revenue_risk, decision = self._run_analysis(incident)
            total_revenue_at_risk += revenue_risk.gross_amount_at_risk
            total_recoverable_revenue += revenue_risk.recoverable_amount
            action_counts[decision.recommended_action.value] = (
                action_counts.get(decision.recommended_action.value, 0) + 1
            )

        return {
            "total_transactions": total_transactions,
            "overall_success_rate": overall_success_rate,
            "current_health_score": self._latest_health_score(),
            "confirmed_incident_count": len(confirmed),
            "active_incident_count": len(active),
            "total_revenue_at_risk": round(total_revenue_at_risk, 2),
            "total_recoverable_revenue": round(total_recoverable_revenue, 2),
            "recommended_action_counts": action_counts,
        }


_orchestrator_instance: FlowShieldOrchestrator | None = None


def get_orchestrator() -> FlowShieldOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = FlowShieldOrchestrator()
    return _orchestrator_instance
