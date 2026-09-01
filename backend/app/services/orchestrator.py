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
from backend.app.services.reasoning.mock_provider import MockReasoningProvider
from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult
from backend.app.services.recovery.candidates import generate_candidates
from backend.app.services.recovery.policy import RecoveryPolicyEngine
from backend.app.services.recovery.revenue_risk import calculate_revenue_risk
from backend.app.services.recovery.schemas import RecoveryDecision, RevenueRisk, SimulatedExecution
from backend.app.services.recovery.simulator import simulate_execution
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity

# Resolve relative to the repository root (three levels up from this file:
# backend/app/services/orchestrator.py -> backend/app/services -> backend/app
# -> backend -> repo root), NOT the process working directory. A relative
# Path("data/synthetic") broke in production (e.g. Render) where the
# working directory at process start isn't guaranteed to be the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "synthetic"


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
        self._summary_cache: dict[str, tuple[RevenueRisk, RecoveryDecision]] = {}
        self._policy_engine = RecoveryPolicyEngine()
        # Used ONLY by summary()'s aggregate scoring -- never by
        # analyze()/simulate()/audit_trail(), which always go through
        # get_reasoning_provider() (respecting REASONING_PROVIDER). See
        # summary()'s docstring for why.
        self._summary_reasoning_provider = MockReasoningProvider()

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

    def _overall_success_rate(self) -> float:
        """Prefer the precomputed Phase 1 analysis_report.json (the exact
        figure `scripts/run_analysis.py` reports). Falls back to computing
        the same statistic directly from the already-loaded events_df when
        that file isn't present in a given deployment (e.g. it's
        gitignored and the deploy step never ran run_analysis.py) --
        `success_rate` is just `successful_count / total`, the same
        definition Phase 1 already uses, not new business logic."""
        report_value = self.analysis_report.get("success_rate")
        if report_value is not None:
            return float(report_value)

        df = self.events_df
        if df.empty:
            return 0.0
        return float((df["status"] == "success").mean())

    def _latest_health_score(self) -> float:
        """Prefer the precomputed Phase 2 phase2_health_snapshots.csv (the
        exact score `scripts/evaluate_phase2.py` computed, baseline-aware).
        Falls back to a cheap approximation using the already-loaded
        events_df when that file isn't present in a given deployment:
        aggregates the most recent 15-minute window with the EXISTING
        `ml.health.aggregation.aggregate_time_windows`, builds a simple
        whole-dataset-average baseline, and scores it with the EXISTING,
        unmodified `ml.health.scoring.compute_health_score` -- reuses
        Phase 2's own functions rather than inventing a new formula. This
        is an approximation (a single global average, not the full
        hour-of-day-aware baseline the precomputed snapshot used), so it
        is only used when the real precomputed value is unavailable."""
        path = self._data_dir / "phase2_health_snapshots.csv"
        if path.exists():
            snapshots = pd.read_csv(path, parse_dates=["window_start"])
            if not snapshots.empty:
                return float(snapshots.sort_values("window_start").iloc[-1]["health_score"])

        from ml.health.aggregation import aggregate_time_windows
        from ml.health.scoring import compute_health_score

        df = self.events_df
        if df.empty:
            return 0.0

        windows = aggregate_time_windows(df, window_minutes=15)
        if windows.empty:
            return 0.0

        windows = windows.sort_values("window_start")
        latest = windows.iloc[-1]
        baseline_means = windows[
            ["success_rate", "failure_rate", "average_latency_ms", "p95_latency_ms", "transaction_count"]
        ].mean()

        row = latest.to_dict()
        row["baseline_success_rate"] = float(baseline_means["success_rate"])
        row["baseline_failure_rate"] = float(baseline_means["failure_rate"])
        row["baseline_average_latency_ms"] = float(baseline_means["average_latency_ms"])
        row["baseline_p95_latency_ms"] = float(baseline_means["p95_latency_ms"])
        row["baseline_transaction_count"] = float(baseline_means["transaction_count"])

        return compute_health_score(row).score

    def batch_recovery_evaluation(self):
        """Phase 6: batch evaluation across the WHOLE dataset (not
        incident-scoped). Reuses the cached events_df; not cached itself
        since it's cheap (~tens of ms) and always reflects current data."""
        from backend.app.services.recovery.batch_evaluation import run_batch_evaluation

        return run_batch_evaluation(self.events_df)

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

    def audit_trail(self, incident_id: str) -> list:
        """Reuses the full analyze+simulate pipeline (no new business
        logic) and formats it into a structured audit trail. See
        backend.app.services.audit for the formatting-only logic."""
        from backend.app.services.audit import build_audit_trail
        from backend.app.services.recovery.outcome import build_outcome

        incident, rca, revenue_risk, decision, simulated = self.simulate(incident_id)
        outcome = build_outcome(simulated, decision, revenue_risk)
        return build_audit_trail(incident, rca, revenue_risk, decision, simulated, outcome)

    def _summary_decision(self, incident: Incident) -> tuple[RevenueRisk, RecoveryDecision]:
        """Revenue risk + recovery decision for the AGGREGATE dashboard
        summary only. Deliberately uses MockReasoningProvider (fast,
        deterministic, no network call) instead of get_reasoning_provider()
        -- with REASONING_PROVIDER=nemotron, calling the real provider
        once per confirmed incident (107 in the current dataset) here
        would mean 107 live LLM requests on every /api/v1/summary call,
        which is what caused production timeouts. revenue_risk and the
        candidate generation are already fully deterministic (no
        reasoning-provider involvement at all); only the RCA confidence
        fed into the policy engine's guardrail is a stand-in here. This
        means a given incident's `recommended_action` in the summary
        aggregate can differ from what POST /analyze or /simulate return
        for that same incident when a real Nemotron RCA yields a
        different confidence -- those two endpoints are unaffected by
        this method and always use the configured provider. Cached
        separately from `_analysis_cache` so this never contaminates the
        real-provider results used by analyze/simulate/audit_trail.
        """
        if incident.incident_id in self._summary_cache:
            return self._summary_cache[incident.incident_id]

        evidence = IncidentEvidence.from_incident(incident)
        rca = self._summary_reasoning_provider.analyze_incident(evidence)
        revenue_risk = calculate_revenue_risk(incident, self.events_df)
        candidates = generate_candidates(incident, revenue_risk)
        decision = self._policy_engine.decide(evidence, rca, revenue_risk, candidates)

        result = (revenue_risk, decision)
        self._summary_cache[incident.incident_id] = result
        return result

    def summary(self) -> dict:
        report = self.analysis_report
        total_transactions = int(report.get("total_events", len(self.events_df)))
        overall_success_rate = self._overall_success_rate()

        confirmed = self.incidents
        active = [i for i in confirmed if i.severity in (Severity.WARNING, Severity.CRITICAL)]

        total_revenue_at_risk = 0.0
        total_recoverable_revenue = 0.0
        action_counts: dict[str, int] = {}

        for incident in confirmed:
            revenue_risk, decision = self._summary_decision(incident)
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
