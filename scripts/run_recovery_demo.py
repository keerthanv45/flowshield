"""
Phase 4 recovery demo.

Loads one CONFIRMED Phase 2 incident, generates a Phase 3 RCA (mock
provider by default -- no NVIDIA API key required), calculates
revenue-at-risk from actual synthetic events, generates recovery
candidates, runs the deterministic policy engine, simulates the
recommended action, and prints a concise end-to-end result.

Everything downstream of the policy decision is SIMULATED ONLY -- no
real payment action is ever taken.

Usage:
    python scripts/run_recovery_demo.py
    python scripts/run_recovery_demo.py --incident-index 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.reasoning.factory import get_reasoning_provider
from backend.app.services.reasoning.schemas import IncidentEvidence
from backend.app.services.recovery.candidates import generate_candidates
from backend.app.services.recovery.outcome import build_outcome
from backend.app.services.recovery.policy import RecoveryPolicyEngine
from backend.app.services.recovery.revenue_risk import calculate_revenue_risk
from backend.app.services.recovery.simulator import simulate_execution
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity


def _load_incident(path: Path, index: int | None) -> Incident:
    payload = json.loads(path.read_text())
    if not payload:
        raise SystemExit(f"No incidents found in {path}")

    if index is None:
        # Default: pick the first CRITICAL incident with a non-trivial
        # scope (more interesting for a demo than an arbitrary index 0),
        # falling back to the first incident of any severity.
        d = next((i for i in payload if i["severity"] == "CRITICAL"), payload[0])
    else:
        if index >= len(payload):
            raise SystemExit(f"--incident-index {index} out of range (0..{len(payload) - 1})")
        d = payload[index]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 revenue-risk + recovery decision demo")
    parser.add_argument("--incidents-path", default="data/synthetic/phase2_incidents.json")
    parser.add_argument("--events-path", default="data/synthetic/events.csv")
    parser.add_argument("--incident-index", type=int, default=None)
    args = parser.parse_args()

    incidents_path = Path(args.incidents_path)
    events_path = Path(args.events_path)
    if not incidents_path.exists() or not events_path.exists():
        raise SystemExit("Run scripts/evaluate_phase2.py first to produce Phase 2 artifacts.")

    incident = _load_incident(incidents_path, args.incident_index)
    events_df = pd.read_csv(events_path, parse_dates=["timestamp"])

    print(f"Confirmed incident: {incident.incident_id} ({incident.incident_type.value}, {incident.severity.value})")

    evidence = IncidentEvidence.from_incident(incident)
    rca = get_reasoning_provider().analyze_incident(evidence)
    print(f"RCA (source={rca.source.value}): {rca.root_cause} (confidence={rca.confidence:.2f})")

    revenue_risk = calculate_revenue_risk(incident, events_df)
    print(
        f"\nRevenue at risk: {revenue_risk.transactions_at_risk} txns, "
        f"Rs.{revenue_risk.gross_amount_at_risk:.2f} gross | "
        f"recoverable: {revenue_risk.recoverable_transactions} txns, "
        f"Rs.{revenue_risk.recoverable_amount:.2f} | "
        f"expected recovered: Rs.{revenue_risk.expected_recovered_amount:.2f}"
    )
    for entry in revenue_risk.failure_breakdown:
        print(
            f"  - {entry.failure_reason}: {entry.count} txns, Rs.{entry.amount:.2f}, "
            f"assumed recovery rate {entry.assumed_recovery_rate:.2f}"
        )

    candidates = generate_candidates(incident, revenue_risk)
    decision = RecoveryPolicyEngine().decide(evidence, rca, revenue_risk, candidates)
    print(f"\nRecommended action: {decision.recommended_action.value} (score={decision.decision_score:.2f}, risk={decision.risk_level.value})")
    for line in decision.reasoning:
        print(f"  - {line}")

    simulated = simulate_execution(decision, revenue_risk)
    outcome = build_outcome(simulated, decision, revenue_risk)
    print(f"\n[SIMULATED -- no real payment action taken] status={simulated.status}")
    print(
        f"  attempted={outcome.attempted_transactions} txns (Rs.{outcome.attempted_amount:.2f}), "
        f"recovered={outcome.recovered_transactions} txns (Rs.{outcome.recovered_amount:.2f}), "
        f"recovery_rate={outcome.recovery_rate:.2%}"
    )


if __name__ == "__main__":
    main()
