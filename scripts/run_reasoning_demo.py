"""
Phase 3 reasoning demo.

Loads one CONFIRMED Phase 2 incident (from
data/synthetic/phase2_incidents.json, produced by
scripts/evaluate_phase2.py), converts it to IncidentEvidence, runs the
configured ReasoningProvider (mock by default -- set
REASONING_PROVIDER=nemotron plus NEMOTRON_API_KEY/MODEL/BASE_URL to use
the real API), and prints the structured RCAResult.

Usage:
    python scripts/run_reasoning_demo.py
    python scripts/run_reasoning_demo.py --incident-index 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.reasoning.factory import get_reasoning_provider
from backend.app.services.reasoning.schemas import IncidentEvidence
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity


def _load_incident(path: Path, index: int) -> Incident:
    payload = json.loads(path.read_text())
    if not payload:
        raise SystemExit(f"No incidents found in {path}")
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
    parser = argparse.ArgumentParser(description="Run the Phase 3 reasoning layer over one confirmed incident")
    parser.add_argument("--incidents-path", default="data/synthetic/phase2_incidents.json")
    parser.add_argument("--incident-index", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.incidents_path)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/evaluate_phase2.py first.")

    incident = _load_incident(path, args.incident_index)
    print(f"Loaded confirmed incident: {incident.incident_id} ({incident.incident_type.value}, {incident.severity.value})")

    evidence = IncidentEvidence.from_incident(incident)
    provider = get_reasoning_provider()

    result = provider.analyze_incident(evidence)

    print(f"\n--- RCA Result (source={result.source.value}) ---")
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
