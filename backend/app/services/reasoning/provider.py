"""
ReasoningProvider abstraction.

CRITICAL RULE: a ReasoningProvider only explains an incident Phase 2 has
ALREADY confirmed. It never decides whether an incident exists, and it
never overrides Phase 2's classification — `analyze_incident` takes an
`IncidentEvidence` (built from a confirmed `Incident`) and returns an
`RCAResult`; Phase 2's `incident_type`/`severity` pass through unchanged
in `affected_scope`/evidence, they are not re-derived by the provider.
"""

from __future__ import annotations

from typing import Protocol

from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult


class ReasoningProvider(Protocol):
    def analyze_incident(self, evidence: IncidentEvidence) -> RCAResult: ...
