"""
MockReasoningProvider: deterministic, template-based RCA generation from
`IncidentEvidence` alone — no network call, no LLM. Used as the default
provider, and as the safe fallback when Nemotron is unconfigured or
fails (see `factory.py`).
"""

from __future__ import annotations

from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult, ReasoningSource

_ROOT_CAUSE_TEMPLATES = {
    "BANK_RAIL_DEGRADATION": "Localized degradation on a specific bank + payment method rail",
    "REGIONAL_DEGRADATION": "Regional payment infrastructure degradation",
    "LATENCY_SPIKE": "System-wide latency increase affecting payment processing",
    "MERCHANT_SYSTEM_DEGRADATION": "Broad merchant/system-level payment degradation",
    "ISOLATED_FAILURES": "Scattered failures with no systemic concentration identified",
}

_ACTION_TEMPLATES = {
    "BANK_RAIL_DEGRADATION": [
        "Verify status with the affected bank/rail provider",
        "Consider temporary routing away from the affected combination",
    ],
    "REGIONAL_DEGRADATION": [
        "Check regional infrastructure/network status",
        "Monitor for spread to adjacent regions",
    ],
    "LATENCY_SPIKE": [
        "Check upstream gateway/network latency",
        "Review recent deployments or capacity changes",
    ],
    "MERCHANT_SYSTEM_DEGRADATION": [
        "Escalate to platform/infrastructure on-call",
        "Check for shared dependency failures across banks/methods",
    ],
    "ISOLATED_FAILURES": [
        "Continue monitoring; no action indicated by current evidence",
    ],
}


class MockReasoningProvider:
    """Template-based provider. Deterministic given the same evidence —
    no randomness, no external calls."""

    def analyze_incident(self, evidence: IncidentEvidence) -> RCAResult:
        incident_type = evidence.incident_type.value if hasattr(evidence.incident_type, "value") else str(evidence.incident_type)

        root_cause = _ROOT_CAUSE_TEMPLATES.get(incident_type, "Unclassified payment degradation")
        actions = _ACTION_TEMPLATES.get(incident_type, ["Investigate further before taking action"])

        explanation = (
            f"Phase 2 classified this as {incident_type} with {evidence.severity} severity "
            f"over {evidence.n_windows} window(s), health score {evidence.health_score:.1f}. "
            f"{len(evidence.signals)} rule signal(s) fired: {', '.join(evidence.signals) or 'none'}. "
            "This is a template-based summary of the supplied evidence, not an independent diagnosis."
        )

        return RCAResult(
            root_cause=root_cause,
            confidence=0.5,  # fixed, deliberately mid-range — mock has no real basis to vary this
            explanation=explanation,
            supporting_evidence=list(evidence.evidence),
            affected_scope=list(evidence.affected_scope),
            recommended_actions=actions,
            source=ReasoningSource.MOCK,
        )
