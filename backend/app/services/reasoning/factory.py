"""
Provider selection + safe fallback.

Configuration is read ONLY from environment variables:
  REASONING_PROVIDER  - "mock" | "nemotron" (default: "mock")
  NEMOTRON_API_KEY
  NEMOTRON_MODEL
  NEMOTRON_BASE_URL

Three distinct, always-visible outcomes (`RCAResult.source`):
  MOCK     - REASONING_PROVIDER was "mock" (or unset) -- mock chosen deliberately.
  NEMOTRON - REASONING_PROVIDER was "nemotron" and a real response was
             returned and validated successfully.
  FALLBACK - REASONING_PROVIDER was "nemotron" but nemotron could not be
             used for any reason (missing/incomplete config, request
             failure, malformed/invalid response) -- mock output is used,
             but never mislabeled as a deliberate "mock" choice or as a
             real nemotron response.
"""

from __future__ import annotations

import os

from backend.app.services.reasoning.mock_provider import MockReasoningProvider
from backend.app.services.reasoning.nemotron_provider import NemotronError, NemotronProvider
from backend.app.services.reasoning.provider import ReasoningProvider
from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult, ReasoningSource


class _MisconfiguredProvider:
    """Stand-in used when REASONING_PROVIDER=nemotron but required env
    vars are missing/incomplete. Always raises on use, so it flows
    through the exact same fallback path as a runtime request failure --
    no separate code path to keep in sync."""

    def __init__(self, reason: str):
        self._reason = reason

    def analyze_incident(self, evidence: IncidentEvidence) -> RCAResult:
        raise NemotronError(self._reason)


class SafeReasoningProvider:
    """Wraps a configured provider (nemotron or mock) with fail-safe
    fallback to MockReasoningProvider on any failure. This is what
    `get_reasoning_provider()` returns — callers never need their own
    try/except around a bare NemotronProvider."""

    def __init__(self, primary: ReasoningProvider | None):
        self._primary = primary
        self._mock = MockReasoningProvider()

    def analyze_incident(self, evidence: IncidentEvidence) -> RCAResult:
        if self._primary is None:
            return self._mock.analyze_incident(evidence)

        try:
            return self._primary.analyze_incident(evidence)
        except NemotronError:
            result = self._mock.analyze_incident(evidence)
            return result.model_copy(update={"source": ReasoningSource.FALLBACK})


def get_reasoning_provider() -> SafeReasoningProvider:
    provider_name = os.environ.get("REASONING_PROVIDER", "mock").strip().lower()

    if provider_name != "nemotron":
        return SafeReasoningProvider(primary=None)

    api_key = os.environ.get("NEMOTRON_API_KEY", "")
    model = os.environ.get("NEMOTRON_MODEL", "")
    base_url = os.environ.get("NEMOTRON_BASE_URL", "")

    if not api_key or not model or not base_url:
        return SafeReasoningProvider(primary=_MisconfiguredProvider("Nemotron requested but not fully configured"))

    try:
        primary = NemotronProvider(api_key=api_key, model=model, base_url=base_url)
    except NemotronError as exc:
        return SafeReasoningProvider(primary=_MisconfiguredProvider(str(exc)))

    return SafeReasoningProvider(primary=primary)
