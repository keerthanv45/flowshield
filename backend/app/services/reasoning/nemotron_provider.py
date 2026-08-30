"""
NemotronProvider: calls NVIDIA's OpenAI-compatible API (NIM) to produce
an RCAResult from IncidentEvidence.

Never decides whether an incident exists — Phase 2 already confirmed it.
Only reasons over the supplied structured evidence. On ANY failure
(missing config, network error, malformed/invalid response), raises
`NemotronError` — callers (see `factory.py`) are responsible for falling
back to `MockReasoningProvider` and marking the result as `FALLBACK`,
never silently presenting a fallback as a real Nemotron response.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult, ReasoningSource

SYSTEM_PROMPT = """You are FlowShield's payment reliability reasoning engine.

Analyze ONLY the supplied structured incident evidence. Do not invent facts.
Do not claim an incident exists or does not exist — that has already been
decided upstream; your job is only to explain the CONFIRMED incident you are
given. Do not invent transactions, banks, metrics, or causes not present in
the evidence. Clearly distinguish observed evidence (what the data shows)
from inferred root cause (your interpretation of why).

Return ONLY a JSON object with exactly these fields:
{
  "root_cause": string,
  "confidence": number between 0 and 1,
  "explanation": string,
  "supporting_evidence": array of strings (drawn from the input evidence),
  "affected_scope": array of objects (copy from the input affected_scope),
  "recommended_actions": array of strings
}
No prose outside the JSON object."""


class NemotronError(Exception):
    """Raised for any Nemotron configuration, request, or validation
    failure. Never includes secret values in its message."""


def _build_user_prompt(evidence: IncidentEvidence) -> str:
    payload = evidence.model_dump(mode="json")
    return (
        "Structured incident evidence (confirmed by FlowShield's Phase 2 "
        "detection engine — do not question whether it occurred):\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


class NemotronProvider:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 30.0):
        if not api_key:
            raise NemotronError("Nemotron API key is not configured")
        if not base_url:
            raise NemotronError("Nemotron base URL is not configured")
        if not model:
            raise NemotronError("Nemotron model is not configured")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency always installed in this project
            raise NemotronError("openai package is not installed") from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model

    def analyze_incident(self, evidence: IncidentEvidence) -> RCAResult:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(evidence)},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # Never include the API key (never passed to this exception
            # path in the first place) -- str(exc) from the openai client
            # does not echo request headers/auth.
            raise NemotronError(f"Nemotron request failed: {type(exc).__name__}") from exc

        raw_content = response.choices[0].message.content if response.choices else None
        if not raw_content:
            raise NemotronError("Nemotron returned an empty response")

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise NemotronError("Nemotron response was not valid JSON") from exc

        parsed["source"] = ReasoningSource.NEMOTRON
        try:
            return RCAResult(**parsed)
        except ValidationError as exc:
            raise NemotronError(f"Nemotron response failed schema validation: {exc.error_count()} error(s)") from exc
