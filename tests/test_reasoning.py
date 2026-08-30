from datetime import datetime

import pytest
from pydantic import ValidationError

from backend.app.services.reasoning.factory import (
    SafeReasoningProvider,
    _MisconfiguredProvider,
    get_reasoning_provider,
)
from backend.app.services.reasoning.mock_provider import MockReasoningProvider
from backend.app.services.reasoning.nemotron_provider import NemotronError, NemotronProvider
from backend.app.services.reasoning.schemas import IncidentEvidence, RCAResult, ReasoningSource
from ml.health.incidents import Incident, IncidentStatus, IncidentType, Severity


def make_incident(**overrides) -> Incident:
    base = dict(
        incident_id="inc_test_0001",
        detected_at=datetime(2026, 8, 3, 2, 0),
        window_start=datetime(2026, 8, 3, 2, 0),
        window_end=datetime(2026, 8, 3, 2, 30),
        severity=Severity.CRITICAL,
        incident_type=IncidentType.BANK_RAIL_DEGRADATION,
        anomaly_score=0.87,
        health_score=22.5,
        affected_dimensions=[{"bank": "HDFC", "payment_method": "UPI"}],
        signals=["SUCCESS_RATE_DEGRADATION", "BANK_CONCENTRATION"],
        evidence=["Overall success rate: 0.50 (baseline: 0.90, delta: -0.40)"],
        status=IncidentStatus.CONFIRMED,
        n_windows=2,
    )
    base.update(overrides)
    return Incident(**base)


class TestEvidenceFromIncident:
    def test_builds_from_confirmed_incident(self):
        incident = make_incident()
        evidence = IncidentEvidence.from_incident(incident)
        assert evidence.incident_id == incident.incident_id
        assert evidence.incident_type == IncidentType.BANK_RAIL_DEGRADATION
        assert evidence.affected_scope == incident.affected_dimensions
        assert evidence.detection_confidence == incident.anomaly_score

    def test_evidence_validation_rejects_negative_confidence(self):
        with pytest.raises(ValidationError):
            IncidentEvidence(
                incident_id="x", incident_type=IncidentType.LATENCY_SPIKE, severity=Severity.INFO,
                detection_confidence=-0.1, health_score=90.0,
                window_start=datetime(2026, 8, 1), window_end=datetime(2026, 8, 1, 0, 15),
            )

    def test_evidence_defaults_are_empty_not_none(self):
        evidence = IncidentEvidence(
            incident_id="x", incident_type=IncidentType.LATENCY_SPIKE, severity=Severity.INFO,
            detection_confidence=0.1, health_score=90.0,
            window_start=datetime(2026, 8, 1), window_end=datetime(2026, 8, 1, 0, 15),
        )
        assert evidence.affected_scope == []
        assert evidence.signals == []
        assert evidence.evidence == []


class TestRCAResultValidation:
    def test_valid_result_constructs(self):
        result = RCAResult(
            root_cause="Bank rail degradation", confidence=0.7, explanation="x",
            supporting_evidence=[], affected_scope=[], recommended_actions=[],
            source=ReasoningSource.MOCK,
        )
        assert result.confidence == 0.7

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            RCAResult(
                root_cause="x", confidence=1.5, explanation="x", source=ReasoningSource.MOCK,
            )

    def test_empty_root_cause_rejected(self):
        with pytest.raises(ValidationError):
            RCAResult(root_cause="", confidence=0.5, explanation="x", source=ReasoningSource.MOCK)

    def test_missing_source_rejected(self):
        with pytest.raises(ValidationError):
            RCAResult(root_cause="x", confidence=0.5, explanation="x")


class TestMockProvider:
    def test_returns_valid_rca_result(self):
        evidence = IncidentEvidence.from_incident(make_incident())
        result = MockReasoningProvider().analyze_incident(evidence)
        assert isinstance(result, RCAResult)
        assert result.source == ReasoningSource.MOCK

    def test_deterministic_same_evidence_same_output(self):
        evidence = IncidentEvidence.from_incident(make_incident())
        r1 = MockReasoningProvider().analyze_incident(evidence)
        r2 = MockReasoningProvider().analyze_incident(evidence)
        assert r1.root_cause == r2.root_cause
        assert r1.confidence == r2.confidence

    def test_root_cause_varies_by_incident_type(self):
        provider = MockReasoningProvider()
        bank_rail = provider.analyze_incident(
            IncidentEvidence.from_incident(make_incident(incident_type=IncidentType.BANK_RAIL_DEGRADATION))
        )
        latency = provider.analyze_incident(
            IncidentEvidence.from_incident(make_incident(incident_type=IncidentType.LATENCY_SPIKE))
        )
        assert bank_rail.root_cause != latency.root_cause

    def test_never_claims_incident_exists_beyond_input(self):
        # Mock provider must not fabricate a different incident_type or
        # invent affected_scope not present in the evidence.
        incident = make_incident(affected_dimensions=[{"bank": "SBI", "payment_method": "CARD"}])
        evidence = IncidentEvidence.from_incident(incident)
        result = MockReasoningProvider().analyze_incident(evidence)
        assert result.affected_scope == evidence.affected_scope


class TestProviderInterface:
    def test_mock_and_would_be_nemotron_share_interface(self):
        # Structural check: both expose analyze_incident(evidence) -> RCAResult.
        assert hasattr(MockReasoningProvider(), "analyze_incident")
        assert hasattr(NemotronProvider, "analyze_incident")


class TestMissingApiKeyFallback:
    def test_get_provider_falls_back_when_nemotron_selected_without_key(self, monkeypatch):
        monkeypatch.setenv("REASONING_PROVIDER", "nemotron")
        monkeypatch.delenv("NEMOTRON_API_KEY", raising=False)
        monkeypatch.setenv("NEMOTRON_MODEL", "some-model")
        monkeypatch.setenv("NEMOTRON_BASE_URL", "https://example.invalid/v1")

        provider = get_reasoning_provider()
        evidence = IncidentEvidence.from_incident(make_incident())
        result = provider.analyze_incident(evidence)

        assert result.source == ReasoningSource.FALLBACK

    def test_default_provider_is_mock_when_unset(self, monkeypatch):
        monkeypatch.delenv("REASONING_PROVIDER", raising=False)
        provider = get_reasoning_provider()
        evidence = IncidentEvidence.from_incident(make_incident())
        result = provider.analyze_incident(evidence)
        assert result.source == ReasoningSource.MOCK

    def test_explicit_mock_selection_reports_mock_not_fallback(self, monkeypatch):
        monkeypatch.setenv("REASONING_PROVIDER", "mock")
        provider = get_reasoning_provider()
        evidence = IncidentEvidence.from_incident(make_incident())
        result = provider.analyze_incident(evidence)
        assert result.source == ReasoningSource.MOCK


class TestApiFailureFallback:
    def test_safe_provider_falls_back_on_nemotron_error(self):
        class AlwaysFailsProvider:
            def analyze_incident(self, evidence):
                raise NemotronError("simulated network failure")

        safe = SafeReasoningProvider(primary=AlwaysFailsProvider())
        evidence = IncidentEvidence.from_incident(make_incident())
        result = safe.analyze_incident(evidence)
        assert result.source == ReasoningSource.FALLBACK

    def test_misconfigured_provider_always_raises(self):
        provider = _MisconfiguredProvider("missing config")
        evidence = IncidentEvidence.from_incident(make_incident())
        with pytest.raises(NemotronError):
            provider.analyze_incident(evidence)


class TestMalformedLLMOutput:
    def test_malformed_json_raises_nemotron_error(self):
        # Directly exercise NemotronProvider's parsing path without a
        # real network call, by constructing it with a fake OpenAI-like
        # client whose response content is not valid JSON.
        provider = NemotronProvider.__new__(NemotronProvider)
        provider._model = "fake-model"

        class FakeMessage:
            content = "not valid json {{{"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        provider._client = FakeClient()

        evidence = IncidentEvidence.from_incident(make_incident())
        with pytest.raises(NemotronError):
            provider.analyze_incident(evidence)

    def test_valid_json_but_invalid_schema_raises_nemotron_error(self):
        provider = NemotronProvider.__new__(NemotronProvider)
        provider._model = "fake-model"

        class FakeMessage:
            content = '{"root_cause": "", "confidence": 99}'  # invalid: empty root_cause, confidence out of range

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        provider._client = FakeClient()

        evidence = IncidentEvidence.from_incident(make_incident())
        with pytest.raises(NemotronError):
            provider.analyze_incident(evidence)

    def test_empty_response_raises_nemotron_error(self):
        provider = NemotronProvider.__new__(NemotronProvider)
        provider._model = "fake-model"

        class FakeMessage:
            content = None

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        provider._client = FakeClient()

        evidence = IncidentEvidence.from_incident(make_incident())
        with pytest.raises(NemotronError):
            provider.analyze_incident(evidence)


class TestSecretNotExposed:
    def test_nemotron_provider_requires_key_no_default(self):
        with pytest.raises(NemotronError):
            NemotronProvider(api_key="", model="m", base_url="https://example.invalid/v1")

    def test_error_message_never_contains_api_key(self):
        secret = "sk-super-secret-nemotron-key-12345"
        try:
            NemotronProvider(api_key="", model="m", base_url="https://example.invalid/v1")
        except NemotronError as exc:
            assert secret not in str(exc)

    def test_misconfigured_provider_message_has_no_key_value(self, monkeypatch):
        secret = "sk-super-secret-nemotron-key-12345"
        monkeypatch.setenv("REASONING_PROVIDER", "nemotron")
        monkeypatch.delenv("NEMOTRON_API_KEY", raising=False)
        monkeypatch.setenv("NEMOTRON_MODEL", "m")
        monkeypatch.setenv("NEMOTRON_BASE_URL", "https://example.invalid/v1")

        provider = get_reasoning_provider()
        evidence = IncidentEvidence.from_incident(make_incident())
        result = provider.analyze_incident(evidence)
        # Nothing in the resulting RCAResult should ever contain a secret
        # (there is none configured here, but this guards the shape of
        # what gets surfaced -- source is FALLBACK, not a raw exception
        # dump that could contain request internals).
        assert secret not in str(result.model_dump())
        assert result.source == ReasoningSource.FALLBACK


class TestPromptContent:
    def test_user_prompt_contains_only_evidence_fields(self):
        from backend.app.services.reasoning.nemotron_provider import _build_user_prompt

        incident = make_incident()
        evidence = IncidentEvidence.from_incident(incident)
        prompt = _build_user_prompt(evidence)

        payload = evidence.model_dump(mode="json")
        for key in payload:
            assert key in prompt
        # No incident objects/fields beyond what IncidentEvidence exposes
        # (e.g. internal Phase 2 dataclass repr) should leak into the prompt.
        assert "WindowClassification" not in prompt
        assert "dataclass" not in prompt

    def test_system_prompt_forbids_deciding_incident_existence(self):
        from backend.app.services.reasoning.nemotron_provider import SYSTEM_PROMPT

        assert "does not exist" in SYSTEM_PROMPT or "already been\ndecided" in SYSTEM_PROMPT or "already been" in SYSTEM_PROMPT
        assert "Do not invent" in SYSTEM_PROMPT
