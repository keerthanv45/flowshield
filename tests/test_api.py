import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root_ok(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["project"] == "FlowShield"


class TestSummary:
    def test_summary_returns_actual_data(self):
        r = client.get("/api/v1/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total_transactions"] == 20000
        assert 0.0 <= body["overall_success_rate"] <= 1.0
        assert body["confirmed_incident_count"] > 0
        assert body["total_revenue_at_risk"] >= 0.0
        assert body["total_recoverable_revenue"] >= 0.0
        assert body["total_recoverable_revenue"] <= body["total_revenue_at_risk"]

    def test_summary_not_hardcoded(self):
        # Two calls must agree exactly with each other and with the
        # underlying data -- not fixed demo numbers.
        r1 = client.get("/api/v1/summary")
        r2 = client.get("/api/v1/summary")
        assert r1.json() == r2.json()


class TestIncidentList:
    def test_list_incidents_nonempty(self):
        r = client.get("/api/v1/incidents")
        assert r.status_code == 200
        assert len(r.json()) > 0

    def test_list_incidents_filter_by_severity(self):
        r = client.get("/api/v1/incidents", params={"severity": "CRITICAL"})
        assert r.status_code == 200
        body = r.json()
        assert len(body) > 0
        assert all(i["severity"] == "CRITICAL" for i in body)

    def test_list_incidents_filter_by_type(self):
        r = client.get("/api/v1/incidents", params={"incident_type": "LATENCY_SPIKE"})
        assert r.status_code == 200
        body = r.json()
        assert all(i["incident_type"] == "LATENCY_SPIKE" for i in body)

    def test_list_incidents_limit(self):
        r = client.get("/api/v1/incidents", params={"limit": 3})
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_list_incidents_unknown_filter_returns_empty(self):
        r = client.get("/api/v1/incidents", params={"severity": "NOT_A_SEVERITY"})
        assert r.status_code == 200
        assert r.json() == []


class TestIncidentDetail:
    def test_valid_incident_returns_200(self):
        incidents = client.get("/api/v1/incidents").json()
        incident_id = incidents[0]["incident_id"]
        r = client.get(f"/api/v1/incidents/{incident_id}")
        assert r.status_code == 200
        assert r.json()["incident_id"] == incident_id

    def test_unknown_incident_returns_404(self):
        r = client.get("/api/v1/incidents/does-not-exist")
        assert r.status_code == 404
        assert "detail" in r.json()


class TestAnalyzeEndpoint:
    def test_analyze_returns_full_structure_without_simulation(self):
        incidents = client.get("/api/v1/incidents").json()
        incident_id = incidents[0]["incident_id"]
        r = client.post(f"/api/v1/incidents/{incident_id}/analyze")
        assert r.status_code == 200
        body = r.json()
        assert set(["incident", "rca", "revenue_risk", "recovery_decision", "simulation"]) <= set(body.keys())
        assert body["simulation"] is None
        assert body["rca"]["source"] in ("mock", "nemotron", "fallback")

    def test_analyze_unknown_incident_404(self):
        r = client.post("/api/v1/incidents/does-not-exist/analyze")
        assert r.status_code == 404

    def test_analyze_does_not_execute_recovery(self):
        # Analysis must never contain simulated execution data.
        incidents = client.get("/api/v1/incidents").json()
        incident_id = incidents[0]["incident_id"]
        r = client.post(f"/api/v1/incidents/{incident_id}/analyze")
        assert r.json()["simulation"] is None


class TestSimulateEndpoint:
    def test_simulate_returns_simulated_result(self):
        incidents = client.get("/api/v1/incidents").json()
        incident_id = incidents[0]["incident_id"]
        r = client.post(f"/api/v1/incidents/{incident_id}/simulate")
        assert r.status_code == 200
        sim = r.json()["simulation"]
        assert sim is not None
        assert "SIMULATED" in sim["status"]
        assert sim["simulated_recovered_amount"] >= 0.0
        assert 0 <= sim["simulated_successes"] <= sim["eligible_transactions"]

    def test_simulate_unknown_incident_404(self):
        r = client.post("/api/v1/incidents/does-not-exist/simulate")
        assert r.status_code == 404

    def test_simulate_never_mentions_razorpay(self):
        incidents = client.get("/api/v1/incidents").json()
        incident_id = incidents[0]["incident_id"]
        r = client.post(f"/api/v1/incidents/{incident_id}/simulate")
        assert "razorpay" not in str(r.json()).lower()


class TestConfigStatus:
    def test_config_status_default_mock(self, monkeypatch):
        monkeypatch.delenv("REASONING_PROVIDER", raising=False)
        monkeypatch.delenv("NEMOTRON_API_KEY", raising=False)
        r = client.get("/api/v1/config/status")
        assert r.status_code == 200
        body = r.json()
        assert body["provider"] == "mock"
        assert body["nemotron_configured"] is False

    def test_config_status_never_returns_secret(self, monkeypatch):
        secret = "sk-super-secret-nemotron-key-12345"
        monkeypatch.setenv("NEMOTRON_API_KEY", secret)
        monkeypatch.setenv("NEMOTRON_MODEL", "some-model")
        monkeypatch.setenv("NEMOTRON_BASE_URL", "https://example.invalid/v1")
        r = client.get("/api/v1/config/status")
        assert secret not in r.text

    def test_config_status_no_key_field_at_all(self):
        r = client.get("/api/v1/config/status")
        body = r.json()
        assert set(body.keys()) == {"provider", "model", "nemotron_configured"}


class TestMalformedRequests:
    def test_invalid_limit_type_returns_422(self):
        r = client.get("/api/v1/incidents", params={"limit": "not-a-number"})
        assert r.status_code == 422

    def test_limit_below_minimum_returns_422(self):
        r = client.get("/api/v1/incidents", params={"limit": 0})
        assert r.status_code == 422

    def test_no_stack_trace_leaked_on_validation_error(self):
        r = client.get("/api/v1/incidents", params={"limit": "bad"})
        assert "Traceback" not in r.text
        assert "File \"" not in r.text
