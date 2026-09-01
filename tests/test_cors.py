"""
Regression tests for CORS configuration in backend/app/main.py.
Covers the production incident: the deployed frontend
(https://flowshield-dashboard.onrender.com) got no
Access-Control-Allow-Origin header and the browser blocked all API
calls, even though the endpoints worked fine via direct navigation
(GET requests typed into the browser bar aren't subject to CORS --
only cross-origin fetch()/XHR from JS is).
"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

PRODUCTION_FRONTEND_ORIGIN = "https://flowshield-dashboard.onrender.com"
LOCAL_DEV_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]


class TestCorsProductionOrigin:
    def test_production_frontend_origin_allowed(self):
        r = client.get("/api/v1/summary", headers={"Origin": PRODUCTION_FRONTEND_ORIGIN})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == PRODUCTION_FRONTEND_ORIGIN

    def test_preflight_request_allowed_for_production_origin(self):
        r = client.options(
            "/api/v1/summary",
            headers={
                "Origin": PRODUCTION_FRONTEND_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == PRODUCTION_FRONTEND_ORIGIN

    def test_credentials_allowed(self):
        r = client.get("/api/v1/summary", headers={"Origin": PRODUCTION_FRONTEND_ORIGIN})
        assert r.headers.get("access-control-allow-credentials") == "true"


class TestCorsLocalDevOriginsPreserved:
    def test_localhost_origins_still_allowed(self):
        for origin in LOCAL_DEV_ORIGINS:
            r = client.get("/api/v1/summary", headers={"Origin": origin})
            assert r.headers.get("access-control-allow-origin") == origin


class TestCorsNotWildcard:
    def test_unrelated_origin_not_allowed(self):
        r = client.get("/api/v1/summary", headers={"Origin": "https://some-random-site.com"})
        assert r.headers.get("access-control-allow-origin") is None

    def test_no_wildcard_origin_configured(self):
        from backend.app.main import _cors_origins

        assert "*" not in _cors_origins
