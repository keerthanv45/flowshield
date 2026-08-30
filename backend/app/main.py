"""
FlowShield backend entrypoint.

Phase 5A: exposes the existing Phase 2/3/4 pipeline (health/incident
engine, reasoning provider, revenue recovery) via a FastAPI API. No
business logic lives here -- see backend/app/services/orchestrator.py
and backend/app/routes/v1.py.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.v1 import router as v1_router

app = FastAPI(
    title="FlowShield",
    description=(
        "Autonomous Payment Reliability & Revenue Protection Agent. "
        "Phase 5A: FastAPI orchestration over the Phase 2 health/incident "
        "engine, Phase 3 reasoning layer, and Phase 4 revenue recovery "
        "engine. No live payment integration exists yet."
    ),
    version="0.5.0",
)

_cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else ["http://localhost:3000", "http://localhost:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": 5}


@app.get("/")
def root() -> dict:
    return {
        "project": "FlowShield",
        "phase": "5A - FastAPI orchestration",
        "implemented": [
            "canonical payment event schema",
            "synthetic data generator",
            "controlled incident scenarios",
            "data validation",
            "basic analysis",
            "ml-ready train/validation/test splitting",
            "payment health engine + anomaly detection + incident detection/classification",
            "Nemotron reasoning layer (mock + real provider, safe fallback)",
            "revenue-at-risk + recovery policy engine + simulated execution",
            "FastAPI API over the above (this phase)",
        ],
        "not_yet_implemented": [
            "frontend/dashboard",
            "Razorpay live integration",
            "persistent database",
            "deployment",
        ],
    }

