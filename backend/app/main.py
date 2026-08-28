"""
FlowShield backend entrypoint.

Phase 1 scope: a minimal FastAPI app that exists so the project structure
is real and importable. It intentionally does NOT implement the health
engine, anomaly detection, recovery agent, or any Razorpay integration —
those are later phases. For now it exposes a health check and a stub
endpoint that reports what Phase 1 actually contains.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="FlowShield",
    description=(
        "Autonomous Payment Reliability & Revenue Protection Agent. "
        "Phase 1: project foundation, synthetic data, validation, and analysis. "
        "No live payment integration exists yet."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": 1}


@app.get("/")
def root() -> dict:
    return {
        "project": "FlowShield",
        "phase": "1 - foundation",
        "implemented": [
            "canonical payment event schema",
            "synthetic data generator",
            "controlled incident scenarios",
            "data validation",
            "basic analysis",
            "ml-ready train/validation/test splitting",
        ],
        "not_yet_implemented": [
            "payment health engine",
            "anomaly detection (ML)",
            "AI root cause analysis (Nemotron)",
            "revenue impact / counterfactual simulation",
            "recovery policy engine and execution",
            "Razorpay integration",
            "dashboard",
        ],
    }
