# FlowShield — Autonomous Payment Reliability & Revenue Protection Agent

**Track:** AI Revenue Recovery — Razorpay Buildathon 2026
**Status:** Phase 1 (foundation) complete. See [Known limitations](#known-limitations) and [Next phase](#next-phase).

FlowShield detects payment degradation, identifies likely root causes,
estimates revenue at risk, evaluates recovery strategies, applies
deterministic safety policies, executes bounded recovery actions, verifies
outcomes, and maintains an audit trail.

**Core principle:** AI recommends. The policy engine authorizes. The
execution layer acts. The outcome engine measures. The LLM never directly
authorizes or executes a financial action.

```
Payment Events → Payment Health Engine → Anomaly Detection → Incident
Detection → AI Root Cause Analysis → Revenue Impact Analysis →
Counterfactual Simulation → Policy / Guardrails → Recovery Execution →
Outcome Verification → Audit Trail → Evaluation Dashboard
```

## What Phase 1 actually is

Phase 1 builds the foundation only:

1. Project structure
2. Canonical payment-event schema (Pydantic)
3. Reproducible synthetic payment-event generator
4. Six controlled incident scenarios with distinguishable signatures
5. Data validation (row-level + dataset-level)
6. Basic descriptive analysis
7. ML-ready time-based train/validation/test splitting (no model trained)
8. Tests (pytest)

**Not implemented yet** (later phases): Nemotron/LLM integration, ML-based
anomaly detection, the recovery agent, counterfactual simulation, Razorpay
API integration, and the dashboard. See `backend/app/main.py`'s `/` route
for a live list.

Everything under `data/synthetic/` is **SYNTHETIC** — generated locally
with a fixed seed, not sourced from any real payment gateway.

## Project structure

```
flowshield/
├── backend/
│   └── app/
│       ├── models/         # (reserved for future DB models)
│       ├── schemas/        # payment_event.py — canonical PaymentEvent
│       ├── services/       # validation.py — data quality checks
│       └── main.py         # minimal FastAPI app (health check + status)
├── ml/
│   ├── data_generation/    # generator.py, scenarios.py
│   ├── analysis/           # report.py — descriptive analytics
│   └── evaluation/         # split.py — time-based train/val/test split
├── data/
│   └── synthetic/          # generated CSV/JSON output (gitignored)
├── scripts/                # CLI entrypoints (generate/validate/analyze/split)
├── tests/                  # pytest suite
├── docs/                   # data dictionary, scenario reference
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands to run locally

```bash
# 1. Generate the synthetic dataset (~20,000 events, reproducible)
python scripts/generate_data.py

# 2. Validate it
python scripts/validate_data.py

# 3. Run basic analysis (prints a report, also saves JSON)
python scripts/run_analysis.py

# 4. Prepare ML-ready time-based splits (no training happens)
python scripts/prepare_ml_splits.py

# 5. Run the test suite
pytest -v

# 6. (optional) Run the minimal API to confirm the backend skeleton boots
uvicorn backend.app.main:app --reload
```

### Expected outputs

- `data/synthetic/events.csv` — ~20,000 rows matching the `PaymentEvent` schema
- `data/synthetic/incident_windows.json` — ground-truth incident schedule (5 injected windows + implicit normal traffic)
- `data/synthetic/analysis_report.json` — the same report `run_analysis.py` prints, saved as JSON
- `data/synthetic/splits/{train,validation,test}.csv` — time-ordered splits (~70/15/15)
- `pytest -v` — all tests passing (see [Known limitations](#known-limitations) about execution)

## Dataset generation

`ml/data_generation/generator.py` builds each event from a baseline
behavior model (success rate by payment method × bank, latency by method,
failure-reason mix by method) and then layers **controlled incident
windows** on top. Generation is fully deterministic for a given `--seed`
(default `42`) — no wall-clock or OS randomness is used. See
`docs/data_dictionary.md` for the full field reference and the
correlations that are deliberately modeled (e.g. timeouts/network errors
raising latency, insufficient-funds failures not raising latency,
right-skewed transaction amounts, retries behaving differently from first
attempts).

## Incident scenarios

Six scenario types, each with a distinguishable signature. See
`docs/scenarios.md` for full detail and `ml/data_generation/scenarios.py`
for the implementation.

| # | Scenario | Signature |
|---|----------|-----------|
| 1 | Bank/rail degradation | One (method, bank) pair — e.g. UPI+HDFC — degrades hard; other banks stay normal |
| 2 | Regional degradation | One region degrades across multiple methods/banks |
| 3 | Latency spike | Global latency and timeout/network failures rise together |
| 4 | Merchant/system degradation | Broad degradation spread across banks & methods (not one bank) |
| 5 | Isolated failures | Scattered failures, no systemic pattern — false-positive test case |
| 6 | Normal traffic | No incident; baseline behavior only |

## Validation

`backend/app/services/validation.py` validates:
missing required fields, invalid enum values (status/failure reason/etc.),
invalid amounts, invalid latency, invalid attempt numbers, invalid
status/failure-reason combinations, and duplicate `event_id`s across the
dataset. `scripts/validate_data.py` runs this over the generated CSV and
prints a human-readable report with the first offending rows.

## Tests

`pytest -v` covers: valid/invalid payment events, amount/latency/attempt
validation, all failure categories, duplicate detection, reproducibility
(same seed ⇒ identical dataset), incident injection (each scenario's
signature is statistically present), and basic analysis correctness.

**This project was developed and tested inside an isolated Linux sandbox.**
All commands above were actually executed there, and `pytest` results were
observed directly rather than assumed — see the message that shipped this
code for the exact pass/fail counts at delivery time. Re-run `pytest -v`
yourself after transferring the project, since your local Python/OS
environment may differ.

## Known limitations

- No real payment gateway is involved anywhere in Phase 1; all data is synthetic.
- Incident windows are a small, fixed, hand-authored schedule (5 windows across
  a 14-day synthetic period) — enough to exercise every scenario signature at
  the ~20k-event scale, not a large-scale stress test.
- The FastAPI app exposes only a health check and a status endpoint; there is
  no health engine, anomaly detector, or any other business logic behind it yet.
- `backend/app/models/` is an empty placeholder for future persistence — Phase 1
  has no database.
- Recovery potential per failure reason (e.g. timeouts being more recoverable
  than insufficient-funds declines) is documented in `docs/data_dictionary.md`
  as a business rule for later phases, but is not yet a modeled field or used
  by any code.

## Next phase

**PHASE 2 — Payment Health Engine + Baseline + Anomaly Detection**
