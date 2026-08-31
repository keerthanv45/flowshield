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

**PHASE 2 — Payment Health Engine + Baseline + Anomaly Detection** (see below — implemented)

---

# Phase 2 — Payment Health Engine

**Status:** Complete. Builds on Phase 1's synthetic dataset; does not modify Phase 1's schema, generator, or validation logic.

This is evaluated entirely on SYNTHETIC payment data generated in Phase 1.
Nothing here reflects real production payment performance.

## Pipeline

```
Raw Payment Events
    -> Time Window Aggregation      (ml/health/aggregation.py)
    -> Payment Health Metrics       (ml/health/aggregation.py)
    -> Expected Baseline            (ml/health/baseline.py)
    -> Deviation Detection          (ml/health/features.py)
    -> Anomaly Score                (ml/health/anomaly.py)
    -> Incident Detection           (ml/health/incidents.py)
    -> Incident Classification      (ml/health/incidents.py)
    -> Health Snapshot              (ml/health/scoring.py)
```

`ml/health/pipeline.py` wires all of the above together (`run_health_pipeline`), and is the single place `scripts/evaluate_phase2.py` and the test suite build the pipeline from — avoids re-deriving the wiring three times.

**AI recommends, policy authorizes, execution acts, outcome measures** — Phase 2 stops at "detects and classifies". No recovery action is taken or even proposed here (that's Phase 3+). No LLM is used anywhere in this phase — every score, signal, and classification is a deterministic function of aggregated metrics.

## Time-window aggregation

Default window: **15 minutes**. `ml/health/aggregation.py` groups raw events by a fixed time bucket (optionally + dimension columns) and computes: transaction/success/failure counts and rates, total/successful/failed/average/median amount, average and p95 latency, and seven failure-reason rates (each a fraction of *all* transactions in the window, so they sum exactly to `failure_rate`).

Supported groupings (`STANDARD_GROUPINGS`): overall, bank, payment_method, region, bank+payment_method, region+payment_method — the six combinations actually needed; no combinatorial explosion.

**Dataset-scale adaptation:** the default 15-minute window works well for headline metrics, but at this dataset's volume (~20k events/14 days ⇒ ~15 events/15-min window overall), a bank+payment_method slice gets under 1 event per 15-minute window on average — far too sparse for reliable concentration analysis. This was caught in development: BANK_RAIL_DEGRADATION was never confirmed even during the real injected bank-rail incident, because the (HDFC, UPI) slice almost always had 0-1 transactions per 15-minute window. Fix: dimension-**concentration** analysis (used only for classification, not the headline health score) runs on a coarser 2-hour aggregation instead, giving each slice enough volume to judge degradation reliably (`ml/health/incidents.py::CONCENTRATION_WINDOW_MINUTES`).

## Baseline methodology

`ml/health/baseline.py`. Learns expected success rate, failure rate, latency, and volume **only from NORMAL (non-incident) windows in the TRAIN split** — see Data leakage prevention below. Temporal awareness: baseline is keyed by hour-of-day (0-23); day-of-week was considered but rejected as a key — with only 14 days of data that's ≤2 samples per weekday per hour, too thin to estimate reliably. Falls back hourly cell → group-level average → global average when a cell has too few samples (`MIN_SAMPLES_FOR_HOURLY_BASELINE = 3`).

Also tracks the **empirical standard deviation** of each metric per cell (not just the mean) — this is what statistical-significance gating in the incident detector uses (see below).

## Health score methodology

`ml/health/scoring.py`. A deterministic, explainable 0-100 score from four documented, weighted components:

| Component | Weight | Rationale |
|---|---|---|
| Success-rate health | 40% | Most direct revenue-impact signal |
| Latency health | 25% | Often precedes/accompanies failures; matters even when payments still succeed |
| Failure-pattern health | 20% | Distinguishes normal failure mix from a systemic (timeout/network/technical) surge |
| Volume health | 15% | Weakest standalone signal (legitimate traffic varies), but a severe drop matters |

Each component is bounded [0,100] by its own documented formula (ratio-to-baseline for success rate/volume, linear falloff for latency ratio and systemic-failure-rate deviation — see full derivations and floor/ceiling constants in the `ml/health/scoring.py` module docstring). Status thresholds: `HEALTHY >= 80`, `DEGRADED >= 50`, else `CRITICAL` — chosen so one dominant component failing pulls the score into DEGRADED but not automatically CRITICAL; CRITICAL needs either a severe hit on the highest-weighted component or multiple components degrading together.

## Anomaly detection methodology

`ml/health/anomaly.py`. Isolation Forest (scikit-learn) over **aggregated window features** — never raw transaction rows (`ANOMALY_FEATURES`: success/failure rate, p95 latency, transaction count, 3 systemic failure-reason rates, success_rate_delta, latency_ratio, volume_ratio). Features are standardized (`StandardScaler`, fit on the same training data) before fitting. `n_estimators=200`, `contamination=0.05`, `random_state=42` — fully deterministic given a fixed seed (verified by a dedicated test).

**Fit ONLY on NORMAL windows in the TRAIN split** (same exclusion rule as the baseline). `anomaly_score` is **not** a calibrated probability — it's `-score_samples()`, min-max normalized against the *training* data's own score range (0 ≈ typical of training data; can exceed 1 for windows more extreme than anything seen while fitting). `is_anomaly` uses IsolationForest's own `contamination`-based threshold. A separate `low_reliability` flag (not a "confidence") marks windows below `MIN_RELIABLE_VOLUME = 5` transactions.

## Incident detection & classification methodology

`ml/health/incidents.py`. **No LLM anywhere in this module** — every signal, threshold, and classification decision is a fixed, documented rule.

**Rule-based signals** (deterministic, each independently explainable): `SUCCESS_RATE_DEGRADATION`, `LATENCY_SPIKE`, `FAILURE_SURGE`, `VOLUME_ANOMALY`.

**Statistical significance gating (false-positive fix):** an early version gated `SUCCESS_RATE_DEGRADATION`/`FAILURE_SURGE` with a fixed percentage-point threshold only, and — measured directly during development — flagged noise on days with *no* injected incident at all (up to ~20% of windows). Root cause: at ~15 transactions/window, a proportion estimate has real sampling noise, and each window blends multiple payment methods/banks with different true success rates (overdispersion), so even a parametric binomial standard error under-estimated the true variance (still ~8% false-positive rate on pure-normal data). Fix actually shipped: gate on a z-score against the baseline's own **empirical** standard deviation of historical normal-window values (`ml.health.baseline.BaselineModel` tracks std per cell), not a formula — `RATE_SIGNAL_Z_THRESHOLD = 2.0` for the single overall-level test. Concentration analysis checks ~20 (bank, payment_method) combos per bucket independently, which inflates the family-wise false-positive rate if tested at the same threshold (~35-40% chance at least one combo looks "degraded" by chance per bucket, measured directly) — so member-level degradation inside concentration analysis uses a stricter, roughly Bonferroni-adjusted `CONCENTRATION_Z_THRESHOLD = 2.8`.

**Classification** (deterministic cascade, evaluated in order — see module docstring for full detail):
1. No signal, not anomalous → `NORMAL`
2. Small, concentrated (bank, payment_method) slice degraded, bank/method-level concentration low → `BANK_RAIL_DEGRADATION`
3. Exactly one region degraded → `REGIONAL_DEGRADATION`
4. Latency the dominant signal, degradation broad across banks & methods → `LATENCY_SPIKE`
5. Broad degradation across banks & methods, latency not dominant → `MERCHANT_SYSTEM_DEGRADATION`
6. Any remaining signal/anomaly without concentration → `ISOLATED_FAILURES`

**Severity** (deterministic): `CRITICAL` if `health_score < 30` or `success_rate_delta <= -0.30`; `WARNING` if `health_score < 60` or 2+ signals fired; else `INFO`.

**Persistence (false-positive handling):** consecutive same-type windows are grouped into "episodes"; an episode is only CONFIRMED (returned as an `Incident`) if it spans ≥2 consecutive windows OR any window in it is independently CRITICAL. A single noisy, non-extreme window never becomes an Incident.

**Evidence:** every non-NORMAL classification gets dynamically generated, human-readable evidence lines (observed vs. baseline for every metric, plus dimension-specific detail) — see `generate_evidence()`. Nothing is hardcoded; all values come from the actual window's computed metrics.

## Data leakage prevention

- **Baseline** fit ONLY on rows chronologically before a single train-cutoff timestamp (first 70% of the headline aggregation, matching `ml/evaluation/split.py`'s ratios), applied consistently to every grouping via `pipeline.compute_train_cutoff`/`_train_slice` — a sparser grouping (e.g. bank_payment_method) cannot pull in its own later rows just because it has fewer total rows. Only non-incident rows within that cutoff are used (`label_incident_affected`). *(Fixed: an earlier version fit baseline on the full aggregated dataframe before splitting, letting validation/test windows influence baseline statistics — caught by `tests/test_pipeline_leakage.py`.)*
- **Anomaly detector** fit on that exact same TRAIN + non-incident subset.
- Both are then used to **score** every window (train, validation, test) — scoring never refits, so this introduces no leakage; it's applying an already-fixed model.
- `scripts/evaluate_phase2.py` reports metrics twice: once over the full dataset (train windows are in-sample for the fitted models, though ground truth is never fed to the detector/classifier itself) and once restricted to the **held-out validation+test period only** (genuinely out-of-sample).
- Ground-truth incident windows are used ONLY to (a) exclude data from baseline/anomaly training and (b) score the evaluation — never as an input to detection or classification itself.

## Evaluation methodology & actual results

`ml/evaluation/incident_evaluation.py` + `scripts/evaluate_phase2.py`, run against the real 20,000-event Phase 1 dataset (not fabricated — see the message that shipped this code for the exact run). Every 15-minute window gets a ground-truth label from `data/synthetic/incident_windows.json`. Metrics: 6-class classification accuracy (raw per-window prediction vs. ground truth), and — evaluated on CONFIRMED (post-persistence) incidents, since that's what would actually reach a human — precision/recall/F1 on the binary "is this a systemic incident" task (systemic = anything except NORMAL/ISOLATED_FAILURES), plus false-positive rate and per-scenario detection/classification breakdowns.

**Actual measured results (full dataset, 1,344 windows):**

| Metric | Value |
|---|---|
| Classification accuracy (6-class) | 0.5350 |
| Detection rate (recall, systemic) | 0.5222 |
| False positive rate | 0.0229 |
| Precision | 0.9947 |
| Recall | 0.5222 |
| F1 | 0.6849 |

**Per-scenario (full dataset):**

| Scenario | Ground-truth windows | Confirmed (any type) | Confirmed (correct type) |
|---|---|---|---|
| bank_rail_degradation | 96 | 11 (11.5%) | 9 (9.4%) |
| regional_degradation | 144 | 11 (7.6%) | 8 (5.6%) |
| latency_spike | 96 | 96 (100%) | 96 (100%) |
| merchant_system_degradation | 144 | 134 (93.1%) | 59 (41.0%) |
| isolated_failures | 144 | 4 (2.8%) | 0 (0.0%) |

Anomaly detector: 301 windows flagged `is_anomaly` (267 during a known incident, 34 during normal traffic).

**Held-out (validation+test period, 404 windows, genuinely out-of-sample):** classification accuracy 0.2772; precision 0.9847, recall 0.5375, F1 0.6954. The incident schedule (`ml/data_generation/generator.py`) now includes a second, smaller occurrence of each systemic scenario after the train/val/test cutoff (~day 9.8), additive to the original train-period occurrences — so held-out evaluation has genuine systemic ground truth (48-72 windows per scenario) instead of the vacuous 0/0 recall from the original schedule. Per-scenario held-out: bank_rail 12.5% detected, regional 15.3%, latency_spike 100%, merchant_system 88.9%, isolated_failures 1.4% confirmed (none CRITICAL — false-positive control intact).

**Reading these numbers honestly:** precision is very high (0.99) — when the system confirms an incident, it's almost always right, and `isolated_failures` is correctly kept from flooding CRITICAL alerts (this was the core Phase 2 false-positive requirement). Recall is uneven: `latency_spike` (system-wide, overwhelming signal) and `merchant_system_degradation` (broad) are detected well; `bank_rail_degradation` and `regional_degradation` (concentrated in a small slice of the traffic) are detected much less often per-window, even though the *day* is correctly identified as having activity there (see `data/synthetic/phase2_report.png`) — persistence + concentration-window granularity mean many individual 15-minute windows within a correctly-identified incident day still don't individually cross the confirmation bar. See Known limitations.

## How to run Phase 2

```bash
python scripts/generate_data.py        # if not already run (Phase 1)
python scripts/evaluate_phase2.py      # runs the full pipeline + evaluation
pytest -v                              # Phase 1 + Phase 2 tests
```

Outputs: `data/synthetic/phase2_incidents.json` (confirmed incidents), `phase2_health_snapshots.csv` (per-window snapshot), `phase2_evaluation.json` (metrics), `phase2_report.png` (health score / success rate / latency / anomaly score over time, with ground-truth and confirmed-incident windows shaded).

## Known limitations (Phase 2)

- Recall for concentrated scenarios (bank-rail, regional) is modest (7-9%) at the per-window level, even though the affected day is visibly identifiable in the health/anomaly charts — the concentration-window/persistence combination is conservative by design (favors precision over recall) and would benefit from tuning in a later phase.
- `CONCENTRATION_WINDOW_MINUTES = 120` and the Bonferroni-style `CONCENTRATION_Z_THRESHOLD = 2.8` were chosen empirically against this dataset's specific volume (~20k events/14 days) and thresholds/schedule — they are not universal constants and would need re-tuning at a different data scale.
- `failure_pattern_health` approximates a per-reason baseline by scaling the window's own systemic-failure share against the baseline's *overall* failure rate (documented in `scoring.py`), rather than tracking each failure reason's baseline independently — a reasonable simplification, not a per-reason baseline.
- The held-out evaluation period contains no systemic-scenario ground truth by construction of the fixed 5-window incident schedule (see Evaluation results above) — genuine out-of-sample recall for bank-rail/regional/latency/merchant scenarios specifically was not measurable with this exact schedule; only `isolated_failures` (false-positive control) is represented out-of-sample.
- Pipeline runtime (~95s for the 20k-event dataset) is dominated by per-row `DataFrame.apply()` calls in feature/health-score computation — fine for a batch script, not yet optimized for a low-latency serving path.

## Next phase

**PHASE 3 — Nemotron reasoning layer + root-cause analysis + revenue-at-risk estimation**

---

# Phase 6 — Batch Recovery Evaluation

Proves recovery works at scale: runs Phase 4's revenue-at-risk and simulator logic across **every failed transaction in the full 20,000-event synthetic dataset** (not scoped to a single incident), and reports one aggregate business-impact result.

**How the batch is selected:** every row in `data/synthetic/events.csv` with `status == "failed"` — the entire dataset, not a subset of incidents. `total_transactions` in the report is the full 20,000.

**How each failure reason is treated** (reuses Phase 4's existing guardrail sets from `candidates.py`/`policy.py` unchanged, just applied per failure-reason group instead of per single incident):
- `timeout`, `network_error`, `technical_error` → automated RETRY/route, simulated (system-side, no customer action needed)
- `authentication_failed` → **excluded** from this automated batch (Phase 4 assumption: requires customer action, so a batch job can't execute it)
- `insufficient_funds` → **deferred** to WAIT_AND_MONITOR, never simulated as an attempt
- `bank_declined`, `unknown` → **excluded**, never retried

**Definitions — never presented as guaranteed real-world outcomes:**
- *Revenue at risk*: actual gross amount of all failed transactions (fact, from data).
- *Expected recovery*: probability-weighted estimate using Phase 1's documented, illustrative `ASSUMED_RECOVERY_RATE` table (not measured real-world rates).
- *Simulated recovered revenue*: one deterministic (seed=42) Monte Carlo draw against those same assumed rates — a simulation output, not a guarantee.
- `recovery_rate` = recovered / attempted; `revenue_recovery_rate` = recovered amount / **total** revenue at risk (a stricter, whole-batch denominator).

**Endpoint:** `GET /api/v1/recovery/evaluation` (reuses `orchestrator.batch_recovery_evaluation()` → `backend/app/services/recovery/batch_evaluation.py`; no duplicate logic in the route).

**CLI:** `python scripts/evaluate_batch_recovery.py` — prints the same result as a report.

**Dashboard:** compact "Batch Recovery Performance" panel (revenue at risk, expected recovery, simulated recovered revenue, recovery rate, transactions recovered, guardrail counts), clearly labeled SIMULATED.

**Known limitation:** the ALTERNATE_PAYMENT_METHOD/auth-failure exclusion and insufficient-funds deferral are batch-level simplifications (see module docstring in `batch_evaluation.py`) — per-incident Phase 4 analysis still offers those as real candidates when a human is in the loop; this batch view only simulates what could plausibly run unattended.
