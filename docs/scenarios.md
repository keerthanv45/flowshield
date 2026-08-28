# Controlled Incident Scenarios — FlowShield Phase 1

Implemented in `ml/data_generation/scenarios.py` (definitions) and
`ml/data_generation/generator.py` (`default_incident_schedule`, applied
during generation). The exact windows used for the default 14-day, 20,000
event dataset:

| Scenario | Window (days from period start) | Target | Effect summary |
|---|---|---|---|
| 1. Bank/rail degradation | 2.0 – 3.0 | UPI + HDFC | Success rate ×0.35, latency ×2.2, timeout/network weight boosted. Other banks unaffected. |
| 2. Regional degradation | 4.0 – 5.5 | Region KA | Success rate ×0.55, latency ×1.6, affects all methods/banks within KA. |
| 3. Latency spike | 6.0 – 7.0 | Global | Latency ×3.0, success rate ×0.70, timeout/network weight boosted, no method/bank/region targeting. |
| 4. Merchant/system degradation | 7.5 – 9.0 | Global | Success rate ×0.60, latency ×1.4, technical_error/bank_declined weight boosted — spread evenly, not concentrated in one bank. |
| 5. Isolated failures | 10.0 – 11.5 | None (dataset-wide, low rate) | Extra ~3% per-event failure probability applied uniformly, uncorrelated with method/bank/region — a false-positive test case for future anomaly detection. |
| 6. Normal traffic | everywhere else | — | Baseline behavior only. |

## Why time-windowed, not row-flagged

Incidents are modeled as time windows rather than a per-row "is this an
incident" label, because that mirrors how a real payment system would
actually be observed — you see degraded aggregate behavior over a period,
not a ground-truth label on each transaction. The windows and their exact
parameters (target dimensions, multipliers) are still written out to
`data/synthetic/incident_windows.json` at generation time, so they can be
used as ground truth to evaluate the anomaly/incident detector that Phase
2 will build — without that detector ever seeing the label during
"production" use.

## Distinguishing signatures at a glance

- **Scenario 1** only shows degradation when you slice by (method=UPI,
  bank=HDFC); every other slice looks normal during the same window.
- **Scenario 2** only shows degradation when you slice by region=KA, but
  *within* that slice, multiple methods and banks are affected.
- **Scenario 3** shows degradation in the aggregate/global numbers with no
  particular method, bank, or region standing out — and latency moves
  first/most.
- **Scenario 4** looks similar in shape to Scenario 3 (broad, not
  concentrated) but is driven by different failure reasons
  (`technical_error`, `bank_declined`) and a smaller latency effect.
- **Scenario 5** shows a small bump in the overall failure rate with no
  concentration in any single method/bank/region — deliberately meant to
  look like noise, not a systemic incident.
