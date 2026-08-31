"""
Revenue-at-risk calculation for a confirmed Phase 2 incident.

Scopes ACTUAL synthetic events (never invented) to the incident's time
window, and to its affected dimensions if any (bank/payment_method/
region-targeted incidents only see their own slice; broad incidents
like LATENCY_SPIKE/MERCHANT_SYSTEM_DEGRADATION with no dimension target
see the whole window).

============================================================
ASSUMED RECOVERY RATES (documented assumption, not fabricated silently)
============================================================
Sourced directly from `docs/data_dictionary.md`'s "Recovery potential"
table (High/Medium/Low), translated into indicative point estimates for
this simulation:

  timeout               0.55  (High  -- often transient, retry/reroute succeeds)
  network_error         0.55  (High  -- often transient)
  technical_error       0.35  (Medium -- sometimes transient, sometimes systemic)
  authentication_failed 0.25  (Medium -- may need a different auth flow/method)
  insufficient_funds    0.08  (Low   -- "temporarily retryable" but rarely
                                        succeeds soon; per Phase 4 brief,
                                        NOT treated as zero)
  bank_declined         0.03  (Low   -- genuine decline; should NOT be
                                        blindly retried)
  unknown               0.05  (Low   -- insufficient information to act
                                        confidently)

These are illustrative point estimates for a simulated decision engine,
not measured real-world recovery rates. `RECOVERABLE_THRESHOLD` (0.20)
separates "genuinely worth attempting" (timeout, network_error,
technical_error, authentication_failed) from "not worth attempting by
default" (insufficient_funds, bank_declined, unknown) for the
`recoverable_transactions`/`recoverable_amount` fields -- the LOW-rate
reasons still contribute to `expected_recovered_amount` (the
probability-weighted figure), just not to the "recoverable pool".
"""

from __future__ import annotations

import pandas as pd

from backend.app.services.recovery.schemas import FailureBreakdownEntry, RevenueRisk
from ml.health.incidents import Incident

ASSUMED_RECOVERY_RATE: dict[str, float] = {
    "timeout": 0.55,
    "network_error": 0.55,
    "technical_error": 0.35,
    "authentication_failed": 0.25,
    "insufficient_funds": 0.08,
    "bank_declined": 0.03,
    "unknown": 0.05,
}

RECOVERABLE_THRESHOLD = 0.20


def _scope_mask(events_df: pd.DataFrame, incident: Incident) -> pd.Series:
    timestamps = pd.to_datetime(events_df["timestamp"])
    mask = (timestamps >= incident.window_start) & (timestamps < incident.window_end)

    if not incident.affected_dimensions:
        return mask

    dim_mask = pd.Series(False, index=events_df.index)
    for dims in incident.affected_dimensions:
        sub_mask = pd.Series(True, index=events_df.index)
        for key, value in dims.items():
            if key in events_df.columns:
                sub_mask &= events_df[key] == value
        dim_mask |= sub_mask
    return mask & dim_mask


def compute_revenue_risk_for_failed(scope_id: str, failed: pd.DataFrame) -> RevenueRisk:
    """Shared computation: given a DataFrame already filtered to the
    failed transactions in scope (any scope -- one incident, or an
    entire dataset batch), build the RevenueRisk breakdown. `scope_id`
    becomes `RevenueRisk.incident_id` (a scope label, not necessarily a
    real incident -- e.g. Phase 6's batch evaluation uses a synthetic
    scope id here since it isn't tied to one confirmed incident).
    """
    transactions_at_risk = int(len(failed))
    gross_amount_at_risk = float(failed["amount"].sum())

    breakdown: list[FailureBreakdownEntry] = []
    recoverable_transactions = 0
    recoverable_amount = 0.0
    expected_recovered_amount = 0.0

    for reason, group in failed.groupby("failure_reason"):
        rate = ASSUMED_RECOVERY_RATE.get(reason, 0.0)
        count = int(len(group))
        amount = float(group["amount"].sum())

        breakdown.append(
            FailureBreakdownEntry(
                failure_reason=reason, count=count, amount=amount, assumed_recovery_rate=rate,
            )
        )
        expected_recovered_amount += amount * rate
        if rate >= RECOVERABLE_THRESHOLD:
            recoverable_transactions += count
            recoverable_amount += amount

    return RevenueRisk(
        incident_id=scope_id,
        transactions_at_risk=transactions_at_risk,
        gross_amount_at_risk=gross_amount_at_risk,
        recoverable_transactions=recoverable_transactions,
        recoverable_amount=recoverable_amount,
        expected_recovered_amount=expected_recovered_amount,
        failure_breakdown=breakdown,
    )


def calculate_revenue_risk(incident: Incident, events_df: pd.DataFrame) -> RevenueRisk:
    """Compute RevenueRisk for one confirmed incident from ACTUAL
    synthetic events (no invented values)."""
    scoped = events_df.loc[_scope_mask(events_df, incident)]
    failed = scoped.loc[scoped["status"] == "failed"]
    return compute_revenue_risk_for_failed(incident.incident_id, failed)
