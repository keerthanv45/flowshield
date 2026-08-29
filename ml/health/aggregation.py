"""
Time-window aggregation for the Payment Health Engine.

Turns raw PaymentEvent rows into per-time-window metrics, optionally
grouped by bank, payment_method, region, or combinations of those. This
is the first stage of the Phase 2 pipeline:

    Raw Payment Events -> Time Window Aggregation -> Payment Health Metrics -> ...

Design notes:
  - Windows are fixed-size, non-overlapping, wall-clock buckets (default
    15 minutes), using pandas' `Grouper`. Only windows that actually
    contain events appear in the output — there is no zero-filling for
    empty windows, since callers (baseline/anomaly) work off a
    (window_start, *group_cols) key and can decide how to handle gaps.
  - Failure-reason rates are expressed as a fraction of ALL transactions
    in the window (not just failed ones), so `failure_rate` always equals
    the sum of the seven `*_rate` columns. That keeps them directly
    comparable to `success_rate`/`failure_rate` and avoids a second,
    inconsistent denominator.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAILURE_REASONS = [
    "timeout",
    "network_error",
    "insufficient_funds",
    "authentication_failed",
    "bank_declined",
    "technical_error",
    "unknown",
]

# Canonical metric columns produced by aggregation, in a stable order.
METRIC_COLUMNS = [
    "transaction_count",
    "successful_count",
    "failed_count",
    "success_rate",
    "failure_rate",
    "total_amount",
    "successful_amount",
    "failed_amount",
    "average_amount",
    "median_amount",
    "average_latency_ms",
    "p95_latency_ms",
] + [f"{reason}_rate" for reason in FAILURE_REASONS]

# The standard set of groupings the health engine needs. `None` means the
# overall (ungrouped) window. Deliberately excludes combinations not asked
# for (e.g. bank + region) to avoid an explosion of sparse groups.
STANDARD_GROUPINGS: dict[str, list[str] | None] = {
    "overall": None,
    "bank": ["bank"],
    "payment_method": ["payment_method"],
    "region": ["region"],
    "bank_payment_method": ["bank", "payment_method"],
    "region_payment_method": ["region", "payment_method"],
}

DEFAULT_WINDOW_MINUTES = 15


def _compute_window_metrics(group: pd.DataFrame) -> pd.Series:
    n = len(group)
    success_mask = group["status"] == "success"
    failed_mask = group["status"] == "failed"

    successful_count = int(success_mask.sum())
    failed_count = int(failed_mask.sum())

    metrics = {
        "transaction_count": n,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "success_rate": successful_count / n if n else 0.0,
        "failure_rate": failed_count / n if n else 0.0,
        "total_amount": float(group["amount"].sum()),
        "successful_amount": float(group.loc[success_mask, "amount"].sum()),
        "failed_amount": float(group.loc[failed_mask, "amount"].sum()),
        "average_amount": float(group["amount"].mean()) if n else 0.0,
        "median_amount": float(group["amount"].median()) if n else 0.0,
        "average_latency_ms": float(group["latency_ms"].mean()) if n else 0.0,
        "p95_latency_ms": float(group["latency_ms"].quantile(0.95)) if n else 0.0,
    }

    for reason in FAILURE_REASONS:
        count = int((group["failure_reason"] == reason).sum())
        metrics[f"{reason}_rate"] = count / n if n else 0.0

    return pd.Series(metrics)


def aggregate_time_windows(
    df: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate raw payment events into per-time-window metrics.

    Args:
        df: raw payment events with at least `timestamp`, `status`,
            `amount`, `latency_ms`, `failure_reason`, and any columns in
            `group_cols`.
        window_minutes: fixed window size in minutes.
        group_cols: additional columns to group by within each time
            window (e.g. `["bank"]`, `["bank", "payment_method"]`).
            `None` produces the overall (ungrouped) aggregation.

    Returns:
        DataFrame with one row per (window_start, *group_cols), a
        `window_start` and `window_end` column, the grouping columns (if
        any), and the metric columns defined in `METRIC_COLUMNS`.
    """
    if df.empty:
        cols = ["window_start", "window_end"] + (group_cols or []) + METRIC_COLUMNS
        return pd.DataFrame(columns=cols)

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])

    grouper_keys = [pd.Grouper(key="timestamp", freq=f"{window_minutes}min")]
    if group_cols:
        grouper_keys += group_cols

    grouped = working.groupby(grouper_keys, observed=True)
    result = grouped.apply(_compute_window_metrics, include_groups=False).reset_index()

    result = result.rename(columns={"timestamp": "window_start"})
    result["window_end"] = result["window_start"] + pd.Timedelta(minutes=window_minutes)

    ordered_cols = ["window_start", "window_end"] + (group_cols or []) + METRIC_COLUMNS
    return result[ordered_cols].sort_values("window_start").reset_index(drop=True)


def aggregate_all_standard_groupings(
    df: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> dict[str, pd.DataFrame]:
    """Run `aggregate_time_windows` for every grouping in STANDARD_GROUPINGS."""
    return {
        name: aggregate_time_windows(df, window_minutes=window_minutes, group_cols=cols)
        for name, cols in STANDARD_GROUPINGS.items()
    }


@dataclass
class AggregationConfig:
    window_minutes: int = DEFAULT_WINDOW_MINUTES
