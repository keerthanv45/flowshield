"""
Deviation features for the Payment Health Engine.

For every aggregated time window, compute how far its metrics deviate
from the baseline's expectation for that hour/group. These features feed
both the rule-based signals and the anomaly detector.
"""

from __future__ import annotations

import pandas as pd

from ml.health.baseline import BaselineModel

RATIO_EPSILON = 1e-6


def safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    """`numerator / denominator`, returning `default` if the denominator
    is ~0 (avoids inf/NaN on quiet windows or a zero baseline)."""
    if abs(denominator) < RATIO_EPSILON:
        return default
    return numerator / denominator


def compute_deviation_row(
    row: pd.Series,
    baseline: BaselineModel,
    group_values: tuple | None = None,
) -> dict[str, float]:
    """Compute deviation features for a single aggregated-window row."""
    expected = baseline.predict(row["window_start"], group_values=group_values)
    expected_std = baseline.predict_std(row["window_start"], group_values=group_values)

    success_rate_delta = row["success_rate"] - expected["success_rate"]
    failure_rate_delta = row["failure_rate"] - expected["failure_rate"]
    latency_delta = row["average_latency_ms"] - expected["average_latency_ms"]
    volume_delta = row["transaction_count"] - expected["transaction_count"]

    latency_ratio = safe_ratio(row["average_latency_ms"], expected["average_latency_ms"], default=1.0)
    volume_ratio = safe_ratio(row["transaction_count"], expected["transaction_count"], default=1.0)
    success_rate_ratio = safe_ratio(row["success_rate"], expected["success_rate"], default=1.0)

    # z-scores against the EMPIRICAL std of normal window-level values
    # (not a parametric/binomial formula — see BaselineModel docstring
    # for why). Used by ml.health.incidents for significance gating.
    success_rate_z = safe_ratio(success_rate_delta, expected_std["success_rate"], default=0.0)
    failure_rate_z = safe_ratio(failure_rate_delta, expected_std["failure_rate"], default=0.0)

    return {
        "baseline_success_rate": expected["success_rate"],
        "baseline_failure_rate": expected["failure_rate"],
        "baseline_average_latency_ms": expected["average_latency_ms"],
        "baseline_p95_latency_ms": expected["p95_latency_ms"],
        "baseline_transaction_count": expected["transaction_count"],
        "baseline_success_rate_std": expected_std["success_rate"],
        "baseline_failure_rate_std": expected_std["failure_rate"],
        "success_rate_delta": success_rate_delta,
        "failure_rate_delta": failure_rate_delta,
        "latency_delta": latency_delta,
        "volume_delta": volume_delta,
        "latency_ratio": latency_ratio,
        "volume_ratio": volume_ratio,
        "success_rate_ratio": success_rate_ratio,
        "success_rate_z": success_rate_z,
        "failure_rate_z": failure_rate_z,
    }


def add_deviation_features(
    aggregated_df: pd.DataFrame,
    baseline: BaselineModel,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Attach deviation-feature columns to an aggregated-window DataFrame."""
    if aggregated_df.empty:
        return aggregated_df.copy()

    def _row_features(row: pd.Series) -> pd.Series:
        group_values = tuple(row[c] for c in group_cols) if group_cols else None
        return pd.Series(compute_deviation_row(row, baseline, group_values=group_values))

    feature_df = aggregated_df.apply(_row_features, axis=1)
    return pd.concat([aggregated_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
