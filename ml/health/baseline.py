"""
Baseline model for the Payment Health Engine.

Estimates "expected" payment behavior (success rate, failure rate,
latency, volume) so that later stages can compute deviations. The
baseline is deliberately learned ONLY from normal traffic — see
`normal_traffic.py`-equivalent logic below (`label_incident_affected`) —
because fitting a baseline on incident windows would make incidents look
"expected" and defeat the point of the detector.

Temporal awareness: expected behavior varies by hour of day (e.g. night
vs. peak hours), so the baseline is keyed by `hour_of_day` (0-23) within
each grouping. With only a 14-day synthetic period, day-of-week has at
most 2 samples per weekday per hour — too few to estimate reliably, so
day-of-week is NOT used as a separate key (documented simplification,
avoids overengineering on thin data). If a given (group, hour) cell has
too few samples, we fall back to the group's overall (hour-independent)
average, and if that's also empty, to the global (ungrouped, hour-
independent) average.

DATA LEAKAGE: `BaselineModel.fit` should only ever be called with the
TRAIN split (see `ml/evaluation/split.py`), and only rows that are not
inside a known incident window (see `label_incident_affected`). This
module does not enforce that by itself — the caller (the Phase 2
pipeline script) is responsible for passing the right slice. This keeps
the module testable without hard-wiring a specific split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

MIN_SAMPLES_FOR_HOURLY_BASELINE = 3

BASELINE_METRICS = [
    "success_rate",
    "failure_rate",
    "average_latency_ms",
    "p95_latency_ms",
    "transaction_count",
]


def label_incident_affected(
    window_start: pd.Series,
    window_end: pd.Series,
    incident_windows: list[dict[str, Any]],
    group_values: Optional[dict[str, pd.Series]] = None,
) -> pd.Series:
    """Vectorized-ish helper: returns a boolean Series, True where the
    aggregation row overlaps a known incident window.

    For a global incident (no target dimension), any overlapping window is
    affected. For a dimension-targeted incident (e.g. bank=HDFC,
    payment_method=UPI), a row is affected only if `group_values` matches
    ALL of the incident's specified target dimensions; if `group_values`
    is None (the "overall" / ungrouped aggregation), we conservatively
    treat any time-overlap as affected, since an ungrouped window's
    metrics are still partially contaminated by a dimension-targeted
    incident happening somewhere inside it.
    """
    affected = pd.Series(False, index=window_start.index)

    for w in incident_windows:
        w_start = pd.Timestamp(w["start"])
        w_end = pd.Timestamp(w["end"])
        time_overlap = (window_start < w_end) & (window_end > w_start)
        if not time_overlap.any():
            continue

        has_target = any(
            w.get(k) for k in ("target_payment_method", "target_bank", "target_region")
        )

        if not has_target:
            affected |= time_overlap
            continue

        if group_values is None:
            # Conservative: ungrouped/overall row, any overlap counts.
            affected |= time_overlap
            continue

        dim_match = pd.Series(True, index=window_start.index)
        if w.get("target_payment_method") and "payment_method" in group_values:
            dim_match &= group_values["payment_method"] == w["target_payment_method"]
        if w.get("target_bank") and "bank" in group_values:
            dim_match &= group_values["bank"] == w["target_bank"]
        if w.get("target_region") and "region" in group_values:
            dim_match &= group_values["region"] == w["target_region"]

        affected |= time_overlap & dim_match

    return affected


@dataclass
class BaselineCell:
    n_samples: int
    values: dict[str, float]
    stds: dict[str, float] = field(default_factory=dict)


# Floor applied to any baseline std used as a z-test denominator, as a
# fraction of the metric's own value (e.g. 5% of the baseline success
# rate). Prevents a division-by-near-zero blowing up z-scores for
# metrics that happen to be extremely stable in a given cell.
MIN_STD_FRACTION = 0.02


def _relative_floor(value: float) -> float:
    return max(abs(value) * MIN_STD_FRACTION, 1e-6)


@dataclass
class BaselineModel:
    """Baseline expected-behavior lookup for one grouping (e.g. overall,
    or bank+payment_method).

    Tracks both the MEAN and the empirical STANDARD DEVIATION of each
    metric across normal windows in a cell. The std is used (via
    `predict_std`) as the denominator for statistical-significance gating
    in `ml.health.incidents` — deliberately an EMPIRICAL std of
    window-level values, not a parametric (e.g. binomial) formula, because
    each window blends multiple payment methods/banks/attempt numbers
    with different underlying success rates; that mixture has more
    window-to-window variance than a simple binomial(n, p) model would
    predict (verified during development — a binomial-only z-test flagged
    normal traffic far too often). The empirical std captures this
    "real" variability directly from observed normal windows.
    """

    group_cols: list[str] | None
    hourly_cells: dict[tuple, BaselineCell] = field(default_factory=dict)
    group_fallback: dict[tuple, BaselineCell] = field(default_factory=dict)
    global_fallback: Optional[BaselineCell] = None

    def fit(self, normal_df: pd.DataFrame) -> "BaselineModel":
        """Fit on a DataFrame of NORMAL (non-incident) aggregated windows
        for this grouping. Expects a `window_start` column and the
        `BASELINE_METRICS` columns, plus `group_cols` if any.
        """
        if normal_df.empty:
            return self

        working = normal_df.copy()
        working["hour_of_day"] = pd.to_datetime(working["window_start"]).dt.hour

        keys = (self.group_cols or []) + ["hour_of_day"]
        hourly = working.groupby(keys, observed=True)[BASELINE_METRICS].agg(["mean", "std", "count"])
        for idx, row in hourly.iterrows():
            n = int(row[(BASELINE_METRICS[0], "count")])
            values = {m: float(row[(m, "mean")]) for m in BASELINE_METRICS}
            stds = {m: self._safe_std(row[(m, "std")], values[m]) for m in BASELINE_METRICS}
            key = idx if isinstance(idx, tuple) else (idx,)
            self.hourly_cells[key] = BaselineCell(n_samples=n, values=values, stds=stds)

        if self.group_cols:
            group_level = working.groupby(self.group_cols, observed=True)[BASELINE_METRICS].agg(
                ["mean", "std", "count"]
            )
            for idx, row in group_level.iterrows():
                n = int(row[(BASELINE_METRICS[0], "count")])
                values = {m: float(row[(m, "mean")]) for m in BASELINE_METRICS}
                stds = {m: self._safe_std(row[(m, "std")], values[m]) for m in BASELINE_METRICS}
                key = idx if isinstance(idx, tuple) else (idx,)
                self.group_fallback[key] = BaselineCell(n_samples=n, values=values, stds=stds)

        global_means = working[BASELINE_METRICS].mean()
        global_stds = working[BASELINE_METRICS].std()
        self.global_fallback = BaselineCell(
            n_samples=len(working),
            values={m: float(global_means[m]) for m in BASELINE_METRICS},
            stds={
                m: self._safe_std(global_stds[m], global_means[m]) for m in BASELINE_METRICS
            },
        )
        return self

    @staticmethod
    def _safe_std(std_value: float, mean_value: float) -> float:
        """NaN (single-sample cell) or ~0 std both fall back to a small
        floor relative to the mean, so a z-test denominator is never
        zero/undefined."""
        floor = _relative_floor(mean_value)
        if std_value is None or pd.isna(std_value):
            return floor
        return max(float(std_value), floor)

    def predict(
        self,
        window_start: datetime,
        group_values: Optional[tuple] = None,
    ) -> dict[str, float]:
        """Return expected metric values for a given timestamp and
        (optionally) a group key tuple matching `group_cols`'s order.

        Falls back: hourly cell (if enough samples) -> group-level average
        (ignoring hour) -> global average -> zeros (only if the model was
        never fit).
        """
        hour = pd.Timestamp(window_start).hour
        hourly_key = (tuple(group_values) if group_values else ()) + (hour,)

        cell = self.hourly_cells.get(hourly_key)
        if cell is not None and cell.n_samples >= MIN_SAMPLES_FOR_HOURLY_BASELINE:
            return dict(cell.values)

        if group_values is not None:
            group_key = tuple(group_values)
            group_cell = self.group_fallback.get(group_key)
            if group_cell is not None and group_cell.n_samples > 0:
                return dict(group_cell.values)

        if self.global_fallback is not None:
            return dict(self.global_fallback.values)

        return {m: 0.0 for m in BASELINE_METRICS}

    def predict_std(
        self,
        window_start: datetime,
        group_values: Optional[tuple] = None,
    ) -> dict[str, float]:
        """Same fallback order as `predict`, but returns the empirical
        standard deviation of each metric instead of its mean. Used as
        the denominator for statistical-significance gating."""
        hour = pd.Timestamp(window_start).hour
        hourly_key = (tuple(group_values) if group_values else ()) + (hour,)

        cell = self.hourly_cells.get(hourly_key)
        if cell is not None and cell.n_samples >= MIN_SAMPLES_FOR_HOURLY_BASELINE:
            return dict(cell.stds)

        if group_values is not None:
            group_key = tuple(group_values)
            group_cell = self.group_fallback.get(group_key)
            if group_cell is not None and group_cell.n_samples > 0:
                return dict(group_cell.stds)

        if self.global_fallback is not None:
            return dict(self.global_fallback.stds)

        return {m: 1.0 for m in BASELINE_METRICS}


def fit_baseline_for_grouping(
    aggregated_df: pd.DataFrame,
    incident_windows: list[dict[str, Any]],
    group_cols: list[str] | None,
) -> BaselineModel:
    """Convenience: label incident-affected rows, filter to normal rows,
    and fit a BaselineModel for one grouping.
    """
    if aggregated_df.empty:
        return BaselineModel(group_cols=group_cols)

    group_values = (
        {col: aggregated_df[col] for col in group_cols} if group_cols else None
    )
    affected = label_incident_affected(
        aggregated_df["window_start"],
        aggregated_df["window_end"],
        incident_windows,
        group_values=group_values,
    )
    normal_df = aggregated_df.loc[~affected]
    return BaselineModel(group_cols=group_cols).fit(normal_df)
