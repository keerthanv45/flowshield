"""
End-to-end Payment Health Engine pipeline.

Wires together every Phase 2 component in the order described in the
brief:

    Raw Payment Events
        -> Time Window Aggregation
        -> Payment Health Metrics
        -> Expected Baseline
        -> Deviation Detection
        -> Anomaly Score
        -> Incident Detection
        -> Incident Classification
        -> Health Snapshot

This module exists so `scripts/run_health_pipeline.py`,
`scripts/evaluate_phase2.py`, and the test suite all build the pipeline
the same way instead of re-deriving the wiring three times.

============================================================
DATA LEAKAGE / TRAIN-VAL-TEST USAGE (see also ml/evaluation/split.py)
============================================================
  - Aggregated windows are split chronologically FIRST, using a single
    cutoff timestamp derived from the headline (15-minute, ungrouped)
    aggregation's first `TRAIN_FRACTION` of rows. The SAME cutoff
    timestamp is then applied to every other grouping/granularity
    (bank, payment_method, region, bank_payment_method, and the coarser
    120-minute concentration groupings) — so "train" means the same
    wall-clock period everywhere, regardless of how many rows a given
    grouping happens to produce before or after that instant.
  - The BASELINE (`ml.health.baseline.BaselineModel`) for every grouping
    is fit ONLY on rows with `window_start` before that cutoff, and only
    on windows that do not overlap a known incident window.
  - The ANOMALY DETECTOR (`ml.health.anomaly.AnomalyDetector`) is fit on
    that same train-cutoff, non-incident subset of the headline
    aggregation.
  - Both are FROZEN after fitting: scoring/predicting on validation/test
    windows never refits or updates them (`BaselineModel.predict` is
    pure lookup, `AnomalyDetector.score` does not touch training state).
  - Ground-truth ORDER/labels (`incident_windows`) are used only to
    decide what NOT to train on — never as a detection/classification
    input, and never derived from validation/test data specifically to
    influence anything at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.health.aggregation import STANDARD_GROUPINGS, aggregate_time_windows
from ml.health.anomaly import AnomalyDetector
from ml.health.baseline import BaselineModel, fit_baseline_for_grouping, label_incident_affected
from ml.health.features import add_deviation_features
from ml.health.incidents import (
    CONCENTRATION_WINDOW_MINUTES,
    Incident,
    WindowClassification,
    apply_persistence,
    classify_all_windows,
)
from ml.health.scoring import compute_health_score

HEADLINE_WINDOW_MINUTES = 15
CONCENTRATION_GROUPINGS = {
    "bank": ["bank"],
    "payment_method": ["payment_method"],
    "region": ["region"],
    "bank_payment_method": ["bank", "payment_method"],
}
TRAIN_FRACTION = 0.70


@dataclass
class PipelineResult:
    overall_scored: pd.DataFrame
    grouped_features: dict[str, pd.DataFrame]  # STANDARD_GROUPINGS, headline window
    concentration_features: dict[str, pd.DataFrame]  # CONCENTRATION_GROUPINGS, coarse window
    baselines: dict[str, BaselineModel]
    anomaly_detector: AnomalyDetector
    classifications: list[WindowClassification]
    incidents: list[Incident]
    train_end_index: int  # row index in overall_scored splitting train from val/test


def chronological_train_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, int]:
    """Return (train_df, train_end_index) — the first `train_fraction`
    of rows by time order. Mirrors `ml.evaluation.split.time_based_split`
    but returns just the train boundary, which is all the health engine
    needs (validation/test are used only for evaluation, not fitting)."""
    sorted_df = df.sort_values("window_start").reset_index(drop=True)
    train_end = int(len(sorted_df) * train_fraction)
    return sorted_df.iloc[:train_end], train_end


def compute_train_cutoff(overall_agg: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> pd.Timestamp:
    """A single wall-clock cutoff timestamp, derived from the headline
    (15-minute, ungrouped) aggregation's first `train_fraction` of rows.
    Applying this SAME cutoff to every grouping (rather than taking each
    grouping's own first `train_fraction` of ITS rows) is what prevents
    leakage: a sparser grouping (e.g. bank_payment_method) must not be
    allowed to pull in rows that are chronologically inside the
    validation/test period just because it has fewer total rows."""
    train_df, _ = chronological_train_split(overall_agg, train_fraction)
    if train_df.empty:
        return pd.Timestamp(overall_agg["window_start"].min())
    return train_df["window_end"].max()


def _train_slice(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before the train cutoff — the only rows any
    baseline/anomaly model is allowed to fit on."""
    return df.loc[df["window_start"] < cutoff]


def run_health_pipeline(
    events_df: pd.DataFrame,
    incident_windows: list[dict],
    headline_window_minutes: int = HEADLINE_WINDOW_MINUTES,
    concentration_window_minutes: int = CONCENTRATION_WINDOW_MINUTES,
    train_fraction: float = TRAIN_FRACTION,
) -> PipelineResult:
    """Run the full Phase 2 pipeline on raw payment events.

    `incident_windows`: the ground-truth incident schedule (from
    `data/synthetic/incident_windows.json`'s `"windows"` list), used ONLY
    to exclude incident-affected windows from baseline/anomaly TRAINING —
    never used by the classification/detection logic itself, which sees
    only aggregated metrics.
    """
    overall_agg = aggregate_time_windows(events_df, window_minutes=headline_window_minutes)
    cutoff = compute_train_cutoff(overall_agg, train_fraction)

    overall_train = _train_slice(overall_agg, cutoff)
    overall_baseline = fit_baseline_for_grouping(overall_train, incident_windows, group_cols=None)
    overall_feat = add_deviation_features(overall_agg, overall_baseline, group_cols=None)

    overall_feat_train = _train_slice(overall_feat, cutoff)
    affected_train = label_incident_affected(
        overall_feat_train["window_start"], overall_feat_train["window_end"], incident_windows, group_values=None
    )
    normal_train = overall_feat_train.loc[~affected_train]
    train_end = len(overall_feat_train)

    anomaly_detector = AnomalyDetector().fit(normal_train)
    overall_scored = anomaly_detector.score_dataframe(overall_feat)
    overall_scored["health_score"] = overall_scored.apply(
        lambda r: compute_health_score(r.to_dict()).score, axis=1
    )

    baselines = {"overall": overall_baseline}
    grouped_features = {"overall": overall_feat}

    for name, cols in STANDARD_GROUPINGS.items():
        if name == "overall":
            continue
        agg = aggregate_time_windows(events_df, window_minutes=headline_window_minutes, group_cols=cols)
        agg_train = _train_slice(agg, cutoff)
        bm = fit_baseline_for_grouping(agg_train, incident_windows, group_cols=cols)
        grouped_features[name] = add_deviation_features(agg, bm, group_cols=cols)
        baselines[name] = bm

    concentration_features = {}
    for name, cols in CONCENTRATION_GROUPINGS.items():
        agg = aggregate_time_windows(events_df, window_minutes=concentration_window_minutes, group_cols=cols)
        agg_train = _train_slice(agg, cutoff)
        bm = fit_baseline_for_grouping(agg_train, incident_windows, group_cols=cols)
        concentration_features[name] = add_deviation_features(agg, bm, group_cols=cols)

    classifications = classify_all_windows(
        overall_scored, concentration_features, concentration_window_minutes=concentration_window_minutes
    )
    incidents = apply_persistence(classifications)

    return PipelineResult(
        overall_scored=overall_scored,
        grouped_features=grouped_features,
        concentration_features=concentration_features,
        baselines=baselines,
        anomaly_detector=anomaly_detector,
        classifications=classifications,
        incidents=incidents,
        train_end_index=train_end,
    )
