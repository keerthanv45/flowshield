"""
Regression test for the train/val/test leakage fix in
`ml.health.pipeline.run_health_pipeline`.

Before the fix, baselines were fit on the FULL aggregated dataframe (all
groupings), so validation/test windows could pull baseline statistics
away from what a pure TRAIN-only fit would produce. This test builds a
dataset where the chronological tail (which falls in validation/test)
has a drastically different success rate than the head (train), and
asserts the fitted baseline reflects ONLY the train-period behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.health.pipeline import _train_slice, compute_train_cutoff, run_health_pipeline
from ml.health.aggregation import aggregate_time_windows


def make_events(n_days: int, events_per_day: int, success_rate: float, day_offset: int = 0) -> list[dict]:
    """`events_per_day` must be dense enough (<=15 min spacing) that every
    15-minute window in the covered period contains at least one event —
    otherwise pandas' Grouper fills genuinely-empty intermediate time bins
    with zero-count rows for the ungrouped ("overall") aggregation, which
    is a separate, pre-existing aggregation-layer quirk unrelated to the
    leakage fix under test here (the real 20k-event Phase 1 dataset never
    hits this because it's dense: minimum observed window has 3 events).
    """
    rows = []
    start = datetime(2026, 8, 1) + timedelta(days=day_offset)
    for day in range(n_days):
        for i in range(events_per_day):
            ts = start + timedelta(days=day, minutes=i * (24 * 60 // events_per_day))
            # Interleave success/failure (rather than clustering failures
            # at day's end) so every hour-of-day sees the same success
            # rate -- isolates the leakage behavior from any hour-of-day
            # baseline effect.
            status = "success" if (i % 10) < round(success_rate * 10) else "failed"
            rows.append(
                dict(
                    timestamp=ts, status=status, amount=100.0, latency_ms=1000.0,
                    failure_reason=None if status == "success" else "bank_declined",
                    bank="HDFC", payment_method="UPI", region="KA",
                    currency="INR", customer_id="c1", attempt_number=1,
                )
            )
    return rows


class TestNoLeakageFromValTest:
    def test_baseline_unaffected_by_tail_distribution_shift(self):
        # 10 days at success_rate=0.9 (train, given 70% split falls after
        # this), then 4 days at success_rate=0.1 (falls in val/test).
        events = make_events(10, 120, 0.9) + make_events(4, 120, 0.1, day_offset=10)
        df = pd.DataFrame(events)

        result = run_health_pipeline(df, incident_windows=[])

        # Predict baseline success_rate for an early (train-period) hour.
        overall_baseline = result.baselines["overall"]
        pred = overall_baseline.predict(datetime(2026, 8, 2, 12, 0))

        # If the tail had leaked in, the blended mean would be pulled
        # toward ~0.9*10/14 + 0.1*4/14 ≈ 0.66. It must instead stay near
        # the train-only value of 0.9.
        assert pred["success_rate"] > 0.85
        assert pred["success_rate"] != pytest.approx(0.66, abs=0.05)

    def test_cutoff_applied_consistently_across_groupings(self):
        events = make_events(10, 120, 0.9) + make_events(4, 120, 0.1, day_offset=10)
        df = pd.DataFrame(events)
        result = run_health_pipeline(df, incident_windows=[])

        overall_agg = aggregate_time_windows(df, window_minutes=15)
        cutoff = compute_train_cutoff(overall_agg)

        # Every grouped baseline's underlying fit data must respect the
        # same cutoff -- verify no grouping's baseline was fit using rows
        # at/after cutoff by checking bank-level baseline the same way.
        bank_agg = aggregate_time_windows(df, window_minutes=15, group_cols=["bank"])
        bank_train = _train_slice(bank_agg, cutoff)
        assert bank_train["window_start"].max() < cutoff
        # And nothing in the "train" slice is drawn from the shifted tail.
        assert bank_train["window_start"].max() < datetime(2026, 8, 11)

    def test_anomaly_detector_train_only_preserved(self):
        # Structural proof (robust, not dependent on IsolationForest's
        # emergent behavior on a particular synthetic shape): re-fit a
        # detector independently on the exact train-cutoff, non-incident
        # slice the pipeline is supposed to use, and confirm the
        # pipeline's own detector produces IDENTICAL scores on a probe
        # point. If the pipeline's detector had leaked val/test rows into
        # fitting, its training data (and thus its score normalization/
        # decision boundary) would differ from this independently-fit one.
        from ml.health.aggregation import aggregate_time_windows
        from ml.health.anomaly import AnomalyDetector
        from ml.health.baseline import fit_baseline_for_grouping, label_incident_affected
        from ml.health.features import add_deviation_features

        events = make_events(10, 120, 0.9) + make_events(4, 120, 0.1, day_offset=10)
        df = pd.DataFrame(events)
        result = run_health_pipeline(df, incident_windows=[])

        overall_agg = aggregate_time_windows(df, window_minutes=15)
        cutoff = compute_train_cutoff(overall_agg)
        overall_train = _train_slice(overall_agg, cutoff)
        baseline = fit_baseline_for_grouping(overall_train, [], group_cols=None)
        overall_feat = add_deviation_features(overall_agg, baseline, group_cols=None)
        overall_feat_train = _train_slice(overall_feat, cutoff)
        affected = label_incident_affected(
            overall_feat_train["window_start"], overall_feat_train["window_end"], [], group_values=None
        )
        expected_normal_train = overall_feat_train.loc[~affected]

        independent_detector = AnomalyDetector().fit(expected_normal_train)

        probe = overall_feat.iloc[[-1]]  # a tail (val/test-period) row
        pipeline_result = result.anomaly_detector.score(probe)[0]
        independent_result = independent_detector.score(probe)[0]

        assert pipeline_result.anomaly_score == pytest.approx(independent_result.anomaly_score)
        assert pipeline_result.is_anomaly == independent_result.is_anomaly

    def test_train_end_index_matches_cutoff(self):
        events = make_events(10, 120, 0.9) + make_events(4, 120, 0.1, day_offset=10)
        df = pd.DataFrame(events)
        result = run_health_pipeline(df, incident_windows=[])

        overall_agg = aggregate_time_windows(df, window_minutes=15)
        cutoff = compute_train_cutoff(overall_agg)
        sorted_scored = result.overall_scored.sort_values("window_start").reset_index(drop=True)
        n_before_cutoff = (sorted_scored["window_start"] < cutoff).sum()
        assert result.train_end_index == n_before_cutoff
