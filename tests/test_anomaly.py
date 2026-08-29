import numpy as np
import pandas as pd
import pytest

from ml.health.anomaly import ANOMALY_FEATURES, AnomalyDetector


def make_normal_df(n=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "success_rate": rng.normal(0.9, 0.03, n).clip(0, 1),
            "failure_rate": rng.normal(0.1, 0.03, n).clip(0, 1),
            "p95_latency_ms": rng.normal(2000, 200, n).clip(0),
            "transaction_count": rng.integers(10, 20, n),
            "timeout_rate": rng.normal(0.02, 0.01, n).clip(0),
            "network_error_rate": rng.normal(0.02, 0.01, n).clip(0),
            "technical_error_rate": rng.normal(0.01, 0.005, n).clip(0),
            "success_rate_delta": rng.normal(0, 0.02, n),
            "latency_ratio": rng.normal(1.0, 0.05, n).clip(0.1),
            "volume_ratio": rng.normal(1.0, 0.1, n).clip(0.1),
        }
    )


def make_anomalous_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "success_rate": 0.15,
                "failure_rate": 0.85,
                "p95_latency_ms": 15000,
                "transaction_count": 12,
                "timeout_rate": 0.5,
                "network_error_rate": 0.3,
                "technical_error_rate": 0.1,
                "success_rate_delta": -0.75,
                "latency_ratio": 6.0,
                "volume_ratio": 0.9,
            }
        ]
    )


class TestModelTrains:
    def test_fit_succeeds_on_normal_data(self):
        det = AnomalyDetector().fit(make_normal_df())
        assert det._is_fit

    def test_fit_raises_on_too_little_data(self):
        with pytest.raises(ValueError):
            AnomalyDetector().fit(make_normal_df(n=3))

    def test_fit_raises_on_empty_data(self):
        with pytest.raises(ValueError):
            AnomalyDetector().fit(pd.DataFrame(columns=ANOMALY_FEATURES))


class TestInference:
    def test_score_before_fit_raises(self):
        det = AnomalyDetector()
        with pytest.raises(RuntimeError):
            det.score(make_normal_df(n=5))

    def test_scores_normal_data_as_mostly_not_anomalous(self):
        train = make_normal_df(n=200, seed=1)
        det = AnomalyDetector(contamination=0.05).fit(train)
        results = det.score(make_normal_df(n=200, seed=2))
        anomaly_rate = sum(r.is_anomaly for r in results) / len(results)
        # Held-out normal data should have a low anomaly rate, roughly in
        # line with the contamination parameter (with some slack for
        # sampling variation across a fresh draw).
        assert anomaly_rate < 0.20

    def test_extreme_row_flagged_more_anomalous_than_normal(self):
        train = make_normal_df(n=200, seed=1)
        det = AnomalyDetector(contamination=0.05).fit(train)
        normal_result = det.score(make_normal_df(n=1, seed=99))[0]
        anomalous_result = det.score(make_anomalous_row())[0]
        assert anomalous_result.anomaly_score > normal_result.anomaly_score
        assert anomalous_result.is_anomaly is True

    def test_low_reliability_flag_for_low_volume(self):
        train = make_normal_df(n=200, seed=1)
        det = AnomalyDetector(contamination=0.05, min_reliable_volume=5).fit(train)
        low_vol = make_normal_df(n=1, seed=3)
        low_vol["transaction_count"] = 2
        result = det.score(low_vol)[0]
        assert result.low_reliability is True

    def test_score_dataframe_adds_expected_columns(self):
        train = make_normal_df(n=200, seed=1)
        det = AnomalyDetector().fit(train)
        scored = det.score_dataframe(make_normal_df(n=10, seed=5))
        assert "anomaly_score" in scored.columns
        assert "is_anomaly" in scored.columns
        assert "low_reliability" in scored.columns
        assert len(scored) == 10

    def test_empty_score_returns_empty_list(self):
        train = make_normal_df(n=200, seed=1)
        det = AnomalyDetector().fit(train)
        assert det.score(pd.DataFrame(columns=ANOMALY_FEATURES)) == []


class TestDeterminism:
    def test_same_seed_same_results(self):
        train = make_normal_df(n=200, seed=1)
        test = make_normal_df(n=20, seed=2)

        det_a = AnomalyDetector(random_state=42).fit(train)
        det_b = AnomalyDetector(random_state=42).fit(train)

        results_a = det_a.score(test)
        results_b = det_b.score(test)

        for ra, rb in zip(results_a, results_b):
            assert ra.anomaly_score == pytest.approx(rb.anomaly_score)
            assert ra.is_anomaly == rb.is_anomaly
