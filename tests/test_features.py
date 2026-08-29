from datetime import datetime

import pandas as pd
import pytest

from ml.health.baseline import BaselineModel
from ml.health.features import add_deviation_features, safe_ratio


class TestSafeRatio:
    def test_normal_ratio(self):
        assert safe_ratio(10, 5) == pytest.approx(2.0)

    def test_zero_denominator_returns_default(self):
        assert safe_ratio(10, 0) == 1.0
        assert safe_ratio(10, 0, default=0.0) == 0.0

    def test_near_zero_denominator_returns_default(self):
        assert safe_ratio(10, 1e-9) == 1.0


class TestDeviationFeatures:
    def test_deviation_columns_added(self):
        bm = BaselineModel(group_cols=None)
        bm.global_fallback = None  # force zero fallback path
        df = pd.DataFrame(
            [
                dict(
                    window_start=datetime(2026, 8, 1, 0, 0),
                    window_end=datetime(2026, 8, 1, 0, 15),
                    success_rate=0.9, failure_rate=0.1, average_latency_ms=1200.0,
                    p95_latency_ms=2000.0, transaction_count=15,
                )
            ]
        )
        result = add_deviation_features(df, bm, group_cols=None)
        for col in [
            "baseline_success_rate", "success_rate_delta", "latency_ratio",
            "volume_ratio", "success_rate_z", "failure_rate_z",
        ]:
            assert col in result.columns

    def test_empty_dataframe_returns_empty(self):
        bm = BaselineModel(group_cols=None)
        result = add_deviation_features(pd.DataFrame(), bm, group_cols=None)
        assert result.empty

    def test_deviation_zero_when_matches_baseline_exactly(self):
        rows = [
            dict(
                window_start=datetime(2026, 8, 1, h, 0),
                window_end=datetime(2026, 8, 1, h, 15),
                success_rate=0.9, failure_rate=0.1, average_latency_ms=1200.0,
                p95_latency_ms=2000.0, transaction_count=15,
            )
            for h in range(5)
        ]
        df = pd.DataFrame(rows)
        bm = BaselineModel(group_cols=None).fit(df)
        result = add_deviation_features(df, bm, group_cols=None)
        assert result["success_rate_delta"].abs().max() < 1e-9
