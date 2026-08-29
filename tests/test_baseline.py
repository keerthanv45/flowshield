from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.health.baseline import BaselineModel, fit_baseline_for_grouping, label_incident_affected


def make_agg_df(rows: list[dict]) -> pd.DataFrame:
    defaults = dict(
        success_rate=0.9, failure_rate=0.1, average_latency_ms=1200.0,
        p95_latency_ms=2000.0, transaction_count=15,
    )
    full = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        if "window_end" not in row:
            row["window_end"] = row["window_start"] + timedelta(minutes=15)
        full.append(row)
    return pd.DataFrame(full)


class TestNormalDataStableBaseline:
    def test_fit_produces_reasonable_global_average(self):
        rows = [
            dict(window_start=datetime(2026, 8, 1, h, 0), success_rate=0.9)
            for h in range(24)
        ]
        df = make_agg_df(rows)
        bm = BaselineModel(group_cols=None).fit(df)
        pred = bm.predict(datetime(2026, 8, 1, 5, 0))
        assert pred["success_rate"] == pytest.approx(0.9, abs=1e-6)

    def test_hour_of_day_effect_captured(self):
        rows = []
        for day in range(5):
            rows.append(dict(window_start=datetime(2026, 8, 1 + day, 3, 0), success_rate=0.95))
            rows.append(dict(window_start=datetime(2026, 8, 1 + day, 15, 0), success_rate=0.70))
        df = make_agg_df(rows)
        bm = BaselineModel(group_cols=None).fit(df)
        pred_night = bm.predict(datetime(2026, 8, 6, 3, 0))
        pred_day = bm.predict(datetime(2026, 8, 6, 15, 0))
        assert pred_night["success_rate"] > pred_day["success_rate"]


class TestIncidentExclusion:
    def test_incident_windows_excluded_from_fit(self):
        # 10 normal windows at 0.9, 5 "incident" windows at 0.2 -- if the
        # incident rows leak into the baseline, the fitted mean would be
        # pulled well below 0.9.
        normal_rows = [
            dict(window_start=datetime(2026, 8, 1, 0, 0) + timedelta(minutes=15 * i), success_rate=0.9)
            for i in range(10)
        ]
        incident_rows = [
            dict(window_start=datetime(2026, 8, 2, 0, 0) + timedelta(minutes=15 * i), success_rate=0.2)
            for i in range(5)
        ]
        df = make_agg_df(normal_rows + incident_rows)

        incident_windows = [
            {
                "scenario_type": "latency_spike",
                "start": datetime(2026, 8, 2, 0, 0).isoformat(),
                "end": datetime(2026, 8, 2, 2, 0).isoformat(),
            }
        ]
        bm = fit_baseline_for_grouping(df, incident_windows, group_cols=None)
        pred = bm.predict(datetime(2026, 8, 1, 0, 0))
        assert pred["success_rate"] == pytest.approx(0.9, abs=1e-6)

    def test_label_incident_affected_global_incident(self):
        starts = pd.Series([datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 2, 0, 0)])
        ends = starts + timedelta(minutes=15)
        windows = [
            {"scenario_type": "latency_spike", "start": datetime(2026, 8, 1).isoformat(), "end": datetime(2026, 8, 1, 1).isoformat()}
        ]
        affected = label_incident_affected(starts, ends, windows, group_values=None)
        assert affected.tolist() == [True, False]

    def test_label_incident_affected_dimension_targeted(self):
        starts = pd.Series([datetime(2026, 8, 1, 0, 0), datetime(2026, 8, 1, 0, 0)])
        ends = starts + timedelta(minutes=15)
        group_values = {
            "bank": pd.Series(["HDFC", "SBI"]),
            "payment_method": pd.Series(["UPI", "UPI"]),
        }
        windows = [
            {
                "scenario_type": "bank_rail_degradation",
                "start": datetime(2026, 8, 1).isoformat(),
                "end": datetime(2026, 8, 1, 1).isoformat(),
                "target_bank": "HDFC",
                "target_payment_method": "UPI",
            }
        ]
        affected = label_incident_affected(starts, ends, windows, group_values=group_values)
        # HDFC+UPI matches the target -> affected; SBI+UPI does not.
        assert affected.tolist() == [True, False]


class TestZeroDenominatorHandling:
    def test_predict_on_empty_model_returns_zeros(self):
        bm = BaselineModel(group_cols=None)
        pred = bm.predict(datetime(2026, 8, 1))
        assert pred["success_rate"] == 0.0

    def test_predict_std_on_empty_model_returns_ones(self):
        bm = BaselineModel(group_cols=None)
        pred_std = bm.predict_std(datetime(2026, 8, 1))
        assert pred_std["success_rate"] == 1.0

    def test_fit_on_empty_dataframe_does_not_raise(self):
        bm = BaselineModel(group_cols=None).fit(pd.DataFrame())
        pred = bm.predict(datetime(2026, 8, 1))
        assert pred["success_rate"] == 0.0

    def test_std_never_zero_avoids_divide_by_zero(self):
        # All identical success_rate -> std is 0 by definition; the model
        # must apply a floor so downstream z-score division is safe.
        rows = [
            dict(window_start=datetime(2026, 8, 1, 0, 0) + timedelta(minutes=15 * i), success_rate=0.9)
            for i in range(5)
        ]
        df = make_agg_df(rows)
        bm = BaselineModel(group_cols=None).fit(df)
        std = bm.predict_std(datetime(2026, 8, 1, 0, 0))
        assert std["success_rate"] > 0.0
