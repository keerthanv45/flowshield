from datetime import datetime

import pandas as pd
import pytest

from ml.health.aggregation import STANDARD_GROUPINGS, aggregate_time_windows


def make_events(rows: list[dict]) -> pd.DataFrame:
    """Minimal event rows for aggregation tests — only the columns the
    aggregator actually reads."""
    defaults = dict(
        currency="INR", customer_id="cust_1", attempt_number=1, region="KA", bank="HDFC",
        payment_method="UPI",
    )
    full_rows = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


class TestAggregationCounts:
    def test_correct_event_counts_single_window(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), status="success", amount=200, latency_ms=1200, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 3), status="failed", amount=150, latency_ms=3000, failure_reason="timeout"),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["transaction_count"] == 3
        assert row["successful_count"] == 2
        assert row["failed_count"] == 1

    def test_two_separate_windows(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 20), status="success", amount=100, latency_ms=1000, failure_reason=None),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        assert len(result) == 2

    def test_empty_dataframe(self):
        events = make_events([])
        result = aggregate_time_windows(events, window_minutes=15)
        assert result.empty


class TestSuccessFailureRates:
    def test_success_and_failure_rate_correct(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 3), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 4), status="failed", amount=100, latency_ms=1000, failure_reason="bank_declined"),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        row = result.iloc[0]
        assert row["success_rate"] == pytest.approx(0.75)
        assert row["failure_rate"] == pytest.approx(0.25)

    def test_failure_reason_rates_sum_to_failure_rate(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), status="failed", amount=100, latency_ms=1000, failure_reason="timeout"),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), status="failed", amount=100, latency_ms=1000, failure_reason="network_error"),
                dict(timestamp=datetime(2026, 8, 1, 0, 3), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 4), status="success", amount=100, latency_ms=1000, failure_reason=None),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        row = result.iloc[0]
        reason_sum = (
            row["timeout_rate"] + row["network_error_rate"] + row["insufficient_funds_rate"]
            + row["authentication_failed_rate"] + row["bank_declined_rate"]
            + row["technical_error_rate"] + row["unknown_rate"]
        )
        assert reason_sum == pytest.approx(row["failure_rate"])


class TestRevenueTotals:
    def test_revenue_totals_correct(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), status="success", amount=200, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 3), status="failed", amount=50, latency_ms=1000, failure_reason="timeout"),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        row = result.iloc[0]
        assert row["total_amount"] == pytest.approx(350)
        assert row["successful_amount"] == pytest.approx(300)
        assert row["failed_amount"] == pytest.approx(50)
        assert row["average_amount"] == pytest.approx(350 / 3)


class TestLatencyStatistics:
    def test_average_and_p95_latency(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, i), status="success", amount=100, latency_ms=1000 + i * 100, failure_reason=None)
                for i in range(1, 11)
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15)
        row = result.iloc[0]
        latencies = [1000 + i * 100 for i in range(1, 11)]
        assert row["average_latency_ms"] == pytest.approx(sum(latencies) / len(latencies))
        assert row["p95_latency_ms"] == pytest.approx(pd.Series(latencies).quantile(0.95))


class TestGroupedMetrics:
    def test_grouping_by_bank_produces_separate_rows(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), bank="HDFC", status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), bank="SBI", status="failed", amount=100, latency_ms=1000, failure_reason="timeout"),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15, group_cols=["bank"])
        assert len(result) == 2
        assert set(result["bank"]) == {"HDFC", "SBI"}

    def test_bank_payment_method_grouping(self):
        events = make_events(
            [
                dict(timestamp=datetime(2026, 8, 1, 0, 1), bank="HDFC", payment_method="UPI", status="success", amount=100, latency_ms=1000, failure_reason=None),
                dict(timestamp=datetime(2026, 8, 1, 0, 2), bank="HDFC", payment_method="CARD", status="success", amount=100, latency_ms=1000, failure_reason=None),
            ]
        )
        result = aggregate_time_windows(events, window_minutes=15, group_cols=["bank", "payment_method"])
        assert len(result) == 2
        assert set(result["payment_method"]) == {"UPI", "CARD"}

    def test_standard_groupings_all_present(self):
        assert set(STANDARD_GROUPINGS.keys()) == {
            "overall", "bank", "payment_method", "region", "bank_payment_method", "region_payment_method",
        }
