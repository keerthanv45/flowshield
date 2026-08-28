import pytest

from ml.analysis.report import compute_analysis, events_to_dataframe
from ml.data_generation.generator import GenerationConfig, PaymentEventGenerator


@pytest.fixture(scope="module")
def sample_df():
    gen = PaymentEventGenerator(GenerationConfig(n_events=2000, seed=11))
    events = gen.generate()
    return events_to_dataframe(events)


class TestAnalysis:
    def test_compute_analysis_runs_without_error(self, sample_df):
        report = compute_analysis(sample_df)
        assert report.total_events == 2000

    def test_success_and_failure_rates_sum_to_one(self, sample_df):
        report = compute_analysis(sample_df)
        assert report.success_rate + report.failure_rate == pytest.approx(1.0, abs=1e-6)

    def test_distributions_sum_to_total(self, sample_df):
        report = compute_analysis(sample_df)
        assert sum(report.payment_method_distribution.values()) == report.total_events
        assert sum(report.bank_distribution.values()) == report.total_events
        assert sum(report.regional_distribution.values()) == report.total_events

    def test_revenue_figures_are_non_negative(self, sample_df):
        report = compute_analysis(sample_df)
        assert report.processed_revenue >= 0
        assert report.failed_payment_revenue >= 0

    def test_p95_latency_is_at_least_average(self, sample_df):
        # Not mathematically guaranteed in general, but true for realistic
        # right-skewed latency distributions like the ones we generate.
        report = compute_analysis(sample_df)
        assert report.p95_latency_ms >= report.average_latency_ms

    def test_empty_dataset_raises(self):
        gen = PaymentEventGenerator(GenerationConfig(n_events=1, seed=1))
        events = gen.generate()
        df = events_to_dataframe(events).iloc[0:0]
        with pytest.raises(ValueError):
            compute_analysis(df)
