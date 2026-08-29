import pytest

from ml.health.scoring import (
    HealthStatus,
    compute_health_score,
    failure_pattern_health,
    latency_health,
    success_rate_health,
    volume_health,
)


def make_row(**overrides):
    base = dict(
        success_rate=0.9,
        baseline_success_rate=0.9,
        average_latency_ms=1200.0,
        baseline_average_latency_ms=1200.0,
        failure_rate=0.1,
        baseline_failure_rate=0.1,
        timeout_rate=0.03,
        network_error_rate=0.02,
        technical_error_rate=0.01,
        transaction_count=15,
        baseline_transaction_count=15,
    )
    base.update(overrides)
    return base


class TestHealthyData:
    def test_healthy_window_gets_high_score(self):
        row = make_row()
        result = compute_health_score(row)
        assert result.score >= 80
        assert result.status == HealthStatus.HEALTHY

    def test_above_baseline_still_capped_at_100_component(self):
        row = make_row(success_rate=0.99, baseline_success_rate=0.9)
        assert success_rate_health(row["success_rate"], row["baseline_success_rate"]) == 100.0


class TestSevereDegradation:
    def test_severe_success_rate_drop_gets_low_score(self):
        row = make_row(
            success_rate=0.2, baseline_success_rate=0.9,
            failure_rate=0.8, baseline_failure_rate=0.1,
            timeout_rate=0.5, network_error_rate=0.2, technical_error_rate=0.05,
        )
        result = compute_health_score(row)
        assert result.score < 50
        assert result.status == HealthStatus.CRITICAL

    def test_severe_latency_spike_lowers_score(self):
        healthy = compute_health_score(make_row())
        degraded = compute_health_score(
            make_row(average_latency_ms=6000.0, baseline_average_latency_ms=1200.0)
        )
        assert degraded.score < healthy.score

    def test_zero_success_rate_is_critical(self):
        row = make_row(
            success_rate=0.0, baseline_success_rate=0.9,
            failure_rate=1.0, baseline_failure_rate=0.1,
            timeout_rate=0.6, network_error_rate=0.3, technical_error_rate=0.1,
        )
        result = compute_health_score(row)
        assert result.status == HealthStatus.CRITICAL


class TestScoreBounds:
    @pytest.mark.parametrize(
        "row",
        [
            make_row(),
            make_row(success_rate=0.0, baseline_success_rate=0.9),
            make_row(average_latency_ms=100000, baseline_average_latency_ms=1200),
            make_row(transaction_count=0, baseline_transaction_count=15),
            make_row(transaction_count=1000, baseline_transaction_count=15),
        ],
    )
    def test_score_always_between_0_and_100(self, row):
        result = compute_health_score(row)
        assert 0.0 <= result.score <= 100.0

    def test_component_scores_always_bounded(self):
        assert 0.0 <= success_rate_health(0.0, 0.9) <= 100.0
        assert 0.0 <= latency_health(999999, 1200) <= 100.0
        assert 0.0 <= volume_health(0, 15) <= 100.0
        assert 0.0 <= volume_health(9999, 15) <= 100.0
        assert 0.0 <= failure_pattern_health(0.9, {"timeout_rate": 0.9}, 0.1) <= 100.0


class TestStatusThresholds:
    def test_status_boundaries(self):
        from ml.health.scoring import status_from_score

        assert status_from_score(100) == HealthStatus.HEALTHY
        assert status_from_score(80) == HealthStatus.HEALTHY
        assert status_from_score(79.9) == HealthStatus.DEGRADED
        assert status_from_score(50) == HealthStatus.DEGRADED
        assert status_from_score(49.9) == HealthStatus.CRITICAL
        assert status_from_score(0) == HealthStatus.CRITICAL


class TestDeterminism:
    def test_same_input_same_output(self):
        row = make_row(success_rate=0.5, average_latency_ms=3000)
        r1 = compute_health_score(row)
        r2 = compute_health_score(row)
        assert r1.score == r2.score
        assert r1.status == r2.status
