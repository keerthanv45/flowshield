import pandas as pd
import pytest

from backend.app.services.recovery.batch_evaluation import BATCH_SCOPE_ID, run_batch_evaluation


def make_events(rows: list[dict]) -> pd.DataFrame:
    defaults = dict(bank="HDFC", payment_method="UPI", region="KA", timestamp=pd.Timestamp("2026-08-01"))
    full = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        full.append(row)
    return pd.DataFrame(full)


class TestBatchAggregation:
    def test_total_and_failed_counts(self):
        df = make_events(
            [
                dict(status="success", amount=100.0, failure_reason=None),
                dict(status="failed", amount=200.0, failure_reason="timeout"),
                dict(status="failed", amount=50.0, failure_reason="bank_declined"),
            ]
        )
        result = run_batch_evaluation(df)
        assert result.total_transactions == 3
        assert result.failed_transactions == 2
        assert result.gross_failed_amount == pytest.approx(250.0)
        assert result.revenue_at_risk == pytest.approx(250.0)

    def test_guardrail_counts_sum_to_failed_transactions(self):
        df = make_events(
            [
                dict(status="failed", amount=100.0, failure_reason="timeout"),
                dict(status="failed", amount=100.0, failure_reason="network_error"),
                dict(status="failed", amount=100.0, failure_reason="technical_error"),
                dict(status="failed", amount=100.0, failure_reason="authentication_failed"),
                dict(status="failed", amount=100.0, failure_reason="insufficient_funds"),
                dict(status="failed", amount=100.0, failure_reason="bank_declined"),
                dict(status="failed", amount=100.0, failure_reason="unknown"),
            ]
        )
        result = run_batch_evaluation(df)
        g = result.guardrails
        total_bucketed = (
            g.retry_routing_actions_selected_count
            + g.auth_failures_excluded_count
            + g.insufficient_funds_deferred_count
            + g.hard_declines_excluded_count
            + g.unsupported_failures_excluded_count
        )
        assert total_bucketed == result.failed_transactions == 7
        assert g.retry_routing_actions_selected_count == 3  # timeout, network_error, technical_error


class TestRecoverableVsNonRecoverable:
    def test_recoverable_pool_excludes_low_rate_reasons(self):
        df = make_events(
            [
                dict(status="failed", amount=1000.0, failure_reason="timeout"),
                dict(status="failed", amount=1000.0, failure_reason="bank_declined"),
            ]
        )
        result = run_batch_evaluation(df)
        assert result.recoverable_transactions == 1
        assert result.recoverable_amount == pytest.approx(1000.0)
        assert result.gross_failed_amount == pytest.approx(2000.0)

    def test_hard_declines_never_selected_for_action(self):
        df = make_events(
            [dict(status="failed", amount=500.0, failure_reason="bank_declined")]
        )
        result = run_batch_evaluation(df)
        assert result.actions_selected == 0
        assert result.simulated_attempts == 0
        assert result.guardrails.hard_declines_excluded_count == 1

    def test_auth_failures_never_selected_for_automated_action(self):
        df = make_events(
            [dict(status="failed", amount=500.0, failure_reason="authentication_failed")]
        )
        result = run_batch_evaluation(df)
        assert result.actions_selected == 0
        assert result.guardrails.auth_failures_excluded_count == 1

    def test_unsupported_never_selected(self):
        df = make_events([dict(status="failed", amount=500.0, failure_reason="unknown")])
        result = run_batch_evaluation(df)
        assert result.actions_selected == 0
        assert result.guardrails.unsupported_failures_excluded_count == 1


class TestDeterminism:
    def test_same_seed_same_results(self):
        df = make_events(
            [dict(status="failed", amount=100.0, failure_reason="timeout") for _ in range(50)]
        )
        r1 = run_batch_evaluation(df, seed=42)
        r2 = run_batch_evaluation(df, seed=42)
        assert r1.simulated_recovered_transactions == r2.simulated_recovered_transactions
        assert r1.simulated_recovered_amount == r2.simulated_recovered_amount

    def test_different_seed_can_differ(self):
        df = make_events(
            [dict(status="failed", amount=100.0, failure_reason="timeout") for _ in range(50)]
        )
        r1 = run_batch_evaluation(df, seed=1)
        r2 = run_batch_evaluation(df, seed=2)
        # Not asserting inequality strictly (could coincide), just that both run cleanly and are valid.
        assert 0 <= r1.simulated_recovered_transactions <= r1.simulated_attempts
        assert 0 <= r2.simulated_recovered_transactions <= r2.simulated_attempts


class TestAggregateRecoveryCalculations:
    def test_recovery_rate_bounds(self):
        df = make_events(
            [dict(status="failed", amount=100.0, failure_reason="timeout") for _ in range(20)]
        )
        result = run_batch_evaluation(df)
        assert 0.0 <= result.recovery_rate <= 1.0
        assert 0.0 <= result.revenue_recovery_rate <= 1.0

    def test_revenue_recovery_rate_relative_to_total_at_risk(self):
        df = make_events(
            [
                dict(status="failed", amount=1000.0, failure_reason="timeout"),
                dict(status="failed", amount=1000.0, failure_reason="bank_declined"),
            ]
        )
        result = run_batch_evaluation(df)
        # revenue_recovery_rate is recovered / TOTAL at-risk (2000), not just the retried subset
        assert result.revenue_recovery_rate <= result.simulated_recovered_amount / 1000.0 + 1e-9


class TestZeroFailureBatch:
    def test_all_success_batch(self):
        df = make_events([dict(status="success", amount=100.0, failure_reason=None) for _ in range(10)])
        result = run_batch_evaluation(df)
        assert result.failed_transactions == 0
        assert result.revenue_at_risk == 0.0
        assert result.simulated_attempts == 0
        assert result.recovery_rate == 0.0
        assert result.revenue_recovery_rate == 0.0

    def test_empty_dataframe(self):
        df = make_events([])
        result = run_batch_evaluation(df)
        assert result.total_transactions == 0
        assert result.failed_transactions == 0


class TestNoRealPaymentExecution:
    def test_scope_id_is_synthetic_not_real_incident(self):
        df = make_events([dict(status="failed", amount=100.0, failure_reason="timeout")])
        result = run_batch_evaluation(df)
        assert result.status.startswith("SIMULATED")

    def test_batch_scope_id_constant(self):
        assert BATCH_SCOPE_ID == "BATCH_ALL_TRANSACTIONS"
