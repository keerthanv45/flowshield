from datetime import datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas.payment_event import (
    Bank,
    FailureReason,
    PaymentEvent,
    PaymentMethod,
    PaymentStatus,
    Region,
)


def make_valid_kwargs(**overrides):
    base = dict(
        event_id="evt_0000001",
        payment_id="pay_0000001",
        timestamp=datetime(2026, 8, 1, 12, 0, 0),
        amount=499.0,
        currency="INR",
        payment_method=PaymentMethod.UPI,
        bank=Bank.HDFC,
        region=Region.KA,
        status=PaymentStatus.SUCCESS,
        failure_reason=None,
        latency_ms=1200.0,
        customer_id="cust_000001",
        attempt_number=1,
    )
    base.update(overrides)
    return base


class TestValidPaymentEvent:
    def test_valid_success_event(self):
        event = PaymentEvent(**make_valid_kwargs())
        assert event.status == PaymentStatus.SUCCESS
        assert event.failure_reason is None

    def test_valid_failed_event_requires_failure_reason(self):
        event = PaymentEvent(
            **make_valid_kwargs(status=PaymentStatus.FAILED, failure_reason=FailureReason.TIMEOUT)
        )
        assert event.status == PaymentStatus.FAILED
        assert event.failure_reason == FailureReason.TIMEOUT


class TestInvalidPaymentEvent:
    def test_missing_required_field_raises(self):
        kwargs = make_valid_kwargs()
        del kwargs["event_id"]
        with pytest.raises(ValidationError):
            PaymentEvent(**kwargs)

    def test_invalid_enum_value_raises(self):
        kwargs = make_valid_kwargs()
        kwargs["payment_method"] = "NOT_A_METHOD"
        with pytest.raises(ValidationError):
            PaymentEvent(**kwargs)

    def test_invalid_currency_raises(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(currency="USD"))


class TestAmountValidation:
    def test_zero_amount_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(amount=0))

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(amount=-50))

    def test_positive_amount_accepted(self):
        event = PaymentEvent(**make_valid_kwargs(amount=0.01))
        assert event.amount == 0.01


class TestLatencyValidation:
    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(latency_ms=-1))

    def test_zero_latency_accepted(self):
        event = PaymentEvent(**make_valid_kwargs(latency_ms=0))
        assert event.latency_ms == 0

    def test_absurd_latency_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(latency_ms=999_999))


class TestAttemptNumberValidation:
    def test_attempt_zero_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(attempt_number=0))

    def test_attempt_one_accepted(self):
        event = PaymentEvent(**make_valid_kwargs(attempt_number=1))
        assert event.attempt_number == 1

    def test_absurd_attempt_number_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(attempt_number=999))


class TestFailureCategoryConsistency:
    @pytest.mark.parametrize("reason", list(FailureReason))
    def test_all_failure_reasons_valid_when_failed(self, reason):
        event = PaymentEvent(
            **make_valid_kwargs(status=PaymentStatus.FAILED, failure_reason=reason)
        )
        assert event.failure_reason == reason

    def test_failed_status_without_reason_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(**make_valid_kwargs(status=PaymentStatus.FAILED, failure_reason=None))

    def test_success_status_with_reason_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(
                **make_valid_kwargs(
                    status=PaymentStatus.SUCCESS, failure_reason=FailureReason.TIMEOUT
                )
            )

    def test_created_status_with_reason_rejected(self):
        with pytest.raises(ValidationError):
            PaymentEvent(
                **make_valid_kwargs(
                    status=PaymentStatus.CREATED, failure_reason=FailureReason.TIMEOUT
                )
            )
