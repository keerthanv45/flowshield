from datetime import datetime

from backend.app.services.validation import parse_row, validate_dataset


def make_row(**overrides):
    base = dict(
        event_id="evt_0000001",
        payment_id="pay_0000001",
        timestamp=datetime(2026, 8, 1, 12, 0, 0),
        amount=499.0,
        currency="INR",
        payment_method="UPI",
        bank="HDFC",
        region="KA",
        status="success",
        failure_reason=None,
        latency_ms=1200.0,
        customer_id="cust_000001",
        attempt_number=1,
    )
    base.update(overrides)
    return base


class TestParseRow:
    def test_valid_row_parses(self):
        event, errors = parse_row(make_row())
        assert event is not None
        assert errors == []

    def test_invalid_row_returns_errors(self):
        event, errors = parse_row(make_row(amount=-1))
        assert event is None
        assert len(errors) > 0


class TestDatasetValidation:
    def test_all_valid_rows(self):
        rows = [make_row(event_id=f"evt_{i:07d}", payment_id=f"pay_{i:07d}") for i in range(5)]
        report = validate_dataset(rows)
        assert report.is_valid
        assert report.total_rows == 5
        assert report.valid_rows == 5
        assert report.invalid_row_count == 0

    def test_duplicate_event_ids_detected(self):
        rows = [
            make_row(event_id="evt_dup", payment_id="pay_0000001"),
            make_row(event_id="evt_dup", payment_id="pay_0000002"),
            make_row(event_id="evt_unique", payment_id="pay_0000003"),
        ]
        report = validate_dataset(rows)
        assert not report.is_valid
        assert "evt_dup" in report.duplicate_event_ids

    def test_missing_required_field_detected(self):
        rows = [make_row()]
        del rows[0]["event_id"]
        report = validate_dataset(rows)
        assert not report.is_valid
        assert report.invalid_row_count == 1

    def test_invalid_status_detected(self):
        rows = [make_row(status="not_a_status")]
        report = validate_dataset(rows)
        assert not report.is_valid

    def test_invalid_failure_reason_detected(self):
        rows = [make_row(status="failed", failure_reason="not_a_reason")]
        report = validate_dataset(rows)
        assert not report.is_valid

    def test_impossible_attempt_number_detected(self):
        rows = [make_row(attempt_number=0)]
        report = validate_dataset(rows)
        assert not report.is_valid

    def test_invalid_status_failure_combo_detected(self):
        # success status should not carry a failure_reason
        rows = [make_row(status="success", failure_reason="timeout")]
        report = validate_dataset(rows)
        assert not report.is_valid

    def test_empty_dataset_is_valid(self):
        report = validate_dataset([])
        assert report.is_valid
        assert report.total_rows == 0
