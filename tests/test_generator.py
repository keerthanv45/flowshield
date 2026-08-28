from backend.app.schemas.payment_event import Bank, PaymentMethod, Region
from ml.data_generation.generator import GenerationConfig, PaymentEventGenerator
from ml.data_generation.scenarios import ScenarioType


def small_config(**overrides) -> GenerationConfig:
    base = dict(n_events=500, seed=42)
    base.update(overrides)
    return GenerationConfig(**base)


class TestReproducibility:
    def test_same_seed_produces_identical_dataset(self):
        gen_a = PaymentEventGenerator(small_config(seed=7))
        gen_b = PaymentEventGenerator(small_config(seed=7))

        events_a = gen_a.generate()
        events_b = gen_b.generate()

        assert len(events_a) == len(events_b)
        for a, b in zip(events_a, events_b):
            assert a.model_dump() == b.model_dump()

    def test_different_seeds_produce_different_datasets(self):
        gen_a = PaymentEventGenerator(small_config(seed=1))
        gen_b = PaymentEventGenerator(small_config(seed=2))

        events_a = gen_a.generate()
        events_b = gen_b.generate()

        # Extremely unlikely to be identical across 500 rows with different seeds.
        statuses_a = [e.status for e in events_a]
        statuses_b = [e.status for e in events_b]
        assert statuses_a != statuses_b

    def test_generates_requested_number_of_events(self):
        gen = PaymentEventGenerator(small_config(n_events=123))
        events = gen.generate()
        assert len(events) == 123

    def test_all_generated_events_are_valid(self):
        # Since events are constructed through the Pydantic model itself,
        # simply generating them without an exception is the validity check.
        gen = PaymentEventGenerator(small_config(n_events=300))
        events = gen.generate()
        assert len(events) == 300
        for e in events:
            assert e.amount > 0
            assert e.latency_ms >= 0
            assert e.attempt_number >= 1

    def test_event_ids_are_unique(self):
        gen = PaymentEventGenerator(small_config(n_events=1000))
        events = gen.generate()
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_events_sorted_by_timestamp(self):
        gen = PaymentEventGenerator(small_config(n_events=500))
        events = gen.generate()
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)


class TestIncidentInjection:
    def test_bank_rail_degradation_lowers_success_rate(self):
        gen = PaymentEventGenerator(small_config(n_events=20_000, seed=99))
        events = gen.generate()

        window = next(
            w for w in gen.incident_windows
            if w.scenario_type == ScenarioType.BANK_RAIL_DEGRADATION
        )

        affected = [
            e for e in events
            if e.payment_method == window.target_payment_method
            and e.bank == window.target_bank
            and window.start <= e.timestamp < window.end
        ]
        unaffected_same_method = [
            e for e in events
            if e.payment_method == window.target_payment_method
            and e.bank != window.target_bank
        ]

        assert len(affected) > 0
        affected_success_rate = sum(1 for e in affected if e.status == "success") / len(affected)
        baseline_success_rate = sum(
            1 for e in unaffected_same_method if e.status == "success"
        ) / len(unaffected_same_method)

        assert affected_success_rate < baseline_success_rate

    def test_regional_degradation_affects_multiple_methods_and_banks(self):
        gen = PaymentEventGenerator(small_config(n_events=20_000, seed=99))
        events = gen.generate()

        window = next(
            w for w in gen.incident_windows
            if w.scenario_type == ScenarioType.REGIONAL_DEGRADATION
        )

        affected = [
            e for e in events
            if e.region == window.target_region and window.start <= e.timestamp < window.end
        ]
        assert len(affected) > 0

        distinct_methods = {e.payment_method for e in affected}
        distinct_banks = {e.bank for e in affected}
        assert len(distinct_methods) > 1
        assert len(distinct_banks) > 1

    def test_latency_spike_increases_latency(self):
        gen = PaymentEventGenerator(small_config(n_events=20_000, seed=99))
        events = gen.generate()

        window = next(
            w for w in gen.incident_windows if w.scenario_type == ScenarioType.LATENCY_SPIKE
        )

        affected = [e for e in events if window.start <= e.timestamp < window.end]
        unaffected = [e for e in events if not (window.start <= e.timestamp < window.end)]

        assert len(affected) > 0
        avg_affected = sum(e.latency_ms for e in affected) / len(affected)
        avg_unaffected = sum(e.latency_ms for e in unaffected) / len(unaffected)
        assert avg_affected > avg_unaffected

    def test_isolated_failures_do_not_concentrate_in_one_bank(self):
        gen = PaymentEventGenerator(small_config(n_events=20_000, seed=99))
        events = gen.generate()

        window = next(
            w for w in gen.incident_windows if w.scenario_type == ScenarioType.ISOLATED_FAILURES
        )
        affected_failures = [
            e for e in events
            if window.start <= e.timestamp < window.end and e.status == "failed"
        ]
        assert len(affected_failures) > 0
        bank_counts = {}
        for e in affected_failures:
            bank_counts[e.bank] = bank_counts.get(e.bank, 0) + 1

        # No single bank should dominate isolated (non-systemic) failures.
        max_share = max(bank_counts.values()) / len(affected_failures)
        assert max_share < 0.6

    def test_insufficient_funds_does_not_inflate_latency(self):
        gen = PaymentEventGenerator(small_config(n_events=20_000, seed=99))
        events = gen.generate()

        insufficient_funds_events = [
            e for e in events if e.failure_reason == "insufficient_funds"
        ]
        timeout_events = [e for e in events if e.failure_reason == "timeout"]

        assert len(insufficient_funds_events) > 0
        assert len(timeout_events) > 0

        avg_if_latency = sum(e.latency_ms for e in insufficient_funds_events) / len(
            insufficient_funds_events
        )
        avg_timeout_latency = sum(e.latency_ms for e in timeout_events) / len(timeout_events)

        assert avg_if_latency < avg_timeout_latency
