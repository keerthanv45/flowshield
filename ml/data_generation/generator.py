"""
Reproducible synthetic payment-event generator.

ALL DATA PRODUCED BY THIS MODULE IS SYNTHETIC. It does not come from, and
is not derived from, any real payment gateway or real transactions.

Design goals:
  - Fully reproducible given a fixed seed (no wall-clock or OS entropy).
  - Rows are NOT independent draws — payment method, bank, region, attempt
    number, and active incident windows all influence success probability,
    failure-reason distribution, and latency, in line with the correlations
    described in the project brief.

The generator is intentionally a single, readable module rather than a
"framework" — Phase 1 does not need pluggable strategies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from backend.app.schemas.payment_event import (
    Bank,
    FailureReason,
    PaymentEvent,
    PaymentMethod,
    Region,
)
from ml.data_generation.scenarios import IncidentWindow, ScenarioType

# ---------------------------------------------------------------------------
# Baseline behavior model
# ---------------------------------------------------------------------------

# Baseline success probability by payment method (before bank/incident effects).
METHOD_BASE_SUCCESS_RATE: dict[PaymentMethod, float] = {
    PaymentMethod.UPI: 0.94,
    PaymentMethod.CARD: 0.90,
    PaymentMethod.NETBANKING: 0.85,
    PaymentMethod.WALLET: 0.92,
}

# Multiplicative adjustment per bank (1.0 = neutral). Reflects that some
# banks are simply flakier rails than others, independent of any incident.
BANK_SUCCESS_MULTIPLIER: dict[Bank, float] = {
    Bank.HDFC: 1.00,
    Bank.ICICI: 0.99,
    Bank.SBI: 0.95,
    Bank.AXIS: 0.98,
    Bank.KOTAK: 0.97,
}

# Baseline latency (ms): (mean, std) by payment method.
METHOD_BASE_LATENCY: dict[PaymentMethod, tuple[float, float]] = {
    PaymentMethod.UPI: (1200.0, 350.0),
    PaymentMethod.CARD: (2500.0, 700.0),
    PaymentMethod.NETBANKING: (4000.0, 1400.0),
    PaymentMethod.WALLET: (900.0, 250.0),
}

# Failure-reason weights when a payment fails, by method. These are relative
# weights (not probabilities) used to pick a failure_reason once we've
# already decided the event failed.
METHOD_FAILURE_REASON_WEIGHTS: dict[PaymentMethod, dict[FailureReason, float]] = {
    PaymentMethod.UPI: {
        FailureReason.TIMEOUT: 0.30,
        FailureReason.NETWORK_ERROR: 0.25,
        FailureReason.BANK_DECLINED: 0.15,
        FailureReason.TECHNICAL_ERROR: 0.10,
        FailureReason.INSUFFICIENT_FUNDS: 0.10,
        FailureReason.AUTHENTICATION_FAILED: 0.05,
        FailureReason.UNKNOWN: 0.05,
    },
    PaymentMethod.CARD: {
        FailureReason.AUTHENTICATION_FAILED: 0.30,
        FailureReason.INSUFFICIENT_FUNDS: 0.25,
        FailureReason.BANK_DECLINED: 0.20,
        FailureReason.TIMEOUT: 0.10,
        FailureReason.NETWORK_ERROR: 0.05,
        FailureReason.TECHNICAL_ERROR: 0.05,
        FailureReason.UNKNOWN: 0.05,
    },
    PaymentMethod.NETBANKING: {
        FailureReason.BANK_DECLINED: 0.30,
        FailureReason.TIMEOUT: 0.25,
        FailureReason.TECHNICAL_ERROR: 0.15,
        FailureReason.NETWORK_ERROR: 0.15,
        FailureReason.AUTHENTICATION_FAILED: 0.10,
        FailureReason.INSUFFICIENT_FUNDS: 0.03,
        FailureReason.UNKNOWN: 0.02,
    },
    PaymentMethod.WALLET: {
        FailureReason.INSUFFICIENT_FUNDS: 0.35,
        FailureReason.TECHNICAL_ERROR: 0.20,
        FailureReason.TIMEOUT: 0.15,
        FailureReason.NETWORK_ERROR: 0.15,
        FailureReason.BANK_DECLINED: 0.10,
        FailureReason.AUTHENTICATION_FAILED: 0.03,
        FailureReason.UNKNOWN: 0.02,
    },
}

# Failure reasons in this set push latency UP when they occur (the payment
# "hung" before failing). INSUFFICIENT_FUNDS deliberately does NOT get a
# latency boost — a declined-for-funds response is fast.
LATENCY_BOOST_FAILURE_REASONS: dict[FailureReason, float] = {
    FailureReason.TIMEOUT: 2.4,
    FailureReason.NETWORK_ERROR: 1.8,
    FailureReason.TECHNICAL_ERROR: 1.3,
}

REGIONS = list(Region)
BANKS = list(Bank)
METHODS = list(PaymentMethod)

N_CUSTOMERS_POOL = 6000  # distinct synthetic customers to draw from

# Retry attempts: distribution over attempt_number for a payment.
ATTEMPT_NUMBER_WEIGHTS = {1: 0.80, 2: 0.14, 3: 0.04, 4: 0.02}


@dataclass
class GenerationConfig:
    n_events: int = 20_000
    seed: int = 42
    period_start: datetime = datetime(2026, 8, 1, 0, 0, 0)
    period_days: int = 14
    min_amount: float = 10.0
    max_amount: float = 200_000.0


def default_incident_schedule(config: GenerationConfig) -> list[IncidentWindow]:
    """Fixed, deterministic schedule of controlled incidents across the
    generation period. Every scenario type appears at least once, each
    with a distinguishable signature, separated by normal-traffic gaps.
    """
    start = config.period_start

    def at(day_offset: float) -> datetime:
        return start + timedelta(days=day_offset)

    windows = [
        IncidentWindow(
            scenario_type=ScenarioType.BANK_RAIL_DEGRADATION,
            start=at(2.0),
            end=at(3.0),
            target_payment_method=PaymentMethod.UPI,
            target_bank=Bank.HDFC,
            success_rate_multiplier=0.35,
            latency_multiplier=2.2,
            failure_reason_weight_overrides={
                FailureReason.TIMEOUT: 3.0,
                FailureReason.NETWORK_ERROR: 2.5,
            },
            description="UPI + HDFC rail degradation",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.REGIONAL_DEGRADATION,
            start=at(4.0),
            end=at(5.5),
            target_region=Region.KA,
            success_rate_multiplier=0.55,
            latency_multiplier=1.6,
            failure_reason_weight_overrides={
                FailureReason.NETWORK_ERROR: 2.0,
                FailureReason.BANK_DECLINED: 1.5,
            },
            description="KA regional degradation (multiple methods/banks)",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.LATENCY_SPIKE,
            start=at(6.0),
            end=at(7.0),
            success_rate_multiplier=0.70,
            latency_multiplier=3.0,
            failure_reason_weight_overrides={
                FailureReason.TIMEOUT: 3.5,
                FailureReason.NETWORK_ERROR: 2.0,
            },
            description="Global latency spike",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.MERCHANT_SYSTEM_DEGRADATION,
            start=at(7.5),
            end=at(9.0),
            success_rate_multiplier=0.60,
            latency_multiplier=1.4,
            failure_reason_weight_overrides={
                FailureReason.TECHNICAL_ERROR: 2.5,
                FailureReason.BANK_DECLINED: 1.3,
            },
            description="Merchant/system-wide degradation (spread across banks & methods)",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.ISOLATED_FAILURES,
            start=at(10.0),
            end=at(11.5),
            isolated_failure_rate=0.03,
            description="Scattered, non-systemic failures (false-positive test)",
        ),
        # --- Held-out-period representation ---------------------------
        # The 70/15/15 chronological train/val/test split (see
        # ml/evaluation/split.py, ml/health/pipeline.py) puts the cutoff
        # between train and val/test at ~day 9.8. Without any incident
        # after that point, held-out evaluation of systemic-scenario
        # detection is vacuous (0/0 ground-truth windows). These four
        # short, additive occurrences — one per systemic scenario type,
        # smaller than their train-period counterparts — give the
        # held-out period genuine systemic incidents to detect, without
        # altering any window already scheduled above.
        IncidentWindow(
            scenario_type=ScenarioType.BANK_RAIL_DEGRADATION,
            start=at(11.5),
            end=at(12.0),
            target_payment_method=PaymentMethod.UPI,
            target_bank=Bank.HDFC,
            success_rate_multiplier=0.35,
            latency_multiplier=2.2,
            failure_reason_weight_overrides={
                FailureReason.TIMEOUT: 3.0,
                FailureReason.NETWORK_ERROR: 2.5,
            },
            description="UPI + HDFC rail degradation (held-out occurrence)",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.REGIONAL_DEGRADATION,
            start=at(12.0),
            end=at(12.75),
            target_region=Region.KA,
            success_rate_multiplier=0.55,
            latency_multiplier=1.6,
            failure_reason_weight_overrides={
                FailureReason.NETWORK_ERROR: 2.0,
                FailureReason.BANK_DECLINED: 1.5,
            },
            description="KA regional degradation (held-out occurrence)",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.LATENCY_SPIKE,
            start=at(12.75),
            end=at(13.25),
            success_rate_multiplier=0.70,
            latency_multiplier=3.0,
            failure_reason_weight_overrides={
                FailureReason.TIMEOUT: 3.5,
                FailureReason.NETWORK_ERROR: 2.0,
            },
            description="Global latency spike (held-out occurrence)",
        ),
        IncidentWindow(
            scenario_type=ScenarioType.MERCHANT_SYSTEM_DEGRADATION,
            start=at(13.25),
            end=at(14.0),
            success_rate_multiplier=0.60,
            latency_multiplier=1.4,
            failure_reason_weight_overrides={
                FailureReason.TECHNICAL_ERROR: 2.5,
                FailureReason.BANK_DECLINED: 1.3,
            },
            description="Merchant/system-wide degradation (held-out occurrence)",
        ),
    ]
    return windows


class PaymentEventGenerator:
    """Generates a reproducible synthetic dataset of PaymentEvents."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._rng = random.Random(self.config.seed)
        self.incident_windows: list[IncidentWindow] = default_incident_schedule(self.config)
        self._customer_ids = [
            f"cust_{i:06d}" for i in range(N_CUSTOMERS_POOL)
        ]

    # -- public API ---------------------------------------------------

    def generate(self) -> list[PaymentEvent]:
        events: list[PaymentEvent] = []
        period_seconds = self.config.period_days * 24 * 3600

        for i in range(self.config.n_events):
            timestamp = self.config.period_start + timedelta(
                seconds=self._rng.uniform(0, period_seconds)
            )
            payment_method = self._rng.choice(METHODS)
            bank = self._rng.choice(BANKS)
            region = self._rng.choice(REGIONS)
            attempt_number = self._sample_attempt_number()
            customer_id = self._rng.choice(self._customer_ids)
            amount = self._sample_amount()

            active_window = self._find_active_window(
                payment_method=payment_method,
                bank=bank,
                region=region,
                timestamp=timestamp,
            )

            status, failure_reason = self._sample_outcome(
                payment_method=payment_method,
                bank=bank,
                attempt_number=attempt_number,
                active_window=active_window,
            )

            latency_ms = self._sample_latency(
                payment_method=payment_method,
                failure_reason=failure_reason,
                attempt_number=attempt_number,
                active_window=active_window,
            )

            event = PaymentEvent(
                event_id=f"evt_{i:07d}",
                payment_id=f"pay_{i:07d}",
                timestamp=timestamp,
                amount=amount,
                currency="INR",
                payment_method=payment_method,
                bank=bank,
                region=region,
                status=status,
                failure_reason=failure_reason,
                latency_ms=latency_ms,
                customer_id=customer_id,
                attempt_number=attempt_number,
            )
            events.append(event)

        events.sort(key=lambda e: e.timestamp)
        return events

    # -- sampling helpers -----------------------------------------------

    def _sample_attempt_number(self) -> int:
        values = list(ATTEMPT_NUMBER_WEIGHTS.keys())
        weights = list(ATTEMPT_NUMBER_WEIGHTS.values())
        return self._rng.choices(values, weights=weights, k=1)[0]

    def _sample_amount(self) -> float:
        # Right-skewed distribution: lognormal, clipped to a sane range.
        raw = self._rng.lognormvariate(mu=6.0, sigma=1.1)  # median ~ e^6 ~= 403
        clipped = min(max(raw, self.config.min_amount), self.config.max_amount)
        return round(clipped, 2)

    def _find_active_window(
        self,
        *,
        payment_method: PaymentMethod,
        bank: Bank,
        region: Region,
        timestamp: datetime,
    ) -> Optional[IncidentWindow]:
        for window in self.incident_windows:
            if window.scenario_type == ScenarioType.ISOLATED_FAILURES:
                # Isolated failures apply dataset-wide at low rate, handled
                # separately in _sample_outcome rather than as a "match".
                if window.start <= timestamp < window.end:
                    return window
                continue
            if window.matches(
                payment_method=payment_method, bank=bank, region=region, timestamp=timestamp
            ):
                return window
        return None

    def _sample_outcome(
        self,
        *,
        payment_method: PaymentMethod,
        bank: Bank,
        attempt_number: int,
        active_window: Optional[IncidentWindow],
    ) -> tuple:
        base_rate = METHOD_BASE_SUCCESS_RATE[payment_method]
        bank_mult = BANK_SUCCESS_MULTIPLIER[bank]

        # Retries behave differently from first attempts: a retry that made
        # it back into the funnel has a mild extra success boost, modeling
        # transient issues resolving themselves.
        retry_mult = 1.0 + 0.03 * (attempt_number - 1)

        success_prob = base_rate * bank_mult * retry_mult

        extra_failure_weights: dict[FailureReason, float] = {}

        if active_window is not None:
            if active_window.scenario_type == ScenarioType.ISOLATED_FAILURES:
                # Uncorrelated extra failure chance, independent of the
                # method/bank/region-driven probability above.
                if self._rng.random() < active_window.isolated_failure_rate:
                    success_prob *= 0.0  # force this single event to fail
            else:
                success_prob *= active_window.success_rate_multiplier
                extra_failure_weights = active_window.failure_reason_weight_overrides

        success_prob = min(max(success_prob, 0.01), 0.999)

        if self._rng.random() < success_prob:
            return "success", None

        failure_reason = self._sample_failure_reason(payment_method, extra_failure_weights)
        return "failed", failure_reason

    def _sample_failure_reason(
        self,
        payment_method: PaymentMethod,
        extra_weights: dict[FailureReason, float],
    ) -> FailureReason:
        base_weights = METHOD_FAILURE_REASON_WEIGHTS[payment_method]
        reasons = list(base_weights.keys())
        weights = []
        for reason in reasons:
            w = base_weights[reason]
            if reason in extra_weights:
                w *= extra_weights[reason]
            weights.append(w)
        return self._rng.choices(reasons, weights=weights, k=1)[0]

    def _sample_latency(
        self,
        *,
        payment_method: PaymentMethod,
        failure_reason: Optional[FailureReason],
        attempt_number: int,
        active_window: Optional[IncidentWindow],
    ) -> float:
        mean, std = METHOD_BASE_LATENCY[payment_method]

        if failure_reason is not None and failure_reason in LATENCY_BOOST_FAILURE_REASONS:
            mean *= LATENCY_BOOST_FAILURE_REASONS[failure_reason]

        # Retry backoff adds a bit of latency per attempt.
        mean += 150.0 * (attempt_number - 1)

        if active_window is not None and active_window.scenario_type != ScenarioType.ISOLATED_FAILURES:
            mean *= active_window.latency_multiplier

        sample = self._rng.gauss(mean, std)
        return round(max(sample, 5.0), 2)
