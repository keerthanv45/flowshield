"""
Controlled incident scenario definitions.

An `IncidentWindow` describes a time-bounded, dimension-targeted degradation
that the generator applies on top of baseline behavior. Windows are the
*ground truth* used later to evaluate anomaly/incident detection — they are
NOT part of the canonical `PaymentEvent` schema, since a real payment
gateway would never hand you a "this was an incident" label.

Six scenario types, each with a distinguishable signature:

  1. BANK_RAIL_DEGRADATION      - one (payment_method, bank) pair degrades hard
  2. REGIONAL_DEGRADATION       - one region degrades across methods/banks
  3. LATENCY_SPIKE              - global latency + timeout/network failures rise
  4. MERCHANT_SYSTEM_DEGRADATION- broad degradation, spread evenly (not one bank)
  5. ISOLATED_FAILURES          - scattered, non-systemic noise (false-positive bait)
  6. NORMAL_TRAFFIC             - no incident; baseline only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from backend.app.schemas.payment_event import (
    Bank,
    FailureReason,
    PaymentMethod,
    Region,
)


class ScenarioType(str, Enum):
    BANK_RAIL_DEGRADATION = "bank_rail_degradation"
    REGIONAL_DEGRADATION = "regional_degradation"
    LATENCY_SPIKE = "latency_spike"
    MERCHANT_SYSTEM_DEGRADATION = "merchant_system_degradation"
    ISOLATED_FAILURES = "isolated_failures"
    NORMAL_TRAFFIC = "normal_traffic"


@dataclass(frozen=True)
class IncidentWindow:
    """A single scheduled incident.

    Attributes:
        scenario_type: which scenario signature this window represents.
        start: inclusive start timestamp.
        end: exclusive end timestamp.
        target_payment_method: if set, only events with this method are affected.
        target_bank: if set, only events with this bank are affected.
        target_region: if set, only events with this region are affected.
        success_rate_multiplier: multiplies baseline success probability
            (values < 1.0 make failures more likely).
        latency_multiplier: multiplies baseline latency mean.
        failure_reason_weight_overrides: extra weight added to specific
            failure reasons for affected events during this window.
        isolated_failure_rate: for ISOLATED_FAILURES only — an additional
            small, uncorrelated per-event failure probability applied
            dataset-wide regardless of method/bank/region, so it does not
            concentrate anywhere.
        description: human-readable label, useful for the incident log.
    """

    scenario_type: ScenarioType
    start: datetime
    end: datetime
    target_payment_method: Optional[PaymentMethod] = None
    target_bank: Optional[Bank] = None
    target_region: Optional[Region] = None
    success_rate_multiplier: float = 1.0
    latency_multiplier: float = 1.0
    failure_reason_weight_overrides: dict[FailureReason, float] = field(default_factory=dict)
    isolated_failure_rate: float = 0.0
    description: str = ""

    def matches(
        self,
        *,
        payment_method: PaymentMethod,
        bank: Bank,
        region: Region,
        timestamp: datetime,
    ) -> bool:
        if not (self.start <= timestamp < self.end):
            return False
        if self.target_payment_method is not None and payment_method != self.target_payment_method:
            return False
        if self.target_bank is not None and bank != self.target_bank:
            return False
        if self.target_region is not None and region != self.target_region:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "scenario_type": self.scenario_type.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "target_payment_method": self.target_payment_method.value
            if self.target_payment_method
            else None,
            "target_bank": self.target_bank.value if self.target_bank else None,
            "target_region": self.target_region.value if self.target_region else None,
            "success_rate_multiplier": self.success_rate_multiplier,
            "latency_multiplier": self.latency_multiplier,
            "isolated_failure_rate": self.isolated_failure_rate,
            "description": self.description,
        }
