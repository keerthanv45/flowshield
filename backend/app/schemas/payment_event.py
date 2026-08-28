"""
Canonical payment event schema for FlowShield.

This is the single source of truth for what a "payment event" looks like
anywhere in the system (synthetic data generation, validation, analysis,
and — in later phases — the health engine and anomaly detector).

All data produced by this project in Phase 1 is SYNTHETIC. Nothing here
talks to a real payment gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaymentMethod(str, Enum):
    UPI = "UPI"
    CARD = "CARD"
    NETBANKING = "NETBANKING"
    WALLET = "WALLET"


class Bank(str, Enum):
    HDFC = "HDFC"
    ICICI = "ICICI"
    SBI = "SBI"
    AXIS = "AXIS"
    KOTAK = "KOTAK"


class Region(str, Enum):
    KA = "KA"
    MH = "MH"
    DL = "DL"
    TN = "TN"
    TS = "TS"
    AP = "AP"


class PaymentStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class FailureReason(str, Enum):
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    AUTHENTICATION_FAILED = "authentication_failed"
    BANK_DECLINED = "bank_declined"
    TECHNICAL_ERROR = "technical_error"
    UNKNOWN = "unknown"


# Failure reasons that are legitimately associated with elevated latency.
# Used both for validation "sanity" and for the synthetic generator, so the
# two stay consistent with each other.
LATENCY_CORRELATED_FAILURE_REASONS = {
    FailureReason.TIMEOUT,
    FailureReason.NETWORK_ERROR,
    FailureReason.TECHNICAL_ERROR,
}

# Currencies supported by the synthetic dataset. Kept small and explicit
# rather than pulling in a full ISO-4217 library for Phase 1.
ALLOWED_CURRENCIES = {"INR"}

MAX_SENSIBLE_LATENCY_MS = 120_000  # 2 minutes; anything above is almost certainly bad data
MAX_SENSIBLE_ATTEMPT_NUMBER = 10


class PaymentEvent(BaseModel):
    """A single payment lifecycle event.

    NOTE: This models one *event* (e.g. a status transition / attempt),
    not necessarily a whole payment's entire lifecycle. A single
    ``payment_id`` may have multiple events (multiple attempts).
    """

    event_id: str = Field(..., min_length=1, description="Unique identifier for this event")
    payment_id: str = Field(..., min_length=1, description="Identifier for the logical payment")
    timestamp: datetime = Field(..., description="UTC timestamp the event occurred")

    amount: float = Field(..., gt=0, description="Transaction amount, must be > 0")
    currency: str = Field(default="INR", description="ISO-ish currency code")

    payment_method: PaymentMethod
    bank: Bank
    region: Region

    status: PaymentStatus
    failure_reason: Optional[FailureReason] = Field(
        default=None, description="Populated only when status == failed"
    )

    latency_ms: float = Field(..., ge=0, description="End-to-end processing latency in ms")
    customer_id: str = Field(..., min_length=1)
    attempt_number: int = Field(..., ge=1, description="1 for first attempt, 2+ for retries")

    # -- field validators -------------------------------------------------

    @field_validator("currency")
    @classmethod
    def currency_must_be_supported(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in ALLOWED_CURRENCIES:
            raise ValueError(
                f"Unsupported currency '{v}'. Supported currencies: {sorted(ALLOWED_CURRENCIES)}"
            )
        return v

    @field_validator("latency_ms")
    @classmethod
    def latency_must_be_sensible(cls, v: float) -> float:
        if v > MAX_SENSIBLE_LATENCY_MS:
            raise ValueError(
                f"latency_ms={v} exceeds sensible maximum of {MAX_SENSIBLE_LATENCY_MS}ms"
            )
        return v

    @field_validator("attempt_number")
    @classmethod
    def attempt_number_must_be_sensible(cls, v: int) -> int:
        if v > MAX_SENSIBLE_ATTEMPT_NUMBER:
            raise ValueError(
                f"attempt_number={v} exceeds sensible maximum of {MAX_SENSIBLE_ATTEMPT_NUMBER}"
            )
        return v

    # -- cross-field validation -------------------------------------------

    @model_validator(mode="after")
    def status_and_failure_reason_must_be_consistent(self) -> "PaymentEvent":
        if self.status == PaymentStatus.FAILED and self.failure_reason is None:
            raise ValueError("failure_reason is required when status == 'failed'")

        if self.status != PaymentStatus.FAILED and self.failure_reason is not None:
            raise ValueError(
                f"failure_reason must be None when status == '{self.status.value}' "
                "(only 'failed' events may carry a failure_reason)"
            )

        return self

    model_config = ConfigDict(
        use_enum_values=False,
        json_schema_extra={
            "description": (
                "SYNTHETIC payment event schema for FlowShield Phase 1. "
                "No field here originates from a real payment gateway."
            )
        },
    )
