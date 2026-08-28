"""
Data validation service for FlowShield.

Two layers of validation:

1. Row-level: parse a raw dict into a `PaymentEvent`, relying on Pydantic
   for field-level and cross-field checks (see `payment_event.py`).
2. Dataset-level: checks that only make sense across many rows, such as
   duplicate event IDs.

The goal is *useful* error messages a human (or later, an LLM) can act on,
not just "validation failed".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from backend.app.schemas.payment_event import PaymentEvent


@dataclass
class RowError:
    row_index: int
    event_id: str | None
    errors: list[str]

    def __str__(self) -> str:
        joined = "; ".join(self.errors)
        return f"row {self.row_index} (event_id={self.event_id!r}): {joined}"


@dataclass
class ValidationReport:
    total_rows: int
    valid_rows: int
    row_errors: list[RowError] = field(default_factory=list)
    duplicate_event_ids: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.row_errors and not self.duplicate_event_ids

    @property
    def invalid_row_count(self) -> int:
        return len(self.row_errors)

    def summary(self) -> str:
        lines = [
            f"Total rows checked : {self.total_rows}",
            f"Valid rows         : {self.valid_rows}",
            f"Invalid rows        : {self.invalid_row_count}",
            f"Duplicate event_ids : {len(self.duplicate_event_ids)}",
        ]
        if self.row_errors:
            lines.append("")
            lines.append("First errors:")
            for row_error in self.row_errors[:10]:
                lines.append(f"  - {row_error}")
            if len(self.row_errors) > 10:
                lines.append(f"  ... and {len(self.row_errors) - 10} more")
        if self.duplicate_event_ids:
            lines.append("")
            lines.append(
                f"Duplicate event_id examples: {self.duplicate_event_ids[:10]}"
            )
        return "\n".join(lines)


def _format_pydantic_error(exc: ValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        messages.append(f"{loc}: {err['msg']}")
    return messages


def parse_row(row: dict[str, Any]) -> tuple[PaymentEvent | None, list[str]]:
    """Attempt to parse a single raw row into a PaymentEvent.

    Returns (event, errors). event is None if parsing failed.
    """
    try:
        event = PaymentEvent(**row)
        return event, []
    except ValidationError as exc:
        return None, _format_pydantic_error(exc)
    except TypeError as exc:
        # e.g. missing required keys entirely, unexpected kwargs, etc.
        return None, [f"malformed row: {exc}"]


def validate_dataset(rows: Iterable[dict[str, Any]]) -> ValidationReport:
    """Validate a full dataset of raw payment-event rows.

    Checks performed:
      - each row parses into a valid PaymentEvent (field + cross-field rules)
      - event_id uniqueness across the whole dataset
    """
    rows = list(rows)
    total_rows = len(rows)
    row_errors: list[RowError] = []
    valid_events: list[PaymentEvent] = []

    for idx, row in enumerate(rows):
        event, errors = parse_row(row)
        if event is None:
            row_errors.append(
                RowError(row_index=idx, event_id=row.get("event_id"), errors=errors)
            )
        else:
            valid_events.append(event)

    id_counts = Counter(e.event_id for e in valid_events)
    duplicate_event_ids = sorted([eid for eid, count in id_counts.items() if count > 1])

    return ValidationReport(
        total_rows=total_rows,
        valid_rows=len(valid_events) - _duplicate_extra_count(id_counts),
        row_errors=row_errors,
        duplicate_event_ids=duplicate_event_ids,
    )


def _duplicate_extra_count(id_counts: Counter) -> int:
    """Number of 'extra' rows beyond the first occurrence for each duplicated id."""
    return sum(count - 1 for count in id_counts.values() if count > 1)
