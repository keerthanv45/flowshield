"""
Train / validation / test split strategy for future ML work.

We do NOT train any model in Phase 1. This module only prepares the
splitting mechanism so future phases (anomaly detection, recovery
prediction) have a consistent, leakage-free way to slice the dataset.

Strategy: TIME-BASED split, not random shuffling.

Payment reliability is a time-series problem — future incidents must be
predicted from past behavior, not learned from behavior that happens
"after" the point being evaluated. A random row-level split would leak
information across an incident window (e.g. training on some events from
the middle of an incident and testing on others from the same incident),
which would make offline metrics look better than they would in
production. Splitting by contiguous time ranges avoids that.

Default split (by wall-clock time, not row count):
  - train:      first 70% of the time range
  - validation: next 15% of the time range
  - test:       final 15% of the time range
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        return (
            f"train={len(self.train)} rows, "
            f"validation={len(self.validation)} rows, "
            f"test={len(self.test)} rows"
        )


def time_based_split(df: pd.DataFrame, ratios: SplitRatios | None = None) -> DatasetSplit:
    """Split a dataframe into train/validation/test by contiguous time ranges.

    Requires a `timestamp` column. Does not shuffle — order is preserved,
    which is what makes this leakage-safe for a time-series problem.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column for a time-based split")

    ratios = ratios or SplitRatios()
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(sorted_df)
    train_end = int(n * ratios.train)
    validation_end = train_end + int(n * ratios.validation)

    train_df = sorted_df.iloc[:train_end].reset_index(drop=True)
    validation_df = sorted_df.iloc[train_end:validation_end].reset_index(drop=True)
    test_df = sorted_df.iloc[validation_end:].reset_index(drop=True)

    return DatasetSplit(train=train_df, validation=validation_df, test=test_df)


# ---------------------------------------------------------------------------
# Aggregated feature set the future anomaly detector should be able to build
# from these splits. Not implemented here — this is documentation of intent
# so Phase 2 knows what to build on top of the row-level data.
# ---------------------------------------------------------------------------

PLANNED_AGGREGATE_FEATURES = [
    "transaction_volume",       # count of events per time bucket
    "success_rate",             # per time bucket, optionally per segment
    "failure_rate",
    "average_latency_ms",
    "p95_latency_ms",
    "failure_type_proportions",  # distribution over FailureReason per bucket
    "payment_method_distribution",
    "bank_distribution",
    "regional_distribution",
]
