"""
Anomaly detection for the Payment Health Engine.

Wraps scikit-learn's IsolationForest over AGGREGATED time-window features
(never raw transaction rows — a single payment failing is not itself an
anomaly; a window's worth of behavior looking unlike normal windows is).

============================================================
FEATURES USED
============================================================
  success_rate, failure_rate, p95_latency_ms, transaction_count,
  timeout_rate, network_error_rate, technical_error_rate,
  success_rate_delta, latency_ratio, volume_ratio

`p95_latency_ms` and `transaction_count` are on very different scales
than the rate/ratio features, so all features are standardized
(zero mean, unit variance) with a `StandardScaler` fit on the SAME data
used to fit the IsolationForest, before being passed to the model.

============================================================
TRAINING DATA / LEAKAGE
============================================================
The detector is fit ONLY on NORMAL windows (see
`ml.health.baseline.label_incident_affected`) from the TRAIN split (the
first 70% of the time-ordered dataset — see `ml/evaluation/split.py`).
This mirrors the baseline model's leakage rule: the model must never be
told what "normal" looks like using windows that are themselves part of
a known incident, and it must never see validation/test-period data
during fitting.

============================================================
OUTPUT SEMANTICS
============================================================
IsolationForest is unsupervised and its outputs are NOT calibrated
probabilities. This module exposes:

  - `anomaly_score`: higher = more anomalous. Derived from sklearn's
    `score_samples` (which returns higher = more normal), negated and
    then min-max normalized against the TRAINING data's own score range
    so that ~0 means "typical of training data" and ~1 means "as unusual
    as the most unusual point seen while fitting" (values can exceed 1
    for windows more extreme than anything in training — that's
    expected and meaningful, not a bug).
  - `is_anomaly`: boolean, from IsolationForest's own `predict()` (-1 =
    anomaly), which uses the `contamination` parameter set at
    construction time as its internal threshold.
  - `low_reliability`: True when `transaction_count` in the window is
    below `MIN_RELIABLE_VOLUME` — a reliability flag, not a confidence
    score. Rates computed from a handful of transactions are noisy;
    this flag lets downstream logic discount low-volume windows rather
    than pretending the model has graded its own confidence.

We deliberately do NOT report `anomaly_score` as a "confidence" or
"probability" anywhere in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    "success_rate",
    "failure_rate",
    "p95_latency_ms",
    "transaction_count",
    "timeout_rate",
    "network_error_rate",
    "technical_error_rate",
    "success_rate_delta",
    "latency_ratio",
    "volume_ratio",
]

DEFAULT_CONTAMINATION = 0.05
DEFAULT_N_ESTIMATORS = 200
DEFAULT_RANDOM_STATE = 42
MIN_RELIABLE_VOLUME = 5


@dataclass
class AnomalyResult:
    anomaly_score: float
    is_anomaly: bool
    low_reliability: bool


@dataclass
class AnomalyDetector:
    contamination: float = DEFAULT_CONTAMINATION
    n_estimators: int = DEFAULT_N_ESTIMATORS
    random_state: int = DEFAULT_RANDOM_STATE
    min_reliable_volume: int = MIN_RELIABLE_VOLUME

    _scaler: StandardScaler = field(default=None, repr=False)
    _model: IsolationForest = field(default=None, repr=False)
    _train_score_min: float = field(default=0.0, repr=False)
    _train_score_max: float = field(default=1.0, repr=False)
    _is_fit: bool = field(default=False, repr=False)

    def fit(self, normal_df: pd.DataFrame) -> "AnomalyDetector":
        """Fit on aggregated, deviation-feature-enriched NORMAL windows
        (see module docstring for the leakage rule this depends on)."""
        if normal_df.empty or len(normal_df) < 10:
            raise ValueError(
                f"AnomalyDetector.fit requires at least 10 normal windows, got {len(normal_df)}"
            )

        X = normal_df[ANOMALY_FEATURES].to_numpy(dtype=float)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled)

        raw_scores = -self._model.score_samples(X_scaled)  # higher = more anomalous
        self._train_score_min = float(raw_scores.min())
        self._train_score_max = float(raw_scores.max())
        self._is_fit = True
        return self

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        span = self._train_score_max - self._train_score_min
        if span < 1e-9:
            return np.zeros_like(raw_scores)
        return (raw_scores - self._train_score_min) / span

    def score(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Score arbitrary windows (normal or incident, train or held-out
        — the caller decides what to pass in; this method has no leakage
        concerns of its own since it does not refit)."""
        if not self._is_fit:
            raise RuntimeError("AnomalyDetector must be fit before scoring")
        if df.empty:
            return []

        X = df[ANOMALY_FEATURES].to_numpy(dtype=float)
        X_scaled = self._scaler.transform(X)

        raw_scores = -self._model.score_samples(X_scaled)
        normalized = self._normalize_scores(raw_scores)
        predictions = self._model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        volumes = df["transaction_count"].to_numpy(dtype=float)

        results = []
        for score, pred, volume in zip(normalized, predictions, volumes):
            results.append(
                AnomalyResult(
                    anomaly_score=float(score),
                    is_anomaly=bool(pred == -1),
                    low_reliability=bool(volume < self.min_reliable_volume),
                )
            )
        return results

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as `score`, but returns the input df with three new
        columns (`anomaly_score`, `is_anomaly`, `low_reliability`)
        appended, which is the more convenient form for the pipeline."""
        results = self.score(df)
        out = df.reset_index(drop=True).copy()
        out["anomaly_score"] = [r.anomaly_score for r in results]
        out["is_anomaly"] = [r.is_anomaly for r in results]
        out["low_reliability"] = [r.low_reliability for r in results]
        return out
