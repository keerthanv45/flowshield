"""
Phase 2 evaluation: how well does the health/incident engine recover the
KNOWN synthetic incident schedule?

============================================================
GROUND TRUTH LABELING
============================================================
Every 15-minute headline window is labeled from
`data/synthetic/incident_windows.json`: if its time range overlaps a
known incident window, its ground-truth label is that incident's
`scenario_type` (mapped to the matching `IncidentType`); otherwise it is
NORMAL. If a window overlaps more than one incident window (shouldn't
happen with the default schedule, but handled defensively), the first
overlapping window wins.

============================================================
METRICS
============================================================
  - classification_accuracy: exact match between the raw (pre-persistence)
    per-window predicted `incident_type` and the ground-truth label, over
    ALL windows (6-class: NORMAL + 5 scenario types).
  - Binary "systemic incident" task (systemic = anything except NORMAL
    and ISOLATED_FAILURES — i.e. BANK_RAIL_DEGRADATION,
    REGIONAL_DEGRADATION, LATENCY_SPIKE, MERCHANT_SYSTEM_DEGRADATION):
    precision / recall / F1, evaluated on CONFIRMED incidents (i.e. after
    persistence) vs ground truth. This is the metric that matters
    operationally — "did we page someone for a real incident, and did we
    avoid paging for noise".
  - detection_rate: recall of the binary systemic task (same number,
    named per the Phase 2 brief's terminology).
  - false_positive_rate: fraction of ground-truth NORMAL windows that
    are CONFIRMED as some non-NORMAL incident.
  - per_scenario: for each of the 5 injected scenario types, how many of
    its ground-truth windows were (a) confirmed as ANY incident, (b)
    confirmed with the CORRECT type.
  - anomaly-detector-specific counts: number of windows flagged
    `is_anomaly`, split by whether they fall inside vs outside a known
    incident window.

All numbers here are computed from an ACTUAL run of the pipeline over
the ACTUAL synthetic dataset — nothing in this module is hand-authored
or illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ml.health.incidents import Incident, IncidentType, WindowClassification

SYSTEMIC_TYPES = {
    IncidentType.BANK_RAIL_DEGRADATION,
    IncidentType.REGIONAL_DEGRADATION,
    IncidentType.LATENCY_SPIKE,
    IncidentType.MERCHANT_SYSTEM_DEGRADATION,
}

SCENARIO_TYPE_TO_INCIDENT_TYPE = {
    "bank_rail_degradation": IncidentType.BANK_RAIL_DEGRADATION,
    "regional_degradation": IncidentType.REGIONAL_DEGRADATION,
    "latency_spike": IncidentType.LATENCY_SPIKE,
    "merchant_system_degradation": IncidentType.MERCHANT_SYSTEM_DEGRADATION,
    "isolated_failures": IncidentType.ISOLATED_FAILURES,
}


def label_ground_truth(
    window_starts: pd.Series,
    window_ends: pd.Series,
    incident_windows: list[dict],
) -> list[IncidentType]:
    """Ground-truth IncidentType label for each window, from the known
    synthetic incident schedule (time-overlap only — ignores dimension
    targeting, since ground truth here is "what scenario was active",
    not "which dimension slice it targeted")."""
    labels = []
    for ws, we in zip(window_starts, window_ends):
        label = IncidentType.NORMAL
        for w in incident_windows:
            w_start = pd.Timestamp(w["start"])
            w_end = pd.Timestamp(w["end"])
            if ws < w_end and we > w_start:
                label = SCENARIO_TYPE_TO_INCIDENT_TYPE.get(w["scenario_type"], IncidentType.NORMAL)
                break
        labels.append(label)
    return labels


def confirmed_incident_type_by_window(
    window_starts: pd.Series,
    incidents: list[Incident],
) -> list:
    """For each window_start, the incident_type of the CONFIRMED incident
    episode it falls inside, or None if it's not part of any confirmed
    episode."""
    result = []
    for ws in window_starts:
        found = None
        for inc in incidents:
            if inc.window_start <= ws < inc.window_end:
                found = inc.incident_type
                break
        result.append(found)
    return result


@dataclass
class ScenarioResult:
    scenario_type: str
    n_ground_truth_windows: int
    n_confirmed_any: int
    n_confirmed_correct_type: int

    @property
    def detection_rate(self) -> float:
        return self.n_confirmed_any / self.n_ground_truth_windows if self.n_ground_truth_windows else 0.0

    @property
    def correct_type_rate(self) -> float:
        return (
            self.n_confirmed_correct_type / self.n_ground_truth_windows
            if self.n_ground_truth_windows
            else 0.0
        )


@dataclass
class EvaluationReport:
    n_windows: int
    classification_accuracy: float
    detection_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1: float
    per_scenario: list = field(default_factory=list)
    n_anomalous_windows: int = 0
    n_anomalous_during_incident: int = 0
    n_anomalous_during_normal: int = 0
    label: str = "full dataset"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_windows": self.n_windows,
            "classification_accuracy": self.classification_accuracy,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_scenario": [
                {
                    "scenario_type": s.scenario_type,
                    "n_ground_truth_windows": s.n_ground_truth_windows,
                    "n_confirmed_any": s.n_confirmed_any,
                    "n_confirmed_correct_type": s.n_confirmed_correct_type,
                    "detection_rate": s.detection_rate,
                    "correct_type_rate": s.correct_type_rate,
                }
                for s in self.per_scenario
            ],
            "n_anomalous_windows": self.n_anomalous_windows,
            "n_anomalous_during_incident": self.n_anomalous_during_incident,
            "n_anomalous_during_normal": self.n_anomalous_during_normal,
        }

    def format_text(self) -> str:
        lines = [
            f"--- Evaluation: {self.label} ({self.n_windows} windows) ---",
            f"Classification accuracy (6-class, raw per-window): {self.classification_accuracy:.4f}",
            f"Systemic-incident detection rate (recall)         : {self.detection_rate:.4f}",
            f"False positive rate (normal windows confirmed)    : {self.false_positive_rate:.4f}",
            f"Precision                                         : {self.precision:.4f}",
            f"Recall                                            : {self.recall:.4f}",
            f"F1                                                : {self.f1:.4f}",
            f"Anomalous windows (is_anomaly=True)               : {self.n_anomalous_windows} "
            f"({self.n_anomalous_during_incident} during incident, "
            f"{self.n_anomalous_during_normal} during normal)",
            "",
            "Per-scenario results:",
        ]
        for s in self.per_scenario:
            lines.append(
                f"  {s.scenario_type:<28} gt_windows={s.n_ground_truth_windows:<4} "
                f"confirmed_any={s.n_confirmed_any:<4} ({s.detection_rate:.2%})  "
                f"confirmed_correct_type={s.n_confirmed_correct_type:<4} ({s.correct_type_rate:.2%})"
            )
        return "\n".join(lines)


def evaluate(
    classifications: list,
    incidents: list,
    incident_windows: list[dict],
    label: str = "full dataset",
) -> EvaluationReport:
    """Evaluate detection/classification against known ground truth.

    `classifications`: raw per-window classifications (pre-persistence),
    used for the 6-class classification-accuracy metric.
    `incidents`: CONFIRMED incidents (post-persistence), used for the
    detection-rate / false-positive-rate / precision / recall / F1
    metrics, since those are meant to reflect what would actually be
    surfaced to a human, not every noisy candidate window.
    """
    window_starts = pd.Series([c.window_start for c in classifications])
    window_ends = pd.Series([c.window_end for c in classifications])

    ground_truth = label_ground_truth(window_starts, window_ends, incident_windows)
    raw_predicted = [c.incident_type for c in classifications]
    confirmed_predicted = confirmed_incident_type_by_window(window_starts, incidents)

    n = len(classifications)
    correct = sum(1 for gt, pred in zip(ground_truth, raw_predicted) if gt == pred)
    classification_accuracy = correct / n if n else 0.0

    gt_systemic = [gt in SYSTEMIC_TYPES for gt in ground_truth]
    pred_systemic_confirmed = [
        (pred is not None and pred in SYSTEMIC_TYPES) for pred in confirmed_predicted
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_systemic, pred_systemic_confirmed, average="binary", zero_division=0
    )

    gt_normal_mask = [gt == IncidentType.NORMAL for gt in ground_truth]
    n_gt_normal = sum(gt_normal_mask)
    n_false_positive = sum(
        1
        for is_normal, pred in zip(gt_normal_mask, confirmed_predicted)
        if is_normal and pred is not None
    )
    false_positive_rate = n_false_positive / n_gt_normal if n_gt_normal else 0.0

    is_anomaly = [c.is_anomaly for c in classifications]
    is_incident_gt = [gt != IncidentType.NORMAL for gt in ground_truth]
    n_anomalous = sum(is_anomaly)
    n_anomalous_during_incident = sum(1 for a, i in zip(is_anomaly, is_incident_gt) if a and i)
    n_anomalous_during_normal = sum(1 for a, i in zip(is_anomaly, is_incident_gt) if a and not i)

    per_scenario = []
    for scenario_type, incident_type in SCENARIO_TYPE_TO_INCIDENT_TYPE.items():
        gt_mask = [gt == incident_type for gt in ground_truth]
        n_gt = sum(gt_mask)
        n_confirmed_any = sum(
            1 for is_gt, pred in zip(gt_mask, confirmed_predicted) if is_gt and pred is not None
        )
        n_confirmed_correct = sum(
            1 for is_gt, pred in zip(gt_mask, confirmed_predicted) if is_gt and pred == incident_type
        )
        per_scenario.append(
            ScenarioResult(
                scenario_type=scenario_type,
                n_ground_truth_windows=n_gt,
                n_confirmed_any=n_confirmed_any,
                n_confirmed_correct_type=n_confirmed_correct,
            )
        )

    return EvaluationReport(
        n_windows=n,
        classification_accuracy=classification_accuracy,
        detection_rate=float(recall),
        false_positive_rate=false_positive_rate,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        per_scenario=per_scenario,
        n_anomalous_windows=n_anomalous,
        n_anomalous_during_incident=n_anomalous_during_incident,
        n_anomalous_during_normal=n_anomalous_during_normal,
        label=label,
    )
