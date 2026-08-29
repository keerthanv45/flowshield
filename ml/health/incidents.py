"""
Incident detection and classification for the Payment Health Engine.

Combines three signal sources into an `Incident`:
  1. Baseline deviation (from `ml.health.features`)
  2. Deterministic rule-based signals (this module)
  3. ML anomaly detection (`ml.health.anomaly`, overall-level only)

Classification into a scenario type (BANK_RAIL_DEGRADATION,
REGIONAL_DEGRADATION, LATENCY_SPIKE, MERCHANT_SYSTEM_DEGRADATION,
ISOLATED_FAILURES, NORMAL) is fully deterministic evidence-based logic —
NO LLM is used anywhere in this module, per the Phase 2 design principle.

============================================================
RULE-BASED SIGNAL THRESHOLDS (documented, not blindly chosen)
============================================================
  SUCCESS_RATE_DEGRADATION : success_rate_delta <= -0.10  (10 percentage
                              points below baseline) AND statistically
                              significant (see below)
  LATENCY_SPIKE (signal)   : latency_ratio >= 1.5  (50% above baseline)
  FAILURE_SURGE            : failure_rate_delta >= 0.10 AND statistically
                              significant
  VOLUME_ANOMALY           : volume_ratio <= 0.5 or >= 2.0

These thresholds are deliberately looser than the health-score falloff
bounds in `scoring.py` — they are meant to *flag attention*, not to be
the sole trigger for a CRITICAL incident. A single window crossing one
threshold produces a CANDIDATE classification; see PERSISTENCE below for
what promotes a candidate to a confirmed `Incident`.

WHY "statistically significant": with a 15-minute default window, the
observed window volume is small (~15 transactions/window overall, fewer
per bank/method/region slice). A first implementation gated
significance with a parametric binomial standard error
(`sqrt(p*(1-p)/n)`) — this UNDER-estimated real variance and was
measured, during development, to still flag ~8% of windows on a day with
no injected incident at all. The reason: each window blends several
payment methods, banks, and attempt numbers, each with a different true
success rate — that mixture has more window-to-window variance than a
homogeneous binomial(n, p) predicts (a form of overdispersion). The fix
actually shipped here uses the EMPIRICAL standard deviation of
window-level success/failure rates observed across NORMAL baseline
windows (see `ml.health.baseline.BaselineModel`, which now tracks std
alongside mean per hourly/group/global cell) as the z-test denominator
instead. SUCCESS_RATE_DEGRADATION and FAILURE_SURGE require both the
fixed percentage-point threshold AND a z-score (delta / empirical std)
of at least `RATE_SIGNAL_Z_THRESHOLD` (2.0). Latency and volume signals
are left as plain ratio thresholds — latency's coefficient of variation
is low enough at this window size that a 1.5x ratio is already well
outside sampling noise (verified during development), and volume is a
count comparison, not a rate, so the same proportion-noise argument does
not apply the same way.

============================================================
CONCENTRATION ANALYSIS
============================================================
To distinguish "one bank's rail is broken" from "everything is broken",
we look at how many members of a dimension (bank, payment_method,
region, bank+payment_method) show degradation (SUCCESS_RATE_DEGRADATION
or FAILURE_SURGE) at the same time window, restricted to members with at
least `MIN_GROUP_VOLUME` transactions in that window (avoids reacting to
a single unlucky transaction in a near-empty slice).

============================================================
CLASSIFICATION LOGIC (deterministic cascade, evaluated in order)
============================================================
  1. No overall-level signal AND not flagged anomalous -> NORMAL
  2. A small, concentrated set of (bank, payment_method) combos degraded
     (<=15% of considered combos) while bank-level and method-level
     concentration stay low -> BANK_RAIL_DEGRADATION
  3. Exactly one region shows degradation -> REGIONAL_DEGRADATION
  4. Latency ratio itself crosses the signal threshold AND degradation is
     broad across both banks and payment methods -> LATENCY_SPIKE
  5. Degradation is broad across both banks and payment methods (but
     latency is not the dominant driver) -> MERCHANT_SYSTEM_DEGRADATION
  6. Any remaining signal/anomaly without concentration -> ISOLATED_FAILURES

This mirrors the reasoning given in the Phase 2 brief directly.

============================================================
SEVERITY (deterministic, not AI-generated)
============================================================
  CRITICAL : health_score < 30  OR  success_rate_delta <= -0.30
  WARNING  : health_score < 60  OR  2+ rule signals fired
  INFO     : anything else that still produced a non-NORMAL classification

============================================================
PERSISTENCE (false-positive handling)
============================================================
Raw per-window classifications are grouped into contiguous "episodes" of
the same non-NORMAL incident_type. An episode is CONFIRMED (returned as
an `Incident`) only if:
  - it spans 2 or more consecutive windows (>= 30 minutes of evidence), OR
  - any window in the episode is independently CRITICAL severity
    (extreme single-window evidence bypasses the persistence requirement
    by design — we should not wait 30 minutes to surface a payment rail
    that's failing 90% of the time).
A single noisy window that never repeats and isn't extreme is downgraded
to a CANDIDATE and does not become an Incident. This is what keeps the
`isolated_failures` scenario from generating a flood of CRITICAL
incidents (see Phase 2 evaluation for the measured false-positive rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

SUCCESS_RATE_DEGRADATION_THRESHOLD = -0.10
LATENCY_SPIKE_RATIO_THRESHOLD = 1.5
FAILURE_SURGE_THRESHOLD = 0.10
VOLUME_ANOMALY_LOW = 0.5
VOLUME_ANOMALY_HIGH = 2.0

# Minimum z-score (observed deviation / EMPIRICAL baseline standard
# deviation — see ml.health.baseline and ml.health.features) required
# before a rate-based deviation counts as a real signal rather than
# small-sample/mixture noise. See module docstring for why this exists.
RATE_SIGNAL_Z_THRESHOLD = 2.0

# Concentration analysis (`_is_degraded`, used by `compute_concentration`)
# checks ~20 (bank, payment_method) combos, ~5 banks, ~4 methods, and ~6
# regions independently at every bucket. Testing that many members at the
# same z-threshold as a single overall test inflates the family-wise false
# positive rate (with ~20 combos at a 2.3% one-sided per-test rate, the
# chance at least one looks "degraded" purely by chance approaches ~35-40%
# per bucket) — this was measured directly during development as
# BANK_RAIL_DEGRADATION being spuriously confirmed on days with no
# injected bank-rail incident at all. A stricter, roughly
# Bonferroni-adjusted threshold (target ~5% family-wise error over ~20
# comparisons => per-test alpha ~0.0025 => z ~2.8) is used for
# member-level degradation inside concentration analysis specifically,
# while the single overall-level test in `rule_signals_for_row` keeps the
# looser 2.0 threshold.
CONCENTRATION_Z_THRESHOLD = 2.8

MIN_GROUP_VOLUME = 5
BANK_RAIL_MAX_DEGRADED_FRACTION = 0.15
BROAD_MIN_DEGRADED_BANKS = 3
BROAD_MIN_DEGRADED_METHODS = 2
BROAD_MIN_FRACTION = 0.5

# Dimension-concentration analysis (bank/method/region breakdowns) needs
# enough transactions per slice to be meaningful. At this dataset's scale
# (~20k events over 14 days => ~15 events per 15-minute window overall,
# split across 5 banks x 4 methods = 20 combos), a bank+payment_method
# slice gets under 1 event per 15-minute window on average — nowhere
# near MIN_GROUP_VOLUME. This was measured directly during development:
# with concentration computed at the 15-minute grain, BANK_RAIL_DEGRADATION
# was never confirmed even during the real injected bank-rail incident,
# because the (HDFC, UPI) slice almost always had 0-1 transactions in any
# single 15-minute window.
#
# Fix: concentration analysis (only) uses a coarser, separately-aggregated
# window (default 2 hours = 8x the 15-minute grain) so each dimension
# slice has enough volume to judge degradation reliably. The headline
# health snapshot, deviation features, and anomaly score all still operate
# at the 15-minute grain — every 15-minute window within the same 2-hour
# span simply shares the same concentration verdict, which also has the
# side benefit of extra smoothing in line with the persistence philosophy
# elsewhere in this module.
CONCENTRATION_WINDOW_MINUTES = 120

CRITICAL_HEALTH_SCORE = 30.0
CRITICAL_SUCCESS_DELTA = -0.30
WARNING_HEALTH_SCORE = 60.0
WARNING_MIN_SIGNALS = 2

MIN_EPISODE_WINDOWS_FOR_CONFIRMATION = 2


class IncidentType(str, Enum):
    NORMAL = "NORMAL"
    BANK_RAIL_DEGRADATION = "BANK_RAIL_DEGRADATION"
    REGIONAL_DEGRADATION = "REGIONAL_DEGRADATION"
    LATENCY_SPIKE = "LATENCY_SPIKE"
    MERCHANT_SYSTEM_DEGRADATION = "MERCHANT_SYSTEM_DEGRADATION"
    ISOLATED_FAILURES = "ISOLATED_FAILURES"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"


def rule_signals_for_row(row: dict) -> list[str]:
    """Deterministic rule-based signals for one overall-level window.
    Expects `success_rate_z`/`failure_rate_z` to already be present
    (added by `ml.health.features.add_deviation_features`)."""
    signals = []

    success_z = row.get("success_rate_z", 0.0)
    if row["success_rate_delta"] <= SUCCESS_RATE_DEGRADATION_THRESHOLD and success_z <= -RATE_SIGNAL_Z_THRESHOLD:
        signals.append("SUCCESS_RATE_DEGRADATION")

    if row["latency_ratio"] >= LATENCY_SPIKE_RATIO_THRESHOLD:
        signals.append("LATENCY_SPIKE")

    failure_z = row.get("failure_rate_z", 0.0)
    if row["failure_rate_delta"] >= FAILURE_SURGE_THRESHOLD and failure_z >= RATE_SIGNAL_Z_THRESHOLD:
        signals.append("FAILURE_SURGE")

    if row["volume_ratio"] <= VOLUME_ANOMALY_LOW or row["volume_ratio"] >= VOLUME_ANOMALY_HIGH:
        signals.append("VOLUME_ANOMALY")
    return signals


@dataclass
class ConcentrationResult:
    n_considered: int
    n_degraded: int
    degraded_values: list

    @property
    def fraction_degraded(self) -> float:
        return self.n_degraded / self.n_considered if self.n_considered else 0.0


def _is_degraded(row: pd.Series) -> bool:
    """Same significance-gated logic as the overall-level rule signals
    (see module docstring), applied to one member of a grouping (e.g. one
    bank) at one window, so concentration analysis isn't fooled by
    small-sample noise either. Expects `success_rate_z`/`failure_rate_z`
    to already be present. Uses `CONCENTRATION_Z_THRESHOLD` (stricter
    than the single-test `RATE_SIGNAL_Z_THRESHOLD`) because this function
    is applied across ~20 combos/members per bucket — see module
    docstring for the multiple-comparisons rationale."""
    success_degraded = (
        row["success_rate_delta"] <= SUCCESS_RATE_DEGRADATION_THRESHOLD
        and row.get("success_rate_z", 0.0) <= -CONCENTRATION_Z_THRESHOLD
    )
    failure_degraded = (
        row["failure_rate_delta"] >= FAILURE_SURGE_THRESHOLD
        and row.get("failure_rate_z", 0.0) >= CONCENTRATION_Z_THRESHOLD
    )
    return bool(success_degraded or failure_degraded)


def compute_concentration(
    group_df_at_window: pd.DataFrame,
    value_cols: list[str],
    min_volume: int = MIN_GROUP_VOLUME,
) -> ConcentrationResult:
    """`group_df_at_window`: rows for one grouping (e.g. by bank) at a
    single window_start, already deviation-feature-enriched.
    `value_cols`: the column(s) identifying each member (e.g. ["bank"]
    or ["bank", "payment_method"]).
    """
    if group_df_at_window.empty:
        return ConcentrationResult(n_considered=0, n_degraded=0, degraded_values=[])

    considered = group_df_at_window[group_df_at_window["transaction_count"] >= min_volume]
    if considered.empty:
        return ConcentrationResult(n_considered=0, n_degraded=0, degraded_values=[])

    degraded_mask = considered.apply(_is_degraded, axis=1)
    degraded = considered.loc[degraded_mask]

    if len(value_cols) == 1:
        degraded_values = degraded[value_cols[0]].tolist()
    else:
        degraded_values = list(degraded[value_cols].itertuples(index=False, name=None))

    return ConcentrationResult(
        n_considered=len(considered), n_degraded=len(degraded), degraded_values=degraded_values
    )


@dataclass
class ClassificationResult:
    incident_type: IncidentType
    signals: list[str]
    affected_dimensions: list[dict]
    evidence: list[str]


def classify_window(
    overall_row: dict,
    concentration: dict[str, ConcentrationResult],
) -> ClassificationResult:
    """Deterministic evidence-based classification for one overall-level
    window. `concentration` maps grouping name
    ("bank"/"payment_method"/"region"/"bank_payment_method") to its
    ConcentrationResult at this same window_start.
    """
    signals = rule_signals_for_row(overall_row)
    is_anomaly = bool(overall_row.get("is_anomaly", False))

    bank_c = concentration.get("bank", ConcentrationResult(0, 0, []))
    method_c = concentration.get("payment_method", ConcentrationResult(0, 0, []))
    region_c = concentration.get("region", ConcentrationResult(0, 0, []))
    bpm_c = concentration.get("bank_payment_method", ConcentrationResult(0, 0, []))

    if not signals and not is_anomaly:
        return ClassificationResult(IncidentType.NORMAL, [], [], [])

    # 2. Bank/rail degradation: a small, concentrated slice of
    # (bank, payment_method) combos degraded.
    if (
        bpm_c.n_considered >= 5
        and 1 <= bpm_c.n_degraded
        and bpm_c.fraction_degraded <= BANK_RAIL_MAX_DEGRADED_FRACTION
        and bank_c.n_degraded <= 1
        and method_c.n_degraded <= 1
    ):
        affected = [{"bank": b, "payment_method": m} for b, m in bpm_c.degraded_values]
        signals = list(dict.fromkeys(signals + ["BANK_CONCENTRATION"]))
        return ClassificationResult(IncidentType.BANK_RAIL_DEGRADATION, signals, affected, [])

    # 3. Regional degradation: exactly one region shows degradation.
    if region_c.n_considered >= 2 and region_c.n_degraded == 1:
        affected = [{"region": r} for r in region_c.degraded_values]
        signals = list(dict.fromkeys(signals + ["REGIONAL_CONCENTRATION"]))
        return ClassificationResult(IncidentType.REGIONAL_DEGRADATION, signals, affected, [])

    broad_banks = bank_c.n_degraded >= min(
        BROAD_MIN_DEGRADED_BANKS, max(1, round(bank_c.n_considered * BROAD_MIN_FRACTION))
    )
    broad_methods = method_c.n_degraded >= min(
        BROAD_MIN_DEGRADED_METHODS, max(1, round(method_c.n_considered * BROAD_MIN_FRACTION))
    )

    # 4. Global latency spike: latency itself is the dominant signal and
    # degradation is broad (not concentrated in one bank/method).
    if "LATENCY_SPIKE" in signals and broad_banks and broad_methods:
        affected = [{"bank": b} for b in bank_c.degraded_values] + [
            {"payment_method": m} for m in method_c.degraded_values
        ]
        return ClassificationResult(IncidentType.LATENCY_SPIKE, signals, affected, [])

    # 5. Merchant/system-wide degradation: broad across banks & methods,
    # not latency-dominant, not concentrated in one bank.
    if broad_banks and broad_methods:
        affected = [{"bank": b} for b in bank_c.degraded_values] + [
            {"payment_method": m} for m in method_c.degraded_values
        ]
        return ClassificationResult(IncidentType.MERCHANT_SYSTEM_DEGRADATION, signals, affected, [])

    # 6. A real signal or anomaly fired, but with no meaningful
    # concentration anywhere -> treat as scattered / isolated.
    return ClassificationResult(IncidentType.ISOLATED_FAILURES, signals, [], [])


def compute_severity(health_score: float, overall_row: dict, signals: list[str]) -> Severity:
    if health_score < CRITICAL_HEALTH_SCORE or overall_row["success_rate_delta"] <= CRITICAL_SUCCESS_DELTA:
        return Severity.CRITICAL
    if health_score < WARNING_HEALTH_SCORE or len(signals) >= WARNING_MIN_SIGNALS:
        return Severity.WARNING
    return Severity.INFO


def generate_evidence(
    overall_row: dict,
    classification: ClassificationResult,
) -> list[str]:
    lines = [
        f"Overall success rate: {overall_row['success_rate']:.4f} "
        f"(baseline: {overall_row['baseline_success_rate']:.4f}, "
        f"delta: {overall_row['success_rate_delta']:+.4f})",
        f"Overall failure rate: {overall_row['failure_rate']:.4f} "
        f"(baseline: {overall_row['baseline_failure_rate']:.4f})",
        f"Average latency: {overall_row['average_latency_ms']:.1f} ms "
        f"(baseline: {overall_row['baseline_average_latency_ms']:.1f} ms, "
        f"ratio: {overall_row['latency_ratio']:.2f}x)",
        f"Transaction volume: {overall_row['transaction_count']:.0f} "
        f"(baseline: {overall_row['baseline_transaction_count']:.1f}, "
        f"ratio: {overall_row['volume_ratio']:.2f}x)",
        f"Anomaly score: {overall_row.get('anomaly_score', 0.0):.3f} "
        f"(is_anomaly={overall_row.get('is_anomaly', False)})",
    ]

    if classification.incident_type == IncidentType.BANK_RAIL_DEGRADATION:
        for dim in classification.affected_dimensions:
            lines.append(
                f"Concentrated degradation detected in "
                f"{dim['bank']} + {dim['payment_method']}, while other "
                f"banks/methods remained near baseline"
            )
    elif classification.incident_type == IncidentType.REGIONAL_DEGRADATION:
        for dim in classification.affected_dimensions:
            lines.append(
                f"Region {dim['region']} shows degradation across multiple "
                f"banks/payment methods, while other regions remained near baseline"
            )
    elif classification.incident_type in (
        IncidentType.LATENCY_SPIKE,
        IncidentType.MERCHANT_SYSTEM_DEGRADATION,
    ):
        banks = sorted({d["bank"] for d in classification.affected_dimensions if "bank" in d})
        methods = sorted(
            {d["payment_method"] for d in classification.affected_dimensions if "payment_method" in d}
        )
        if banks:
            lines.append(f"Degradation spread across banks: {', '.join(banks)}")
        if methods:
            lines.append(f"Degradation spread across payment methods: {', '.join(methods)}")
    elif classification.incident_type == IncidentType.ISOLATED_FAILURES:
        lines.append(
            "Signal(s) fired without concentration in any single bank, "
            "payment method, or region — consistent with scattered, "
            "non-systemic failures"
        )

    return lines


@dataclass
class WindowClassification:
    """Raw, per-window classification before persistence is applied."""

    window_start: datetime
    window_end: datetime
    incident_type: IncidentType
    signals: list[str]
    affected_dimensions: list[dict]
    evidence: list[str]
    health_score: float
    anomaly_score: float
    is_anomaly: bool
    severity: Severity | None  # None for NORMAL windows


@dataclass
class Incident:
    incident_id: str
    detected_at: datetime
    window_start: datetime
    window_end: datetime
    severity: Severity
    incident_type: IncidentType
    anomaly_score: float
    health_score: float
    affected_dimensions: list[dict]
    signals: list[str]
    evidence: list[str]
    status: IncidentStatus
    n_windows: int = 1

    def as_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "detected_at": self.detected_at.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "severity": self.severity.value,
            "incident_type": self.incident_type.value,
            "anomaly_score": self.anomaly_score,
            "health_score": self.health_score,
            "affected_dimensions": self.affected_dimensions,
            "signals": self.signals,
            "evidence": self.evidence,
            "status": self.status.value,
            "n_windows": self.n_windows,
        }


def classify_all_windows(
    overall_scored_df: pd.DataFrame,
    grouped_scored: dict[str, pd.DataFrame],
    concentration_window_minutes: int = CONCENTRATION_WINDOW_MINUTES,
) -> list[WindowClassification]:
    """Classify every window in `overall_scored_df` (must have health
    score + anomaly columns already attached). `grouped_scored` maps
    grouping name -> deviation-feature-enriched DataFrame for that
    grouping (bank, payment_method, region, bank_payment_method), built
    at `concentration_window_minutes` granularity (see module docstring
    for why this is coarser than the 15-minute headline window).
    """
    results = []

    bank_df = grouped_scored.get("bank")
    method_df = grouped_scored.get("payment_method")
    region_df = grouped_scored.get("region")
    bpm_df = grouped_scored.get("bank_payment_method")

    freq = f"{concentration_window_minutes}min"

    for _, row in overall_scored_df.iterrows():
        ws = row["window_start"]
        bucket = pd.Timestamp(ws).floor(freq)
        concentration = {
            "bank": compute_concentration(bank_df[bank_df["window_start"] == bucket], ["bank"])
            if bank_df is not None
            else ConcentrationResult(0, 0, []),
            "payment_method": compute_concentration(
                method_df[method_df["window_start"] == bucket], ["payment_method"]
            )
            if method_df is not None
            else ConcentrationResult(0, 0, []),
            "region": compute_concentration(region_df[region_df["window_start"] == bucket], ["region"])
            if region_df is not None
            else ConcentrationResult(0, 0, []),
            "bank_payment_method": compute_concentration(
                bpm_df[bpm_df["window_start"] == bucket], ["bank", "payment_method"]
            )
            if bpm_df is not None
            else ConcentrationResult(0, 0, []),
        }

        row_dict = row.to_dict()
        classification = classify_window(row_dict, concentration)
        evidence = generate_evidence(row_dict, classification)

        severity = None
        if classification.incident_type != IncidentType.NORMAL:
            severity = compute_severity(row["health_score"], row_dict, classification.signals)

        results.append(
            WindowClassification(
                window_start=row["window_start"],
                window_end=row["window_end"],
                incident_type=classification.incident_type,
                signals=classification.signals,
                affected_dimensions=classification.affected_dimensions,
                evidence=evidence,
                health_score=row["health_score"],
                anomaly_score=row.get("anomaly_score", 0.0),
                is_anomaly=bool(row.get("is_anomaly", False)),
                severity=severity,
            )
        )

    return results


def apply_persistence(
    classifications: list[WindowClassification],
    min_episode_windows: int = MIN_EPISODE_WINDOWS_FOR_CONFIRMATION,
) -> list[Incident]:
    """Group consecutive same-type non-NORMAL windows into episodes and
    confirm only those meeting the persistence rule (see module
    docstring). Returns CONFIRMED incidents only — candidates that never
    get confirmed are dropped (their information is still visible in the
    raw per-window classification list for evaluation purposes).
    """
    incidents: list[Incident] = []
    episode: list[WindowClassification] = []

    def flush():
        if not episode:
            return
        has_critical = any(w.severity == Severity.CRITICAL for w in episode)
        if len(episode) >= min_episode_windows or has_critical:
            severities = [w.severity for w in episode if w.severity is not None]
            worst = Severity.CRITICAL if Severity.CRITICAL in severities else (
                Severity.WARNING if Severity.WARNING in severities else Severity.INFO
            )
            # Representative evidence/affected dims: the window with the
            # lowest health_score (most degraded) in the episode.
            worst_window = min(episode, key=lambda w: w.health_score)
            incidents.append(
                Incident(
                    incident_id=(
                        f"inc_{episode[0].window_start.strftime('%Y%m%dT%H%M%S')}_"
                        f"{episode[0].incident_type.value}"
                    ),
                    detected_at=episode[0].window_start,
                    window_start=episode[0].window_start,
                    window_end=episode[-1].window_end,
                    severity=worst,
                    incident_type=episode[0].incident_type,
                    anomaly_score=max(w.anomaly_score for w in episode),
                    health_score=min(w.health_score for w in episode),
                    affected_dimensions=worst_window.affected_dimensions,
                    signals=sorted({s for w in episode for s in w.signals}),
                    evidence=worst_window.evidence,
                    status=IncidentStatus.CONFIRMED,
                    n_windows=len(episode),
                )
            )
        episode.clear()

    for wc in classifications:
        if wc.incident_type == IncidentType.NORMAL:
            flush()
            continue
        if episode and episode[-1].incident_type != wc.incident_type:
            flush()
        episode.append(wc)

    flush()
    return incidents
