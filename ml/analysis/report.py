"""
Basic, reproducible analysis over the synthetic payment dataset.

This is intentionally simple descriptive analytics (no ML) — it exists so
that (a) humans can sanity-check the generated data, and (b) later phases
have a known-good baseline to compare a "health engine" against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.app.schemas.payment_event import PaymentEvent


def events_to_dataframe(events: list[PaymentEvent]) -> pd.DataFrame:
    rows = [e.model_dump() for e in events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@dataclass
class AnalysisReport:
    total_events: int
    success_rate: float
    failure_rate: float
    failure_reason_counts: dict[str, int]
    payment_method_distribution: dict[str, int]
    bank_distribution: dict[str, int]
    regional_distribution: dict[str, int]
    average_amount: float
    median_amount: float
    average_latency_ms: float
    p95_latency_ms: float
    processed_revenue: float
    failed_payment_revenue: float
    scenario_window_stats: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "failure_reason_counts": self.failure_reason_counts,
            "payment_method_distribution": self.payment_method_distribution,
            "bank_distribution": self.bank_distribution,
            "regional_distribution": self.regional_distribution,
            "average_amount": self.average_amount,
            "median_amount": self.median_amount,
            "average_latency_ms": self.average_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "processed_revenue": self.processed_revenue,
            "failed_payment_revenue": self.failed_payment_revenue,
            "scenario_window_stats": self.scenario_window_stats,
        }

    def format_text(self) -> str:
        lines = [
            "=" * 60,
            "FlowShield — Synthetic Dataset Analysis Report",
            "=" * 60,
            f"Total events            : {self.total_events}",
            f"Success rate            : {self.success_rate:.4f}",
            f"Failure rate            : {self.failure_rate:.4f}",
            "",
            "Failure reasons:",
        ]
        for reason, count in sorted(self.failure_reason_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {reason:<22} {count}")

        lines.append("")
        lines.append("Payment method distribution:")
        for k, v in sorted(self.payment_method_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k:<22} {v}")

        lines.append("")
        lines.append("Bank distribution:")
        for k, v in sorted(self.bank_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k:<22} {v}")

        lines.append("")
        lines.append("Regional distribution:")
        for k, v in sorted(self.regional_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k:<22} {v}")

        lines += [
            "",
            f"Average amount (INR)    : {self.average_amount:.2f}",
            f"Median amount (INR)     : {self.median_amount:.2f}",
            f"Average latency (ms)    : {self.average_latency_ms:.2f}",
            f"P95 latency (ms)        : {self.p95_latency_ms:.2f}",
            f"Processed revenue (INR) : {self.processed_revenue:.2f}",
            f"Failed-payment revenue  : {self.failed_payment_revenue:.2f}",
        ]

        if self.scenario_window_stats:
            lines.append("")
            lines.append(
                "Scenario / incident window statistics "
                "(compared against pure normal-traffic baseline, i.e. events "
                "outside ALL incident windows — not the blended overall average, "
                "which is itself dragged down by these same incidents):"
            )
            for w in self.scenario_window_stats:
                lines.append(
                    f"  - [{w['scenario_type']}] {w['description']}\n"
                    f"      window   : {w['start']} -> {w['end']}\n"
                    f"      events   : {w['event_count']}\n"
                    f"      success  : {w['success_rate']:.4f} "
                    f"(normal-traffic baseline: {w['baseline_success_rate']:.4f})\n"
                    f"      avg_lat  : {w['avg_latency_ms']:.2f} ms "
                    f"(normal-traffic baseline: {w['baseline_avg_latency_ms']:.2f} ms)"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def _p95(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(0.95))


def compute_analysis(
    df: pd.DataFrame,
    incident_windows: list[dict[str, Any]] | None = None,
) -> AnalysisReport:
    if df.empty:
        raise ValueError("Cannot analyze an empty dataset")

    total = len(df)
    success_mask = df["status"] == "success"
    failed_mask = df["status"] == "failed"

    success_rate = float(success_mask.mean())
    failure_rate = float(failed_mask.mean())

    failure_reason_counts = (
        df.loc[failed_mask, "failure_reason"].value_counts().to_dict()
    )
    payment_method_distribution = df["payment_method"].value_counts().to_dict()
    bank_distribution = df["bank"].value_counts().to_dict()
    regional_distribution = df["region"].value_counts().to_dict()

    average_amount = float(df["amount"].mean())
    median_amount = float(df["amount"].median())
    average_latency = float(df["latency_ms"].mean())
    p95_latency = _p95(df["latency_ms"])

    processed_revenue = float(df.loc[success_mask, "amount"].sum())
    failed_payment_revenue = float(df.loc[failed_mask, "amount"].sum())

    scenario_stats = []
    if incident_windows:
        # Pure normal-traffic baseline: events outside ALL incident windows.
        # Used as the comparison point for each window below, since the
        # blended overall average is itself dragged down by the incidents.
        any_incident_mask = pd.Series(False, index=df.index)
        window_masks = []
        for w in incident_windows:
            start = pd.to_datetime(w["start"])
            end = pd.to_datetime(w["end"])
            mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
            if w.get("target_payment_method"):
                mask &= df["payment_method"] == w["target_payment_method"]
            if w.get("target_bank"):
                mask &= df["bank"] == w["target_bank"]
            if w.get("target_region"):
                mask &= df["region"] == w["target_region"]
            window_masks.append(mask)
            any_incident_mask |= mask

        normal_df = df.loc[~any_incident_mask]
        baseline_success_rate = (
            float((normal_df["status"] == "success").mean()) if len(normal_df) else 0.0
        )
        baseline_avg_latency = (
            float(normal_df["latency_ms"].mean()) if len(normal_df) else 0.0
        )

        for w, mask in zip(incident_windows, window_masks):
            window_df = df.loc[mask]
            event_count = int(len(window_df))
            window_success_rate = (
                float((window_df["status"] == "success").mean()) if event_count else 0.0
            )
            window_avg_latency = float(window_df["latency_ms"].mean()) if event_count else 0.0

            scenario_stats.append(
                {
                    "scenario_type": w["scenario_type"],
                    "description": w.get("description", ""),
                    "start": w["start"],
                    "end": w["end"],
                    "event_count": event_count,
                    "success_rate": window_success_rate,
                    "avg_latency_ms": window_avg_latency,
                    "baseline_success_rate": baseline_success_rate,
                    "baseline_avg_latency_ms": baseline_avg_latency,
                }
            )

    return AnalysisReport(
        total_events=total,
        success_rate=success_rate,
        failure_rate=failure_rate,
        failure_reason_counts=failure_reason_counts,
        payment_method_distribution=payment_method_distribution,
        bank_distribution=bank_distribution,
        regional_distribution=regional_distribution,
        average_amount=average_amount,
        median_amount=median_amount,
        average_latency_ms=average_latency,
        p95_latency_ms=p95_latency,
        processed_revenue=processed_revenue,
        failed_payment_revenue=failed_payment_revenue,
        scenario_window_stats=scenario_stats,
    )
