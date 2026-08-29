"""
Phase 2 visual report: renders the health/anomaly/incident picture over
time so a human can sanity-check whether the detector is doing something
sensible. No frontend — a single static PNG with several stacked panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display available
import matplotlib.pyplot as plt
import pandas as pd

from ml.health.incidents import IncidentType

INCIDENT_COLORS = {
    IncidentType.BANK_RAIL_DEGRADATION: "#e15759",
    IncidentType.REGIONAL_DEGRADATION: "#f28e2b",
    IncidentType.LATENCY_SPIKE: "#af7aa1",
    IncidentType.MERCHANT_SYSTEM_DEGRADATION: "#76b7b2",
    IncidentType.ISOLATED_FAILURES: "#bab0ac",
}


def generate_report_charts(pipeline_result, incident_windows: list[dict], out_path: Path) -> None:
    df = pipeline_result.overall_scored.sort_values("window_start")

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    def shade_ground_truth(ax):
        for w in incident_windows:
            ax.axvspan(
                pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                color="gray", alpha=0.08, zorder=0,
            )

    def shade_confirmed_incidents(ax):
        for inc in pipeline_result.incidents:
            color = INCIDENT_COLORS.get(inc.incident_type, "#4e79a7")
            ax.axvspan(inc.window_start, inc.window_end, color=color, alpha=0.25, zorder=0)

    # Panel 1: health score over time
    ax = axes[0]
    shade_ground_truth(ax)
    shade_confirmed_incidents(ax)
    ax.plot(df["window_start"], df["health_score"], color="#1f77b4", linewidth=0.8)
    ax.axhline(80, color="green", linestyle="--", linewidth=0.6, label="HEALTHY threshold")
    ax.axhline(50, color="orange", linestyle="--", linewidth=0.6, label="DEGRADED threshold")
    ax.set_ylabel("Health score")
    ax.set_title("Payment Health Score over time (gray = ground-truth incident window)")
    ax.legend(loc="lower left", fontsize=8)

    # Panel 2: success rate vs baseline
    ax = axes[1]
    shade_ground_truth(ax)
    shade_confirmed_incidents(ax)
    ax.plot(df["window_start"], df["success_rate"], label="observed", color="#2ca02c", linewidth=0.8)
    ax.plot(
        df["window_start"], df["baseline_success_rate"], label="baseline (expected)",
        color="#d62728", linewidth=0.8, linestyle="--",
    )
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs. baseline")
    ax.legend(loc="lower left", fontsize=8)

    # Panel 3: latency vs baseline
    ax = axes[2]
    shade_ground_truth(ax)
    shade_confirmed_incidents(ax)
    ax.plot(df["window_start"], df["average_latency_ms"], label="observed", color="#9467bd", linewidth=0.8)
    ax.plot(
        df["window_start"], df["baseline_average_latency_ms"], label="baseline (expected)",
        color="#d62728", linewidth=0.8, linestyle="--",
    )
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Latency vs. baseline")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 4: anomaly score + detected incident windows
    ax = axes[3]
    shade_ground_truth(ax)
    shade_confirmed_incidents(ax)
    ax.plot(df["window_start"], df["anomaly_score"], color="#8c564b", linewidth=0.8)
    ax.set_ylabel("Anomaly score")
    ax.set_xlabel("Time")
    ax.set_title(
        "Anomaly score over time (colored bands = CONFIRMED incidents by type; "
        "gray = ground truth)"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
