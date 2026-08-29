"""
Run the full Phase 2 Payment Health Engine pipeline against the SYNTHETIC
dataset and evaluate detection/classification against the known ground
truth incident schedule.

Usage:
    python scripts/evaluate_phase2.py

Outputs:
    data/synthetic/phase2_incidents.json     - confirmed incidents
    data/synthetic/phase2_health_snapshots.csv - per-window health snapshots
    data/synthetic/phase2_evaluation.json    - evaluation metrics (full + held-out)
    data/synthetic/phase2_report.png         - charts (if matplotlib available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.evaluation.incident_evaluation import evaluate
from ml.health.pipeline import run_health_pipeline


def main() -> None:
    events_path = Path("data/synthetic/events.csv")
    windows_path = Path("data/synthetic/incident_windows.json")

    if not events_path.exists() or not windows_path.exists():
        print("ERROR: run scripts/generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(events_path, parse_dates=["timestamp"])
    incident_windows = json.loads(windows_path.read_text())["windows"]

    print(f"Running Phase 2 pipeline over {len(df)} events...")
    result = run_health_pipeline(df, incident_windows)
    print(
        f"Pipeline complete: {len(result.classifications)} windows classified, "
        f"{len(result.incidents)} confirmed incidents."
    )

    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    incidents_payload = [inc.as_dict() for inc in result.incidents]
    (out_dir / "phase2_incidents.json").write_text(json.dumps(incidents_payload, indent=2))
    print(f"Wrote {len(incidents_payload)} confirmed incidents to phase2_incidents.json")

    snapshot_cols = [
        "window_start", "window_end", "transaction_count", "success_rate",
        "baseline_success_rate", "success_rate_delta", "average_latency_ms",
        "baseline_average_latency_ms", "latency_ratio", "p95_latency_ms",
        "health_score", "anomaly_score", "is_anomaly",
    ]
    result.overall_scored[snapshot_cols].to_csv(out_dir / "phase2_health_snapshots.csv", index=False)
    print(f"Wrote {len(result.overall_scored)} health snapshots to phase2_health_snapshots.csv")

    print("\nEvaluating against ground truth (full dataset)...")
    full_report = evaluate(result.classifications, result.incidents, incident_windows, label="full dataset")
    print(full_report.format_text())

    held_out_start_idx = result.train_end_index
    held_out_classifications = result.classifications[held_out_start_idx:]
    held_out_window_starts = {c.window_start for c in held_out_classifications}
    held_out_incidents = [
        inc for inc in result.incidents if inc.window_start in held_out_window_starts
    ]
    print(
        f"\nEvaluating against ground truth "
        f"(held-out validation+test period only, chronologically after train)..."
    )
    held_out_report = evaluate(
        held_out_classifications, held_out_incidents, incident_windows, label="held-out (val+test)"
    )
    print(held_out_report.format_text())

    evaluation_payload = {
        "full_dataset": full_report.as_dict(),
        "held_out": held_out_report.as_dict(),
        "note": (
            "Baseline and anomaly detector were fit ONLY on the "
            "chronologically-first 70% of the dataset (train split), "
            "excluding windows overlapping any known incident. "
            "'full_dataset' scores the fitted models against every window "
            "(train windows are in-sample for the models, though the "
            "detector/classifier itself never sees ground-truth labels). "
            "'held_out' restricts evaluation to the validation+test period "
            "(chronologically after train), which is genuine out-of-sample "
            "performance."
        ),
    }
    (out_dir / "phase2_evaluation.json").write_text(json.dumps(evaluation_payload, indent=2))
    print("\nWrote evaluation metrics to phase2_evaluation.json")

    try:
        from ml.evaluation.phase2_report import generate_report_charts

        chart_path = out_dir / "phase2_report.png"
        generate_report_charts(result, incident_windows, chart_path)
        print(f"Wrote report charts to {chart_path}")
    except Exception as exc:  # pragma: no cover - charting is best-effort
        print(f"(Chart generation skipped: {exc})")


if __name__ == "__main__":
    main()
