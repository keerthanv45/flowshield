"""
Run basic descriptive analysis over the SYNTHETIC FlowShield dataset.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --events data/synthetic/events.csv \
        --windows data/synthetic/incident_windows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.analysis.report import compute_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze synthetic FlowShield payment data")
    parser.add_argument("--events", type=str, default="data/synthetic/events.csv")
    parser.add_argument("--windows", type=str, default="data/synthetic/incident_windows.json")
    parser.add_argument("--out", type=str, default="data/synthetic/analysis_report.json")
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"ERROR: {events_path} does not exist. Run scripts/generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(events_path, parse_dates=["timestamp"])

    incident_windows = None
    windows_path = Path(args.windows)
    if windows_path.exists():
        payload = json.loads(windows_path.read_text())
        incident_windows = payload.get("windows")

    report = compute_analysis(df, incident_windows=incident_windows)
    print(report.format_text())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.as_dict(), indent=2, default=str))
    print(f"\nSaved JSON report to {out_path}")


if __name__ == "__main__":
    main()
