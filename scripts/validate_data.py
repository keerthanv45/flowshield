"""
Validate the SYNTHETIC FlowShield payment dataset.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --path data/synthetic/events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic FlowShield payment data")
    parser.add_argument("--path", type=str, default="data/synthetic/events.csv")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: {path} does not exist. Run scripts/generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    # NaN failure_reason becomes NaN (float) after CSV round-trip; normalize to None.
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict(orient="records")

    print(f"Validating {len(rows)} rows from {path} ...")
    report = validate_dataset(rows)
    print(report.summary())

    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
