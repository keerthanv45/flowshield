"""
Prepare time-based train/validation/test splits from the SYNTHETIC dataset.

Does NOT train a model — Phase 1 only prepares the data for future ML.
See ml/evaluation/split.py for the leakage-avoidance rationale.

Usage:
    python scripts/prepare_ml_splits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.evaluation.split import SplitRatios, time_based_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ML-ready splits")
    parser.add_argument("--events", type=str, default="data/synthetic/events.csv")
    parser.add_argument("--out", type=str, default="data/synthetic/splits")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"ERROR: {events_path} does not exist. Run scripts/generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(events_path, parse_dates=["timestamp"])
    ratios = SplitRatios(train=args.train_ratio, validation=args.val_ratio, test=args.test_ratio)
    split = time_based_split(df, ratios)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    split.train.to_csv(out_dir / "train.csv", index=False)
    split.validation.to_csv(out_dir / "validation.csv", index=False)
    split.test.to_csv(out_dir / "test.csv", index=False)

    print(f"Split complete: {split.summary()}")
    print(f"Written to {out_dir}/{{train,validation,test}}.csv")


if __name__ == "__main__":
    main()
