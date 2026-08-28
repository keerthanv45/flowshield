"""
Generate the SYNTHETIC FlowShield payment dataset.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n-events 20000 --seed 42 --out data/synthetic

Outputs:
    data/synthetic/events.csv            - the payment events (SYNTHETIC)
    data/synthetic/incident_windows.json - ground-truth incident schedule (SYNTHETIC)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/generate_data.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.analysis.report import events_to_dataframe
from ml.data_generation.generator import GenerationConfig, PaymentEventGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FlowShield payment data")
    parser.add_argument("--n-events", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/synthetic")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GenerationConfig(n_events=args.n_events, seed=args.seed)
    generator = PaymentEventGenerator(config)

    print(f"Generating {config.n_events} SYNTHETIC payment events (seed={config.seed})...")
    events = generator.generate()

    df = events_to_dataframe(events)
    events_path = out_dir / "events.csv"
    df.to_csv(events_path, index=False)
    print(f"Wrote {len(df)} events to {events_path}")

    windows_path = out_dir / "incident_windows.json"
    windows_payload = {
        "note": (
            "SYNTHETIC ground-truth incident schedule. Not derived from any "
            "real payment gateway. For use in evaluating future anomaly / "
            "incident detection against known injected incidents."
        ),
        "generation_config": {
            "n_events": config.n_events,
            "seed": config.seed,
            "period_start": config.period_start.isoformat(),
            "period_days": config.period_days,
        },
        "windows": [w.to_dict() for w in generator.incident_windows],
    }
    windows_path.write_text(json.dumps(windows_payload, indent=2))
    print(f"Wrote {len(generator.incident_windows)} incident windows to {windows_path}")

    print("Done.")


if __name__ == "__main__":
    main()
