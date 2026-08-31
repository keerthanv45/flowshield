"""
Phase 6 CLI: prints a concise business-impact report for batch recovery
evaluation across the whole synthetic dataset.

SIMULATED ONLY -- no real payment gateway is called.

Usage:
    python scripts/evaluate_batch_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.recovery.batch_evaluation import run_batch_evaluation


def _rupees(amount: float) -> str:
    return f"Rs.{amount:,.2f}"


def main() -> None:
    events_path = Path("data/synthetic/events.csv")
    if not events_path.exists():
        raise SystemExit("data/synthetic/events.csv not found -- run scripts/generate_data.py first.")

    df = pd.read_csv(events_path, parse_dates=["timestamp"])
    result = run_batch_evaluation(df)
    g = result.guardrails

    print("FLOW SHIELD BATCH RECOVERY")
    print("\u2500" * 40)
    print(f"Transactions analyzed: {result.total_transactions:,}")
    print(f"Failed transactions: {result.failed_transactions:,}")
    print(f"Revenue at risk: {_rupees(result.revenue_at_risk)}")
    print()
    print(f"Eligible for recovery: {result.recoverable_transactions:,}")
    print(f"Expected recovery: {_rupees(result.expected_recovery_amount)}")
    print()
    print("SIMULATED OUTCOME")
    print(f"Actions selected (retry/route): {result.actions_selected:,}")
    print(f"Recovered transactions: {result.simulated_recovered_transactions:,}")
    print(f"Recovered revenue: {_rupees(result.simulated_recovered_amount)}")
    print(f"Recovery rate (of attempted): {result.recovery_rate:.1%}")
    print(f"Revenue recovery rate (of total at risk): {result.revenue_recovery_rate:.1%}")
    print()
    print("GUARDRAILS")
    print(f"Hard declines excluded: {g.hard_declines_excluded_count:,} ({_rupees(g.hard_declines_excluded_amount)})")
    print(f"Auth failures excluded: {g.auth_failures_excluded_count:,} ({_rupees(g.auth_failures_excluded_amount)})")
    print(
        f"Unsupported failures excluded: {g.unsupported_failures_excluded_count:,} "
        f"({_rupees(g.unsupported_failures_excluded_amount)})"
    )
    print(
        f"Insufficient-funds deferred (WAIT_AND_MONITOR): {g.insufficient_funds_deferred_count:,} "
        f"({_rupees(g.insufficient_funds_deferred_amount)})"
    )
    print()
    print(f"Status: {result.status} (seed={result.seed})")
    print("SIMULATED \u2014 NO REAL PAYMENT EXECUTED")


if __name__ == "__main__":
    main()
