import type { SimulationResponse } from "../api/types";
import { ErrorState } from "./StateViews";

function formatCurrency(value: number): string {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

export function SimulationPanel({
  simulation,
  onSimulate,
  simulating,
  simulateError,
  disabled,
}: {
  simulation: SimulationResponse | null;
  onSimulate: () => void;
  simulating: boolean;
  simulateError: string | null;
  disabled: boolean;
}) {
  return (
    <div className="panel fade-in">
      <div className="panel-title">
        <span>Simulate Recovery</span>
      </div>

      <button className="simulate-button" onClick={onSimulate} disabled={disabled || simulating}>
        {simulating ? "Simulating..." : "Simulate Recovery"}
      </button>

      {simulateError && (
        <div style={{ marginTop: 12 }}>
          <ErrorState message={simulateError} />
        </div>
      )}

      {simulation && (
        <div className="fade-in" style={{ marginTop: 18 }}>
          <div className="simulated-banner">⚠ SIMULATED — NO REAL PAYMENT WAS EXECUTED</div>

          <div className="simulation-grid">
            <div className="kv-row" style={{ flexDirection: "column", gap: 2 }}>
              <span className="kv-key">Attempted Transactions</span>
              <span className="kv-value" style={{ textAlign: "left" }}>
                {simulation.eligible_transactions}
              </span>
            </div>
            <div className="kv-row" style={{ flexDirection: "column", gap: 2 }}>
              <span className="kv-key">Recovered Transactions</span>
              <span className="kv-value" style={{ textAlign: "left", color: "var(--success)" }}>
                {simulation.simulated_successes}
              </span>
            </div>
            <div className="kv-row" style={{ flexDirection: "column", gap: 2 }}>
              <span className="kv-key">Recovered Amount</span>
              <span className="kv-value" style={{ textAlign: "left", color: "var(--success)" }}>
                {formatCurrency(simulation.simulated_recovered_amount)}
              </span>
            </div>
            <div className="kv-row" style={{ flexDirection: "column", gap: 2 }}>
              <span className="kv-key">Recovery Rate</span>
              <span className="kv-value" style={{ textAlign: "left" }}>
                {simulation.eligible_transactions > 0
                  ? `${((simulation.simulated_successes / simulation.eligible_transactions) * 100).toFixed(1)}%`
                  : "—"}
              </span>
            </div>
          </div>

          <div className="empty-hint" style={{ marginTop: 12 }}>
            Status: {simulation.status} · Seed: {simulation.seed}
          </div>
        </div>
      )}
    </div>
  );
}
