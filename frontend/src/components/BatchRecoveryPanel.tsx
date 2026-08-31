import { useBatchRecovery } from "../hooks/useBatchRecovery";
import { LoadingState, ErrorState } from "./StateViews";

function formatCurrency(value: number): string {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

export function BatchRecoveryPanel() {
  const { data, loading, error } = useBatchRecovery();

  return (
    <div className="panel fade-in">
      <div className="panel-title">
        <span>Batch Recovery Performance</span>
        <span className="empty-hint">SIMULATED, whole dataset</span>
      </div>

      {loading && <LoadingState label="Evaluating batch recovery..." />}
      {error && <ErrorState message={error} />}

      {data && (
        <>
          <div className="recovery-figures">
            <div className="recovery-figure">
              <div className="recovery-figure-label">Revenue at Risk</div>
              <div className="recovery-figure-value">{formatCurrency(data.revenue_at_risk)}</div>
            </div>
            <div className="recovery-figure">
              <div className="recovery-figure-label">Expected Recovery</div>
              <div className="recovery-figure-value" style={{ color: "var(--accent-strong)" }}>
                {formatCurrency(data.expected_recovery_amount)}
              </div>
            </div>
            <div className="recovery-figure">
              <div className="recovery-figure-label">Simulated Recovered</div>
              <div className="recovery-figure-value" style={{ color: "var(--success)" }}>
                {formatCurrency(data.simulated_recovered_amount)}
              </div>
            </div>
          </div>

          <div className="kv-row">
            <span className="kv-key">Recovery Rate (of attempted)</span>
            <span className="kv-value">{(data.recovery_rate * 100).toFixed(1)}%</span>
          </div>
          <div className="kv-row">
            <span className="kv-key">Transactions Recovered</span>
            <span className="kv-value">
              {data.simulated_recovered_transactions.toLocaleString()} / {data.simulated_attempts.toLocaleString()}
            </span>
          </div>

          <div className="section-title" style={{ marginTop: 16 }}>
            Guardrails
          </div>
          <div className="kv-row">
            <span className="kv-key">Hard declines excluded</span>
            <span className="kv-value">{data.guardrails.hard_declines_excluded_count.toLocaleString()}</span>
          </div>
          <div className="kv-row">
            <span className="kv-key">Auth failures excluded</span>
            <span className="kv-value">{data.guardrails.auth_failures_excluded_count.toLocaleString()}</span>
          </div>
          <div className="kv-row">
            <span className="kv-key">Unsupported excluded</span>
            <span className="kv-value">{data.guardrails.unsupported_failures_excluded_count.toLocaleString()}</span>
          </div>

          <div className="empty-hint" style={{ marginTop: 10 }}>
            SIMULATED — NO REAL PAYMENT WAS EXECUTED
          </div>
        </>
      )}
    </div>
  );
}
