import type { DashboardSummary } from "../api/types";
import { LoadingState, ErrorState } from "./StateViews";

function formatCurrency(value: number): string {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function healthClass(score: number): string {
  if (score >= 80) return "success";
  if (score >= 50) return "accent";
  return "critical";
}

export function SummaryCards({
  summary,
  loading,
  error,
}: {
  summary: DashboardSummary | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading) return <LoadingState label="Loading summary..." />;
  if (error) return <ErrorState message={error} />;
  if (!summary) return null;

  return (
    <div className="summary-grid fade-in">
      <div className="summary-card">
        <div className="summary-label">Health Score</div>
        <div className={`summary-value ${healthClass(summary.current_health_score)}`}>
          {summary.current_health_score.toFixed(1)}
        </div>
        <div className="summary-sub">
          {summary.overall_success_rate >= 0
            ? `${(summary.overall_success_rate * 100).toFixed(1)}% success rate`
            : ""}
        </div>
      </div>

      <div className="summary-card">
        <div className="summary-label">Revenue at Risk</div>
        <div className="summary-value critical">{formatCurrency(summary.total_revenue_at_risk)}</div>
        <div className="summary-sub">{summary.total_transactions.toLocaleString()} total transactions</div>
      </div>

      <div className="summary-card">
        <div className="summary-label">Recoverable Revenue</div>
        <div className="summary-value success">{formatCurrency(summary.total_recoverable_revenue)}</div>
        <div className="summary-sub">across confirmed incidents</div>
      </div>

      <div className="summary-card">
        <div className="summary-label">Active Incidents</div>
        <div className="summary-value accent">{summary.active_incident_count}</div>
        <div className="summary-sub">{summary.confirmed_incident_count} confirmed total</div>
      </div>
    </div>
  );
}
