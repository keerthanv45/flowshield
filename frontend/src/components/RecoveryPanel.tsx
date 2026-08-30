import type { RecoveryDecisionResponse, RevenueRiskResponse } from "../api/types";
import { ActionBadge, RiskBadge } from "./Badges";

function formatCurrency(value: number): string {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
}

export function RecoveryPanel({
  revenueRisk,
  decision,
}: {
  revenueRisk: RevenueRiskResponse;
  decision: RecoveryDecisionResponse;
}) {
  return (
    <div className="panel panel-centerpiece fade-in">
      <div className="panel-title">
        <span>Revenue Recovery</span>
      </div>

      <div className="recovery-figures">
        <div className="recovery-figure">
          <div className="recovery-figure-label">Revenue at Risk</div>
          <div className="recovery-figure-value">{formatCurrency(revenueRisk.gross_amount_at_risk)}</div>
        </div>
        <div className="recovery-figure">
          <div className="recovery-figure-label">Recoverable Revenue</div>
          <div className="recovery-figure-value" style={{ color: "var(--success)" }}>
            {formatCurrency(revenueRisk.recoverable_amount)}
          </div>
        </div>
        <div className="recovery-figure">
          <div className="recovery-figure-label">Expected Recovery</div>
          <div className="recovery-figure-value" style={{ color: "var(--accent-strong)" }}>
            {formatCurrency(revenueRisk.expected_recovered_amount)}
          </div>
        </div>
      </div>

      <div className="decision-block">
        <div className="section-title">Recovery Decision</div>
        <div className="decision-action-row">
          <ActionBadge action={decision.recommended_action} />
          <RiskBadge risk={decision.risk_level} />
          <span className="decision-score">score {decision.decision_score.toFixed(1)}</span>
        </div>

        {decision.reasoning.length > 0 && (
          <ul className="reasoning-list">
            {decision.reasoning.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
