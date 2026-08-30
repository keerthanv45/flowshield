import type { RecoveryActionType, ReasoningSource, Severity } from "../api/types";

export function SeverityBadge({ severity }: { severity: Severity }) {
  const cls =
    severity === "CRITICAL" ? "badge-critical" : severity === "WARNING" ? "badge-warning" : "badge-info";
  return <span className={`badge ${cls}`}>{severity}</span>;
}

export function ActionBadge({ action }: { action: RecoveryActionType }) {
  return <span className={`action-badge action-${action}`}>{action.replace(/_/g, " ")}</span>;
}

export function SourcePill({ source }: { source: ReasoningSource }) {
  return <span className={`source-pill ${source}`}>{source.toUpperCase()}</span>;
}

export function RiskBadge({ risk }: { risk: "LOW" | "MEDIUM" | "HIGH" }) {
  const cls = risk === "HIGH" ? "badge-critical" : risk === "MEDIUM" ? "badge-warning" : "badge-success";
  return <span className={`badge ${cls}`}>{risk} RISK</span>;
}
