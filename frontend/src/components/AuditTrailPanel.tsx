import type { AuditEvent } from "../api/types";
import { EmptyState } from "./StateViews";

const STAGE_LABELS: Record<string, string> = {
  INCIDENT_DETECTED: "Detected",
  RCA_COMPLETED: "RCA",
  REVENUE_RISK_CALCULATED: "Revenue Risk",
  RECOVERY_POLICY_EVALUATED: "Policy",
  GUARDRAILS_CHECKED: "Guardrails",
  RECOVERY_SIMULATED: "Simulation",
  OUTCOME_RECORDED: "Outcome",
};

function statusClass(status: string): string {
  if (status.includes("SIMULATED")) return "simulated";
  if (status === "PASS" || status === "CONFIRMED") return "pass";
  return "";
}

export function AuditTrailPanel({ events }: { events: AuditEvent[] | null }) {
  return (
    <div className="panel fade-in">
      <div className="panel-title">
        <span>Agent Activity — Audit Trail</span>
      </div>

      {!events && (
        <EmptyState label='Click "Simulate Recovery" to generate the audit trail for this incident.' />
      )}

      {events && (
        <>
          <div className="audit-trail">
            {events.map((event, i) => (
              <div className="audit-step" key={event.order}>
                <div className="audit-step-rail">
                  <div className={`audit-step-dot ${statusClass(event.status) === "simulated" ? "warn" : ""}`} />
                  {i < events.length - 1 && <div className="audit-step-line" />}
                </div>
                <div className="audit-step-body">
                  <div className="audit-step-top">
                    <span className="audit-step-name">{STAGE_LABELS[event.event_type] ?? event.event_type}</span>
                    <span className={`audit-step-status ${statusClass(event.status)}`}>{event.status}</span>
                  </div>
                  <div className="audit-step-message">{event.message}</div>
                  {event.value && <div className="audit-step-value">{event.value}</div>}
                </div>
              </div>
            ))}
          </div>

          <div className="audit-legend">
            <span>
              <strong>RCA</strong> = Nemotron reasoning
            </span>
            <span>
              <strong>Policy / Guardrails</strong> = deterministic rules, not AI
            </span>
            <span>
              <strong>Simulation</strong> = SIMULATED, no real payment executed
            </span>
          </div>
        </>
      )}
    </div>
  );
}
