import type { IncidentResponse } from "../api/types";
import { SeverityBadge } from "./Badges";
import { EmptyState, ErrorState, LoadingState } from "./StateViews";

function formatScope(incident: IncidentResponse): string {
  if (incident.affected_dimensions.length === 0) return "System-wide";
  return incident.affected_dimensions
    .map((d) => Object.entries(d).map(([, v]) => v).join(" + "))
    .slice(0, 2)
    .join(", ");
}

function formatWindow(incident: IncidentResponse): string {
  const start = new Date(incident.window_start);
  return start.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function IncidentListPanel({
  incidents,
  loading,
  error,
  selectedId,
  onSelect,
}: {
  incidents: IncidentResponse[];
  loading: boolean;
  error: string | null;
  selectedId: string | null;
  onSelect: (incident: IncidentResponse) => void;
}) {
  return (
    <div className="panel">
      <div className="panel-title">
        <span>Confirmed Incidents</span>
        {!loading && !error && <span>{incidents.length}</span>}
      </div>

      {loading && <LoadingState label="Loading incidents..." />}
      {error && <ErrorState message={error} />}
      {!loading && !error && incidents.length === 0 && (
        <EmptyState label="No confirmed incidents found." />
      )}

      {!loading && !error && incidents.length > 0 && (
        <div className="incident-list">
          {incidents.map((incident) => (
            <button
              key={incident.incident_id}
              className={`incident-row ${incident.incident_id === selectedId ? "selected" : ""}`}
              onClick={() => onSelect(incident)}
            >
              <div className="incident-row-top">
                <span className="incident-type">{incident.incident_type.replace(/_/g, " ")}</span>
                <SeverityBadge severity={incident.severity} />
              </div>
              <div className="incident-meta">
                {formatWindow(incident)} · {formatScope(incident)} · {incident.status}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
