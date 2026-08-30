import type { IncidentResponse } from "../api/types";
import { SeverityBadge } from "./Badges";

function formatScope(incident: IncidentResponse): string {
  if (incident.affected_dimensions.length === 0) return "System-wide";
  return incident.affected_dimensions
    .map((d) => Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(", "))
    .join(" · ");
}

export function SelectedIncidentPanel({
  incident,
  onAnalyze,
  analyzing,
}: {
  incident: IncidentResponse | null;
  onAnalyze: () => void;
  analyzing: boolean;
}) {
  if (!incident) {
    return (
      <div className="panel">
        <div className="panel-title">Selected Incident</div>
        <div className="empty-hint">Select an incident to view details.</div>
      </div>
    );
  }

  return (
    <div className="panel fade-in">
      <div className="panel-title">
        <span>Selected Incident</span>
        <SeverityBadge severity={incident.severity} />
      </div>

      <div className="kv-row">
        <span className="kv-key">Type</span>
        <span className="kv-value">{incident.incident_type.replace(/_/g, " ")}</span>
      </div>
      <div className="kv-row">
        <span className="kv-key">Window</span>
        <span className="kv-value">
          {new Date(incident.window_start).toLocaleString()} –{" "}
          {new Date(incident.window_end).toLocaleTimeString()}
        </span>
      </div>
      <div className="kv-row">
        <span className="kv-key">Affected Scope</span>
        <span className="kv-value">{formatScope(incident)}</span>
      </div>
      <div className="kv-row">
        <span className="kv-key">Anomaly Score</span>
        <span className="kv-value">{incident.anomaly_score.toFixed(3)}</span>
      </div>
      <div className="kv-row">
        <span className="kv-key">Health Score</span>
        <span className="kv-value">{incident.health_score.toFixed(1)}</span>
      </div>

      <div style={{ marginTop: 16 }}>
        <button className="analyze-button" onClick={onAnalyze} disabled={analyzing}>
          {analyzing ? "Analyzing..." : "Analyze Incident"}
        </button>
      </div>
    </div>
  );
}
