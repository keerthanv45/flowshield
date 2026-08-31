import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { SummaryCards } from "./components/SummaryCards";
import { IncidentListPanel } from "./components/IncidentListPanel";
import { SelectedIncidentPanel } from "./components/SelectedIncidentPanel";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { RecoveryPanel } from "./components/RecoveryPanel";
import { SimulationPanel } from "./components/SimulationPanel";
import { BatchRecoveryPanel } from "./components/BatchRecoveryPanel";
import { AuditTrailPanel } from "./components/AuditTrailPanel";
import { EmptyState } from "./components/StateViews";
import { useSummary } from "./hooks/useSummary";
import { useIncidents, pickDefaultIncident } from "./hooks/useIncidents";
import { useIncidentAnalysis } from "./hooks/useIncidentAnalysis";
import type { IncidentResponse } from "./api/types";

function App() {
  const { summary, loading: summaryLoading, error: summaryError } = useSummary();
  const { incidents, loading: incidentsLoading, error: incidentsError } = useIncidents();
  const {
    result,
    loading: analyzing,
    error: analyzeError,
    simulating,
    simulateError,
    auditEvents,
    analyze,
    simulate,
    reset,
  } = useIncidentAnalysis();

  const [selected, setSelected] = useState<IncidentResponse | null>(null);

  const apiUp =
    summaryLoading || incidentsLoading
      ? null
      : !(summaryError?.includes("unavailable") || incidentsError?.includes("unavailable"));

  useEffect(() => {
    if (!selected && incidents.length > 0) {
      setSelected(pickDefaultIncident(incidents));
    }
  }, [incidents, selected]);

  const handleSelect = (incident: IncidentResponse) => {
    setSelected(incident);
    reset();
  };

  const handleAnalyze = () => {
    if (selected) void analyze(selected.incident_id);
  };

  const handleSimulate = () => {
    if (selected) void simulate(selected.incident_id);
  };

  return (
    <div className="app-shell">
      <Header apiUp={apiUp} />

      <div style={{ marginTop: 22 }}>
        <SummaryCards summary={summary} loading={summaryLoading} error={summaryError} />
      </div>

      <div style={{ marginTop: 20 }}>
        <BatchRecoveryPanel />
      </div>

      <div className="app-grid">
        <div className="stack">
          <IncidentListPanel
            incidents={incidents}
            loading={incidentsLoading}
            error={incidentsError}
            selectedId={selected?.incident_id ?? null}
            onSelect={handleSelect}
          />
          <SelectedIncidentPanel incident={selected} onAnalyze={handleAnalyze} analyzing={analyzing} />
        </div>

        <div className="stack">
          {analyzeError && (
            <div className="panel">
              <EmptyState label={analyzeError} />
            </div>
          )}

          {!result && !analyzeError && (
            <div className="panel">
              <EmptyState label='Select an incident and click "Analyze Incident" to see AI root cause and recovery recommendations.' />
            </div>
          )}

          {result && (
            <>
              <AnalysisPanel rca={result.rca} />
              <RecoveryPanel revenueRisk={result.revenue_risk} decision={result.recovery_decision} />
              <SimulationPanel
                simulation={result.simulation}
                onSimulate={handleSimulate}
                simulating={simulating}
                simulateError={simulateError}
                disabled={!result}
              />
              <AuditTrailPanel events={auditEvents} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
