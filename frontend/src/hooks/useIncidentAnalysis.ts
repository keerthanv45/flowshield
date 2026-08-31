import { useCallback, useState } from "react";
import { api, ApiError } from "../api/client";
import type { AuditEvent, FlowShieldAnalysisResponse } from "../api/types";

interface UseIncidentAnalysisResult {
  result: FlowShieldAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  simulating: boolean;
  simulateError: string | null;
  auditEvents: AuditEvent[] | null;
  analyze: (incidentId: string) => Promise<void>;
  simulate: (incidentId: string) => Promise<void>;
  reset: () => void;
}

export function useIncidentAnalysis(): UseIncidentAnalysisResult {
  const [result, setResult] = useState<FlowShieldAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [simulateError, setSimulateError] = useState<string | null>(null);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[] | null>(null);

  const analyze = useCallback(async (incidentId: string) => {
    setLoading(true);
    setError(null);
    setSimulateError(null);
    try {
      const data = await api.analyzeIncident(incidentId);
      setResult(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Analysis failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const simulate = useCallback(async (incidentId: string) => {
    setSimulating(true);
    setSimulateError(null);
    try {
      const data = await api.simulateIncident(incidentId);
      setResult(data);
      // Audit trail reuses the same analyze+simulate pipeline server-side;
      // fetched alongside simulate so the trail reflects what just ran.
      const audit = await api.getAuditTrail(incidentId);
      setAuditEvents(audit.events);
    } catch (err) {
      setSimulateError(err instanceof ApiError ? err.message : "Simulation failed.");
    } finally {
      setSimulating(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setSimulateError(null);
    setAuditEvents(null);
  }, []);

  return { result, loading, error, simulating, simulateError, auditEvents, analyze, simulate, reset };
}
