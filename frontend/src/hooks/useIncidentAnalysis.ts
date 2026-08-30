import { useCallback, useState } from "react";
import { api, ApiError } from "../api/client";
import type { FlowShieldAnalysisResponse } from "../api/types";

interface UseIncidentAnalysisResult {
  result: FlowShieldAnalysisResponse | null;
  loading: boolean;
  error: string | null;
  simulating: boolean;
  simulateError: string | null;
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
  }, []);

  return { result, loading, error, simulating, simulateError, analyze, simulate, reset };
}
