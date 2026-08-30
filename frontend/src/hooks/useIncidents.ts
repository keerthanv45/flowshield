import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { IncidentResponse } from "../api/types";

interface UseIncidentsResult {
  incidents: IncidentResponse[];
  loading: boolean;
  error: string | null;
}

export function useIncidents(): UseIncidentsResult {
  const [incidents, setIncidents] = useState<IncidentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .listIncidents()
      .then((data) => {
        if (!cancelled) setIncidents(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load incidents.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { incidents, loading, error };
}

/** Pick a meaningful default incident: first CRITICAL, else first WARNING,
 * else the first incident overall. Mirrors scripts/run_recovery_demo.py's
 * default-selection logic. */
export function pickDefaultIncident(incidents: IncidentResponse[]): IncidentResponse | null {
  if (incidents.length === 0) return null;
  return (
    incidents.find((i) => i.severity === "CRITICAL") ??
    incidents.find((i) => i.severity === "WARNING") ??
    incidents[0]
  );
}
