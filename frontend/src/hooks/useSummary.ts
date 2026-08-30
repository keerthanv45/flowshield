import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { DashboardSummary } from "../api/types";

interface UseSummaryResult {
  summary: DashboardSummary | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

export function useSummary(): UseSummaryResult {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getSummary()
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load summary.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  return { summary, loading, error, reload: () => setReloadKey((k) => k + 1) };
}
