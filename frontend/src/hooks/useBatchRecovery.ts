import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { BatchEvaluationResult } from "../api/types";

interface UseBatchRecoveryResult {
  data: BatchEvaluationResult | null;
  loading: boolean;
  error: string | null;
}

export function useBatchRecovery(): UseBatchRecoveryResult {
  const [data, setData] = useState<BatchEvaluationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getRecoveryEvaluation()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load batch recovery evaluation.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}
