import type {
  ConfigStatus,
  DashboardSummary,
  FlowShieldAnalysisResponse,
  IncidentResponse,
} from "./types";

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // Network failure -- backend unreachable. Never surface raw errors.
    throw new ApiError(0, "FlowShield API is unavailable. Is the backend running?");
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // response wasn't JSON -- keep statusText, never leak raw body/stack traces
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export const api = {
  getSummary: (): Promise<DashboardSummary> => request<DashboardSummary>("/api/v1/summary"),

  listIncidents: (params?: {
    severity?: string;
    incident_type?: string;
    limit?: number;
  }): Promise<IncidentResponse[]> => {
    const query = new URLSearchParams();
    if (params?.severity) query.set("severity", params.severity);
    if (params?.incident_type) query.set("incident_type", params.incident_type);
    if (params?.limit) query.set("limit", String(params.limit));
    const qs = query.toString();
    return request<IncidentResponse[]>(`/api/v1/incidents${qs ? `?${qs}` : ""}`);
  },

  getIncident: (incidentId: string): Promise<IncidentResponse> =>
    request<IncidentResponse>(`/api/v1/incidents/${encodeURIComponent(incidentId)}`),

  analyzeIncident: (incidentId: string): Promise<FlowShieldAnalysisResponse> =>
    request<FlowShieldAnalysisResponse>(
      `/api/v1/incidents/${encodeURIComponent(incidentId)}/analyze`,
      { method: "POST" },
    ),

  simulateIncident: (incidentId: string): Promise<FlowShieldAnalysisResponse> =>
    request<FlowShieldAnalysisResponse>(
      `/api/v1/incidents/${encodeURIComponent(incidentId)}/simulate`,
      { method: "POST" },
    ),

  getConfigStatus: (): Promise<ConfigStatus> => request<ConfigStatus>("/api/v1/config/status"),
};
