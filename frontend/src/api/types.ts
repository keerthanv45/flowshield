// Mirrors backend/app/schemas/api.py, backend/app/services/reasoning/schemas.py,
// and backend/app/services/recovery/schemas.py. Field names/shapes match the
// actual FastAPI JSON responses exactly -- no invented fields.

export type Severity = "INFO" | "WARNING" | "CRITICAL";

export type IncidentType =
  | "NORMAL"
  | "BANK_RAIL_DEGRADATION"
  | "REGIONAL_DEGRADATION"
  | "LATENCY_SPIKE"
  | "MERCHANT_SYSTEM_DEGRADATION"
  | "ISOLATED_FAILURES";

export type ReasoningSource = "mock" | "nemotron" | "fallback";

export type RecoveryActionType =
  | "RETRY"
  | "ALTERNATE_PAYMENT_METHOD"
  | "ROUTE_ALTERNATE_RAIL"
  | "WAIT_AND_MONITOR"
  | "NO_ACTION";

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface AffectedDimension {
  bank?: string;
  payment_method?: string;
  region?: string;
}

export interface IncidentResponse {
  incident_id: string;
  detected_at: string;
  window_start: string;
  window_end: string;
  severity: Severity;
  incident_type: IncidentType;
  anomaly_score: number;
  health_score: number;
  affected_dimensions: AffectedDimension[];
  signals: string[];
  evidence: string[];
  status: string;
  n_windows: number;
}

export interface RCAResponse {
  root_cause: string;
  confidence: number;
  explanation: string;
  supporting_evidence: string[];
  affected_scope: AffectedDimension[];
  recommended_actions: string[];
  source: ReasoningSource;
}

export interface FailureBreakdownEntry {
  failure_reason: string;
  count: number;
  amount: number;
  assumed_recovery_rate: number;
}

export interface RevenueRiskResponse {
  incident_id: string;
  transactions_at_risk: number;
  gross_amount_at_risk: number;
  recoverable_transactions: number;
  recoverable_amount: number;
  expected_recovered_amount: number;
  failure_breakdown: FailureBreakdownEntry[];
}

export interface RecoveryCandidate {
  action: RecoveryActionType;
  reason: string;
  eligible_failure_types: string[];
  estimated_recovery: number;
  risk: RiskLevel;
  priority: number;
}

export interface RecoveryDecisionResponse {
  incident_id: string;
  recommended_action: RecoveryActionType;
  decision_score: number;
  expected_recovery_amount: number;
  risk_level: RiskLevel;
  reasoning: string[];
  candidates_considered: RecoveryCandidate[];
  policy_notes: string[];
}

export interface SimulationResponse {
  action_id: string;
  incident_id: string;
  action: RecoveryActionType;
  eligible_transactions: number;
  simulated_successes: number;
  simulated_recovered_amount: number;
  seed: number;
  status: string;
}

export interface FlowShieldAnalysisResponse {
  incident: IncidentResponse;
  rca: RCAResponse;
  revenue_risk: RevenueRiskResponse;
  recovery_decision: RecoveryDecisionResponse;
  simulation: SimulationResponse | null;
}

export interface DashboardSummary {
  total_transactions: number;
  overall_success_rate: number;
  current_health_score: number;
  confirmed_incident_count: number;
  active_incident_count: number;
  total_revenue_at_risk: number;
  total_recoverable_revenue: number;
  recommended_action_counts: Record<string, number>;
}

export interface ConfigStatus {
  provider: string;
  model: string | null;
  nemotron_configured: boolean;
}
