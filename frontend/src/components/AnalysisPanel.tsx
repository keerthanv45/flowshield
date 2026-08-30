import type { RCAResponse } from "../api/types";
import { SourcePill } from "./Badges";

export function AnalysisPanel({ rca }: { rca: RCAResponse }) {
  return (
    <div className="panel fade-in">
      <div className="panel-title">
        <span>AI Root Cause Analysis</span>
        <SourcePill source={rca.source} />
      </div>

      <div className="root-cause-text">{rca.root_cause}</div>

      <div className="confidence-bar-track">
        <div className="confidence-bar-fill" style={{ width: `${rca.confidence * 100}%` }} />
      </div>
      <div className="empty-hint" style={{ marginTop: 4, marginBottom: 14 }}>
        Confidence: {(rca.confidence * 100).toFixed(0)}%
      </div>

      <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: 0 }}>
        {rca.explanation}
      </p>

      {rca.supporting_evidence.length > 0 && (
        <>
          <div className="section-title" style={{ marginTop: 16 }}>
            Supporting Evidence
          </div>
          <ul className="evidence-list">
            {rca.supporting_evidence.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
