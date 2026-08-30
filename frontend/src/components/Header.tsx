export function Header({ apiUp }: { apiUp: boolean | null }) {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo">FS</div>
        <div>
          <div className="header-title">FlowShield</div>
          <div className="header-subtitle">Autonomous Revenue Protection</div>
        </div>
      </div>
      <div className="header-status">
        <span className={`status-dot ${apiUp === false ? "down" : ""}`} />
        {apiUp === null ? "Connecting..." : apiUp ? "API Online" : "API Unavailable"}
      </div>
    </header>
  );
}
