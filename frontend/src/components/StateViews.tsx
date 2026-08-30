export function LoadingState({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="state-view">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ label }: { label: string }) {
  return (
    <div className="state-view">
      <span className="empty-hint">{label}</span>
    </div>
  );
}

/** Never renders raw error objects/stack traces -- message only. */
export function ErrorState({ message }: { message: string }) {
  return (
    <div className="state-view error">
      <span>⚠ {message}</span>
    </div>
  );
}
