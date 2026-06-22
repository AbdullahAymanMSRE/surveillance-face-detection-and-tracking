export function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString([], { month: "short", day: "2-digit" });
}

/** Compact "X ago" for a recent timestamp. */
export function fmtRelative(iso: string | null): string {
  if (!iso) return "never";
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

/** Visit span like "09:14 – 09:16" or "09:14 – now". */
export function fmtSpan(startIso: string | null, endIso: string | null): string {
  return `${fmtTime(startIso)} – ${endIso ? fmtTime(endIso) : "now"}`;
}
