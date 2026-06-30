"use client";

import Link from "next/link";
import { listActiveSightings, thumbnailUrl, type ActiveSighting } from "@/lib/api";
import { usePoll } from "@/lib/usePoll";
import { fmtTime } from "@/lib/format";
import { CornerBrackets } from "@/components/CornerBrackets";

type Group = { cameraId: number; cameraName: string; location: string; rows: ActiveSighting[] };

function dedupeByPerson(rows: ActiveSighting[]): ActiveSighting[] {
  // A person whose track fragments can have several open sightings at once.
  // Show one card per identity (the earliest sighting). Not-yet-recognized
  // sightings have no person, so keep each of those.
  const sorted = [...rows].sort((a, b) =>
    (a.startedAt ?? "").localeCompare(b.startedAt ?? ""));
  const seen = new Set<number>();
  const out: ActiveSighting[] = [];
  for (const r of sorted) {
    if (r.person) {
      if (seen.has(r.person.id)) continue;
      seen.add(r.person.id);
    }
    out.push(r);
  }
  return out;
}

function groupByCamera(rows: ActiveSighting[]): Group[] {
  const map = new Map<number, Group>();
  for (const r of rows) {
    const g = map.get(r.cameraId) ?? {
      cameraId: r.cameraId,
      cameraName: r.cameraName ?? `Camera ${r.cameraId}`,
      location: r.location ?? "",
      rows: [],
    };
    g.rows.push(r);
    map.set(r.cameraId, g);
  }
  for (const g of map.values()) g.rows = dedupeByPerson(g.rows);
  return [...map.values()];
}

export default function LivePage() {
  const { data, error } = usePoll(listActiveSightings, 1500);
  const groups = data ? groupByCamera(data) : [];
  // Count unique identities (a person seen on two cameras counts once); each
  // not-yet-recognized sighting counts on its own.
  const visibleCount = data
    ? new Set(data.map((r) => (r.person ? `p${r.person.id}` : `s${r.id}`))).size
    : 0;

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-10 sm:px-10">
      <header className="mb-8 border-b border-edge pb-6">
        <p className="mb-1 text-xs tracking-[0.3em] text-faint">LIVE // CURRENTLY IN VIEW</p>
        <h1 className="font-display text-4xl font-bold tracking-wide text-ink">LIVE FEED</h1>
        <p className="mt-2 text-xs text-faint">
          {data ? `${visibleCount} person(s) currently visible across ${groups.length} camera(s)` : "Connecting…"}
        </p>
      </header>

      {error && (
        <p className="border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          Cannot reach the API. Is the backend running?
        </p>
      )}

      {data && data.length === 0 && (
        <div className="border border-dashed border-edge px-6 py-16 text-center">
          <p className="text-sm uppercase tracking-widest text-faint">No one currently in view</p>
          <p className="mt-2 text-xs text-faint/70">
            Start a camera under <Link href="/cameras" className="text-green">CAMERAS</Link> to begin tracking.
          </p>
        </div>
      )}

      <div className="space-y-8">
        {groups.map((g) => (
          <section key={g.cameraId}>
            <div className="mb-3 flex items-center gap-3">
              <span className="h-1.5 w-1.5 rounded-full bg-green shadow-[0_0_6px_var(--green)] rec-dot" />
              <h2 className="font-display text-lg tracking-widest text-ink">{g.cameraName}</h2>
              {g.location && <span className="text-xs tracking-widest text-faint">/ {g.location}</span>}
              <span className="ml-auto text-xs text-faint">{g.rows.length} tracked</span>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
              {g.rows.map((r) => (
                <Link
                  key={r.id}
                  href={r.person ? `/people/${r.person.id}` : "#"}
                  className="group relative border border-edge bg-surface p-2 transition-colors hover:border-edge-bright"
                >
                  <CornerBrackets color="var(--green)" size={9} />
                  <div className="relative mb-2 aspect-square overflow-hidden border border-edge bg-void">
                    {r.person && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={thumbnailUrl(r.person.id)}
                        alt={r.person.displayName}
                        className="h-full w-full object-cover grayscale-[15%] contrast-125"
                      />
                    )}
                  </div>
                  <p className="truncate text-xs font-medium tracking-wide text-ink">
                    {r.person?.displayName ?? "—"}
                  </p>
                  <p className="text-[10px] tracking-widest text-faint">since {fmtTime(r.startedAt)}</p>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
