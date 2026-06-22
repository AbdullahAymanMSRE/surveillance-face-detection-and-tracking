"use client";

import Link from "next/link";
import { listPeople, thumbnailUrl } from "@/lib/api";
import { usePoll } from "@/lib/usePoll";
import { fmtRelative } from "@/lib/format";
import { CornerBrackets } from "@/components/CornerBrackets";

export default function PeoplePage() {
  const { data, error } = usePoll(listPeople, 2000);

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-10 sm:px-10">
      <header className="mb-8 border-b border-edge pb-6">
        <p className="mb-1 text-xs tracking-[0.3em] text-faint">REGISTRY // ANONYMOUS IDENTITIES</p>
        <h1 className="font-display text-4xl font-bold tracking-wide text-ink">PEOPLE</h1>
        <p className="mt-2 text-xs text-faint">
          {data ? `${data.length} identity(ies) discovered` : "Loading…"}
        </p>
      </header>

      {error && (
        <p className="border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          Cannot reach the API. Is the backend running?
        </p>
      )}

      {data && data.length === 0 && (
        <div className="border border-dashed border-edge px-6 py-16 text-center">
          <p className="text-sm uppercase tracking-widest text-faint">No identities yet</p>
          <p className="mt-2 text-xs text-faint/70">
            People are discovered automatically as cameras see faces.
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 md:grid-cols-4">
        {data?.map((person) => (
          <Link
            key={person.id}
            href={`/people/${person.id}`}
            className="group relative border border-edge bg-surface p-3 transition-colors hover:border-edge-bright"
          >
            <CornerBrackets color={person.active ? "var(--green)" : "var(--edge-bright)"} size={10} />
            <div className="relative mb-3 aspect-square overflow-hidden border border-edge bg-void">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={thumbnailUrl(person.id)}
                alt={person.displayName}
                className="h-full w-full object-cover grayscale-[15%] contrast-125"
              />
              {person.active && (
                <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-green shadow-[0_0_6px_var(--green)] rec-dot" />
              )}
            </div>
            <p className="truncate text-sm font-medium tracking-wide text-ink">{person.displayName}</p>
            <p className="text-[11px] tracking-widest text-faint">
              {person.visits} visit{person.visits === 1 ? "" : "s"} · {fmtRelative(person.lastSeen)}
            </p>
          </Link>
        ))}
      </div>
    </main>
  );
}
