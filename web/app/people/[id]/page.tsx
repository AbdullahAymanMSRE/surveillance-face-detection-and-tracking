"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getPerson, thumbnailUrl, updateLabel } from "@/lib/api";
import { usePoll } from "@/lib/usePoll";
import { fmtDate, fmtSpan, fmtRelative } from "@/lib/format";
import { CornerBrackets } from "@/components/CornerBrackets";

export default function PersonDetailPage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const fetcher = useCallback(() => getPerson(id), [id]);
  const { data, error } = usePoll(fetcher, 2000);

  const [label, setLabel] = useState("");
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (data && !editing) setLabel(data.label ?? "");
  }, [data, editing]);

  const saveLabel = async () => {
    await updateLabel(id, label.trim() || null);
    setEditing(false);
  };

  if (error) {
    return (
      <main className="mx-auto w-full max-w-4xl px-6 py-10">
        <p className="border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          Could not load this person.
        </p>
        <Link href="/people" className="mt-4 inline-block text-xs text-green">← Back to people</Link>
      </main>
    );
  }

  if (!data) {
    return <main className="mx-auto w-full max-w-4xl px-6 py-10 text-sm text-faint">Loading…</main>;
  }

  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-10 sm:px-10">
      <Link href="/people" className="mb-6 inline-block text-xs tracking-widest text-faint hover:text-ink">
        ← PEOPLE
      </Link>

      <div className="mb-8 flex flex-wrap gap-6 border-b border-edge pb-6">
        <div className="relative h-40 w-40 shrink-0 border border-edge bg-void">
          <CornerBrackets color={data.active ? "var(--green)" : "var(--edge-bright)"} size={12} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={thumbnailUrl(data.id)}
            alt={data.displayName}
            className="h-full w-full object-cover grayscale-[15%] contrast-125"
          />
        </div>
        <div className="flex-1">
          <p className="text-xs tracking-[0.3em] text-faint">
            ID-{String(data.id).padStart(3, "0")} {data.active && "· IN VIEW"}
          </p>
          <h1 className="font-display text-3xl font-bold tracking-wide text-ink">{data.displayName}</h1>
          <p className="mt-1 text-xs text-faint">
            {data.visits} visit{data.visits === 1 ? "" : "s"} · last seen {fmtRelative(data.lastSeen)}
          </p>

          <div className="mt-4 flex items-center gap-2">
            <input
              value={label}
              onChange={(e) => {
                setLabel(e.target.value);
                setEditing(true);
              }}
              placeholder="add a label (optional)"
              className="w-56 border border-edge bg-surface px-3 py-1.5 text-sm text-ink outline-none focus:border-edge-bright"
            />
            <button
              onClick={saveLabel}
              disabled={!editing}
              className="border border-green/60 px-3 py-1.5 text-xs uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void disabled:opacity-30"
            >
              Save
            </button>
          </div>
        </div>
      </div>

      <h2 className="mb-3 font-display text-lg tracking-widest text-ink">APPEARANCE TIMELINE</h2>
      {data.timeline.length === 0 ? (
        <p className="text-sm text-faint">No recorded visits yet.</p>
      ) : (
        <ul className="space-y-px">
          {data.timeline.map((s) => (
            <li
              key={s.id}
              className="flex flex-wrap items-center gap-x-4 gap-y-1 border-l-2 bg-surface px-4 py-3"
              style={{ borderColor: s.active ? "var(--green)" : "var(--edge)" }}
            >
              <span className="font-display tracking-widest text-ink">{s.cameraName ?? `Camera ${s.cameraId}`}</span>
              {s.location && <span className="text-xs text-faint">/ {s.location}</span>}
              <span className="ml-auto text-xs tracking-widest text-faint">{fmtDate(s.startedAt)}</span>
              <span className={`text-xs tabular-nums ${s.active ? "text-green" : "text-ink"}`}>
                {fmtSpan(s.startedAt, s.endedAt)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
