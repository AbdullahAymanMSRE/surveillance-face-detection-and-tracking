"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import {
  NoFaceDetectedError,
  search,
  thumbnailUrl,
  type SearchResult,
} from "@/lib/api";
import { fmtDate, fmtSpan, fmtRelative } from "@/lib/format";
import { CornerBrackets } from "@/components/CornerBrackets";

type State =
  | { kind: "idle" }
  | { kind: "searching" }
  | { kind: "result"; result: SearchResult }
  | { kind: "noface" }
  | { kind: "error" };

export default function SearchPage() {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const onPick = async (file: File | undefined) => {
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    setState({ kind: "searching" });
    try {
      const result = await search(file);
      setState({ kind: "result", result });
    } catch (e) {
      setState(e instanceof NoFaceDetectedError ? { kind: "noface" } : { kind: "error" });
    }
  };

  const match = state.kind === "result" ? state.result.match : null;

  return (
    <main className="mx-auto w-full max-w-3xl px-6 py-10 sm:px-10">
      <header className="mb-8 border-b border-edge pb-6">
        <p className="mb-1 text-xs tracking-[0.3em] text-faint">LOOKUP // READ-ONLY</p>
        <h1 className="font-display text-4xl font-bold tracking-wide text-ink">SEARCH BY IMAGE</h1>
        <p className="mt-2 text-xs text-faint">
          Upload a face to check whether this person is on record. Nothing is saved.
        </p>
      </header>

      <div className="mb-8 flex flex-wrap items-center gap-4">
        <button
          onClick={() => fileRef.current?.click()}
          className="border border-green/60 px-5 py-2.5 text-sm uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void"
        >
          Choose image
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => onPick(e.target.files?.[0])}
        />
        {previewUrl && (
          <div className="relative h-20 w-20 border border-edge bg-void">
            <CornerBrackets color="var(--amber)" size={9} />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={previewUrl} alt="query" className="h-full w-full object-cover" />
          </div>
        )}
      </div>

      {state.kind === "searching" && <p className="text-sm text-faint">Searching…</p>}
      {state.kind === "noface" && (
        <p className="border border-amber/50 bg-amber-dim px-4 py-3 text-sm text-amber">
          No face detected in that image.
        </p>
      )}
      {state.kind === "error" && (
        <p className="border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          Search failed. Is the backend running?
        </p>
      )}

      {state.kind === "result" && !match && (
        <div className="border border-dashed border-edge px-6 py-12 text-center">
          <p className="text-sm uppercase tracking-widest text-faint">No record of this person</p>
          <p className="mt-2 text-xs text-faint/70">
            Best similarity {state.result.score.toFixed(2)} (below match threshold).
          </p>
        </div>
      )}

      {match && (
        <div className="border border-green/40 bg-surface p-5">
          <p className="mb-4 text-xs uppercase tracking-widest text-green">
            Match found · score {state.kind === "result" ? state.result.score.toFixed(2) : ""}
          </p>
          <div className="flex flex-wrap gap-5">
            <Link href={`/people/${match.id}`} className="relative h-28 w-28 shrink-0 border border-edge bg-void">
              <CornerBrackets color="var(--green)" size={10} />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={thumbnailUrl(match.id)}
                alt={match.displayName}
                className="h-full w-full object-cover grayscale-[15%] contrast-125"
              />
            </Link>
            <div className="flex-1">
              <Link href={`/people/${match.id}`} className="font-display text-2xl tracking-wide text-ink hover:text-green">
                {match.displayName}
              </Link>
              <p className="mt-1 text-xs text-faint">
                {match.visits} visit{match.visits === 1 ? "" : "s"} · last seen {fmtRelative(match.lastSeen)}
              </p>
              <ul className="mt-3 space-y-px">
                {match.timeline.slice(0, 5).map((s) => (
                  <li key={s.id} className="flex flex-wrap items-center gap-x-3 text-xs">
                    <span className="tracking-widest text-ink">{s.cameraName ?? `Camera ${s.cameraId}`}</span>
                    {s.location && <span className="text-faint">/ {s.location}</span>}
                    <span className="ml-auto text-faint">{fmtDate(s.startedAt)}</span>
                    <span className="tabular-nums text-faint">{fmtSpan(s.startedAt, s.endedAt)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
