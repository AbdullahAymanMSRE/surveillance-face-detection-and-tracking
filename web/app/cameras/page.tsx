"use client";

import { useState } from "react";
import {
  createCamera,
  deleteCamera,
  listCameras,
  previewUrl,
  startCamera,
  stopCamera,
} from "@/lib/api";
import { usePoll } from "@/lib/usePoll";
import { CornerBrackets } from "@/components/CornerBrackets";

export default function CamerasPage() {
  const { data, error } = usePoll(listCameras, 2000);
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [source, setSource] = useState("");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<number | null>(null);

  const addCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !source.trim()) {
      setFormError("Name and source are required.");
      return;
    }
    setBusy(true);
    setFormError(null);
    try {
      await createCamera({ name: name.trim(), location: location.trim(), source: source.trim() });
      setName("");
      setLocation("");
      setSource("");
    } catch {
      setFormError("Could not create the camera.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10">
      <header className="mb-8 border-b border-edge pb-6">
        <p className="mb-1 text-xs tracking-[0.3em] text-faint">FLEET // CAMERA CONTROL</p>
        <h1 className="font-display text-4xl font-bold tracking-wide text-ink">CAMERAS</h1>
      </header>

      <form onSubmit={addCamera} className="mb-10 border border-edge bg-surface p-5">
        <p className="mb-4 text-xs uppercase tracking-widest text-faint">+ Add camera</p>
        <div className="grid gap-3 sm:grid-cols-3">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Name (e.g. Front Gate)"
            className="border border-edge bg-void px-3 py-2 text-sm text-ink outline-none focus:border-edge-bright"
          />
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="Location (e.g. Lobby)"
            className="border border-edge bg-void px-3 py-2 text-sm text-ink outline-none focus:border-edge-bright"
          />
          <input
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="Source (0, rtsp://…, http://…/stream, file.mp4)"
            className="border border-edge bg-void px-3 py-2 text-sm text-ink outline-none focus:border-edge-bright"
          />
        </div>
        {formError && <p className="mt-3 text-xs text-red">{formError}</p>}
        <button
          type="submit"
          disabled={busy}
          className="mt-4 border border-green/60 px-5 py-2 text-sm uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void disabled:opacity-40"
        >
          {busy ? "Adding…" : "Add camera"}
        </button>
      </form>

      {error && (
        <p className="border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          Cannot reach the API. Is the backend running?
        </p>
      )}

      <div className="space-y-px">
        {data?.length === 0 && (
          <p className="border border-dashed border-edge px-6 py-12 text-center text-sm text-faint">
            No cameras yet. Add one above.
          </p>
        )}
        {data?.map((cam) => {
          const running = cam.status === "running";
          const showing = previewId === cam.id;
          return (
            <div key={cam.id} className="border border-edge bg-surface">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
                <span
                  className={`h-2 w-2 rounded-full ${running ? "bg-green shadow-[0_0_6px_var(--green)] rec-dot" : "bg-faint"}`}
                />
                <div className="min-w-0">
                  <p className="font-display tracking-widest text-ink">{cam.name}</p>
                  <p className="truncate text-[11px] text-faint">
                    {cam.location || "—"} · <span className="text-faint/70">{cam.source}</span>
                  </p>
                </div>
                <span
                  className={`ml-auto text-xs uppercase tracking-widest ${running ? "text-green" : "text-faint"}`}
                >
                  {cam.status}
                </span>
                {running && (
                  <button
                    onClick={() => setPreviewId(showing ? null : cam.id)}
                    className={`border px-3 py-1.5 text-xs uppercase tracking-widest transition-colors ${
                      showing
                        ? "border-ink/60 text-ink"
                        : "border-edge-bright text-faint hover:text-ink"
                    }`}
                  >
                    {showing ? "Hide" : "Preview"}
                  </button>
                )}
                {running ? (
                  <button
                    onClick={() => stopCamera(cam.id)}
                    className="border border-amber/60 px-3 py-1.5 text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={() => startCamera(cam.id)}
                    className="border border-green/60 px-3 py-1.5 text-xs uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void"
                  >
                    Start
                  </button>
                )}
                <button
                  onClick={() => {
                    if (confirm(`Delete camera "${cam.name}"?`)) deleteCamera(cam.id);
                  }}
                  className="border border-red/50 px-3 py-1.5 text-xs uppercase tracking-widest text-red transition-colors hover:bg-red hover:text-void"
                >
                  Delete
                </button>
              </div>
              {showing && (
                <div className="border-t border-edge p-4">
                  <div className="relative mx-auto max-w-xl border border-edge bg-void">
                    <CornerBrackets color="var(--green)" size={12} />
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl(cam.id)}
                      alt={`${cam.name} live preview`}
                      className="w-full"
                    />
                    <span className="absolute right-2 top-2 flex items-center gap-1.5 bg-void/70 px-2 py-1 text-[10px] uppercase tracking-widest text-green">
                      <span className="h-1.5 w-1.5 rounded-full bg-green rec-dot" /> live
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </main>
  );
}
