"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  EnrollConflictError,
  NoFaceDetectedError,
  addEmbedding,
  enroll,
} from "@/lib/api";
import { CornerBrackets } from "@/components/CornerBrackets";

export default function EnrollPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [name, setName] = useState("");
  const [captured, setCaptured] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<EnrollConflictError | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
  }

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob) {
        setCaptured(blob);
        setPreviewUrl(URL.createObjectURL(blob));
      }
    }, "image/jpeg");
  }

  function retake() {
    setCaptured(null);
    setPreviewUrl(null);
  }

  async function save(force = false) {
    if (!captured || !name) return;
    setSubmitting(true);
    setError(null);
    try {
      await enroll(name, captured, force);
      router.push("/");
    } catch (err) {
      if (err instanceof NoFaceDetectedError) {
        setError("No face detected, try again.");
      } else if (err instanceof EnrollConflictError) {
        setConflict(err);
      } else {
        setError("Something went wrong, please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function resolveConflictAddPhoto() {
    if (!captured || !conflict) return;
    setSubmitting(true);
    try {
      await addEmbedding(conflict.existingPerson.id, captured);
      router.push("/");
    } catch {
      setError("Something went wrong, please try again.");
    } finally {
      setSubmitting(false);
      setConflict(null);
    }
  }

  async function resolveConflictEnrollAnyway() {
    setConflict(null);
    await save(true);
  }

  return (
    <main className="mx-auto w-full max-w-lg px-6 py-10">
      <header className="mb-8 border-b border-edge pb-6">
        <p className="mb-1 text-xs tracking-[0.3em] text-faint">
          FACE-DASH // NEW ENTRY
        </p>
        <h1 className="font-display text-4xl font-bold tracking-wide text-ink">
          ENROLL SUBJECT
        </h1>
      </header>

      <label className="mb-1 block text-[11px] uppercase tracking-widest text-faint">
        Identity label
      </label>
      <div className="mb-6 flex items-center border border-edge bg-surface px-3 focus-within:border-edge-bright">
        <span className="mr-2 text-green">{">"}</span>
        <input
          className="w-full bg-transparent py-2.5 text-ink outline-none placeholder:text-faint"
          placeholder="e.g. alice"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      {!previewUrl ? (
        <div>
          <div className="relative aspect-video w-full overflow-hidden border border-edge bg-void">
            <CornerBrackets color="var(--green)" />
            <video ref={videoRef} className="h-full w-full object-cover" muted />
            <span className="absolute left-2 top-2 flex items-center gap-1.5 text-[11px] uppercase tracking-widest text-green">
              <span className="h-1.5 w-1.5 rounded-full bg-green rec-dot" />
              live
            </span>
          </div>
          <div className="mt-3 flex gap-3">
            <button
              className="flex-1 border border-edge-bright px-4 py-2.5 text-sm uppercase tracking-widest text-ink transition-colors hover:bg-surface-raised"
              onClick={startCamera}
              type="button"
            >
              Start camera
            </button>
            <button
              className="flex-1 border border-green/60 px-4 py-2.5 text-sm uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void"
              onClick={capture}
              type="button"
            >
              Capture
            </button>
          </div>
        </div>
      ) : (
        <div>
          <div className="relative aspect-video w-full overflow-hidden border border-edge bg-void">
            <CornerBrackets color="var(--amber)" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={previewUrl} alt="Captured face" className="h-full w-full object-cover" />
            <span className="absolute left-2 top-2 text-[11px] uppercase tracking-widest text-amber">
              frame captured
            </span>
          </div>
          <div className="mt-3 flex gap-3">
            <button
              className="flex-1 border border-edge-bright px-4 py-2.5 text-sm uppercase tracking-widest text-ink transition-colors hover:bg-surface-raised"
              onClick={retake}
              type="button"
            >
              Retake
            </button>
            <button
              className="flex-1 border border-green/60 px-4 py-2.5 text-sm uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void disabled:cursor-not-allowed disabled:opacity-40"
              onClick={() => save(false)}
              disabled={!name || submitting}
              type="button"
            >
              {submitting ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden" />

      {error && (
        <div className="mt-5 border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          [!] {error}
        </div>
      )}

      {conflict && (
        <div className="mt-5 border border-amber/60 bg-amber-dim px-4 py-4">
          <p className="mb-3 text-sm leading-relaxed text-ink">
            <span className="text-amber">[!] match found —</span> this looks like{" "}
            <strong className="text-amber">{conflict.existingPerson.name}</strong> (cosine{" "}
            {conflict.score.toFixed(2)}). Same person?
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              className="border border-edge-bright px-3 py-1.5 text-xs uppercase tracking-widest text-ink transition-colors hover:bg-surface-raised"
              onClick={() => setConflict(null)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="border border-green/60 px-3 py-1.5 text-xs uppercase tracking-widest text-green transition-colors hover:bg-green hover:text-void"
              onClick={resolveConflictAddPhoto}
              type="button"
            >
              Add photo to {conflict.existingPerson.name}
            </button>
            <button
              className="border border-amber/60 px-3 py-1.5 text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void"
              onClick={resolveConflictEnrollAnyway}
              type="button"
            >
              Enroll as new person
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
