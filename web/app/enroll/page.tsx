"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  EnrollConflictError,
  NoFaceDetectedError,
  addEmbedding,
  enroll,
} from "@/lib/api";

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
    <main className="mx-auto max-w-md p-8">
      <h1 className="mb-4 text-2xl font-semibold">Enroll a new person</h1>

      <input
        className="mb-4 w-full rounded border p-2"
        placeholder="Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />

      {!previewUrl ? (
        <div>
          <video ref={videoRef} className="w-full rounded bg-black" muted />
          <div className="mt-2 flex gap-2">
            <button
              className="rounded bg-gray-200 px-4 py-2"
              onClick={startCamera}
              type="button"
            >
              Start camera
            </button>
            <button
              className="rounded bg-blue-600 px-4 py-2 text-white"
              onClick={capture}
              type="button"
            >
              Capture
            </button>
          </div>
        </div>
      ) : (
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt="Captured face" className="w-full rounded" />
          <div className="mt-2 flex gap-2">
            <button className="rounded bg-gray-200 px-4 py-2" onClick={retake} type="button">
              Retake
            </button>
            <button
              className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
              onClick={() => save(false)}
              disabled={!name || submitting}
              type="button"
            >
              Save
            </button>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden" />

      {error && <p className="mt-4 text-red-600">{error}</p>}

      {conflict && (
        <div className="mt-4 rounded border border-yellow-400 bg-yellow-50 p-4">
          <p className="mb-3">
            This looks like <strong>{conflict.existingPerson.name}</strong> (score{" "}
            {conflict.score.toFixed(2)}). Is this the same person?
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={() => setConflict(null)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={resolveConflictAddPhoto}
              type="button"
            >
              Add photo to {conflict.existingPerson.name} instead
            </button>
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={resolveConflictEnrollAnyway}
              type="button"
            >
              No, enroll as new person
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
