const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Sighting = {
  id: number;
  cameraId: number;
  cameraName: string | null;
  location: string | null;
  startedAt: string | null;
  endedAt: string | null;
  lastSeen: string | null;
  active: boolean;
};

export type PersonSummary = {
  id: number;
  label: string | null;
  displayName: string;
  thumbnailUrl: string;
  visits: number;
  lastSeen: string | null;
  active: boolean;
};

export type PersonDetail = PersonSummary & { timeline: Sighting[] };

export type ActiveSighting = Sighting & {
  person: Pick<PersonSummary, "id" | "label" | "displayName" | "thumbnailUrl"> | null;
};

export type Camera = {
  id: number;
  name: string;
  location: string;
  source: string;
  enabled: boolean;
  status: "running" | "stopped";
};

export type SearchResult = { match: PersonDetail | null; score: number };

export class NoFaceDetectedError extends Error {
  constructor() {
    super("No face detected in the image");
  }
}

export function absUrl(path: string): string {
  return `${API_URL}${path}`;
}

export function thumbnailUrl(personId: number): string {
  return absUrl(`/people/${personId}/thumbnail`);
}

export function previewUrl(cameraId: number): string {
  return absUrl(`/cameras/${cameraId}/preview`);
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(absUrl(path), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

export const listPeople = () => getJSON<PersonSummary[]>("/people");
export const getPerson = (id: number) => getJSON<PersonDetail>(`/people/${id}`);
export const listActiveSightings = () =>
  getJSON<ActiveSighting[]>("/sightings/active");
export const listCameras = () => getJSON<Camera[]>("/cameras");

export async function updateLabel(id: number, label: string | null): Promise<void> {
  const res = await fetch(absUrl(`/people/${id}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error(`PATCH person ${id} failed: ${res.status}`);
}

export type CameraInput = {
  name: string;
  location: string;
  source: string;
  enabled?: boolean;
};

export async function createCamera(body: CameraInput): Promise<Camera> {
  const res = await fetch(absUrl("/cameras"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Create camera failed: ${res.status}`);
  return res.json();
}

export async function startCamera(id: number): Promise<void> {
  await fetch(absUrl(`/cameras/${id}/start`), { method: "POST" });
}

export async function stopCamera(id: number): Promise<void> {
  await fetch(absUrl(`/cameras/${id}/stop`), { method: "POST" });
}

export async function deleteCamera(id: number): Promise<void> {
  await fetch(absUrl(`/cameras/${id}`), { method: "DELETE" });
}

export async function search(image: Blob): Promise<SearchResult> {
  const form = new FormData();
  form.append("image", image, "search.jpg");
  const res = await fetch(absUrl("/search"), { method: "POST", body: form });
  if (res.status === 422) throw new NoFaceDetectedError();
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}
