const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Person = {
  id: number;
  name: string;
  thumbnailUrl: string;
};

export type EnrollConflict = {
  existingPerson: Person;
  score: number;
};

export class EnrollConflictError extends Error {
  existingPerson: Person;
  score: number;
  constructor(conflict: EnrollConflict) {
    super("Face matches an existing person under a different name");
    this.existingPerson = conflict.existingPerson;
    this.score = conflict.score;
  }
}

export class NoFaceDetectedError extends Error {
  constructor() {
    super("No face detected in the image");
  }
}

export function thumbnailUrl(person: Person): string {
  return `${API_URL}${person.thumbnailUrl}`;
}

export async function listPeople(): Promise<Person[]> {
  const res = await fetch(`${API_URL}/people`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to list people: ${res.status}`);
  return res.json();
}

async function postImage(
  path: string,
  image: Blob,
  fields: Record<string, string> = {}
): Promise<Person> {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    form.append(key, value);
  }
  form.append("image", image, "capture.jpg");

  const res = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  if (res.status === 422) throw new NoFaceDetectedError();
  if (res.status === 409) {
    const body = await res.json();
    throw new EnrollConflictError(body.detail);
  }
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export function enroll(name: string, image: Blob, force = false): Promise<Person> {
  return postImage("/enroll", image, { name, force: String(force) });
}

export function addEmbedding(personId: number, image: Blob): Promise<Person> {
  return postImage(`/people/${personId}/embeddings`, image);
}
