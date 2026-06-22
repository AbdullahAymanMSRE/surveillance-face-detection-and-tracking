# Face Dashboard — Foundation + Enrollment Slice

## Goal

Build a web dashboard on top of the existing face detection/recognition pipeline
(`face_extraction/`, `face_recognition/`) so a person can be enrolled by name +
webcam photo from a browser, and later recognized live (future phase). This spec
covers the **foundation** (project skeleton, stack, repo layout) plus the **first
end-to-end vertical slice**: enrollment, with duplicate-face detection.

Live recognition (continuous webcam stream → real-time identification) is
explicitly **out of scope** for this spec — it's the next phase, built on this
foundation.

## Repo structure

Single git repository (not a managed monorepo — no Turborepo/Nx/pnpm-workspaces;
there's exactly one Next.js app and one Python service, and they only
communicate over HTTP, not shared imports):

```
web/                       # Next.js (TypeScript, App Router, Tailwind) — pure frontend
  app/
    page.tsx               # dashboard: list enrolled people
    enroll/page.tsx        # enrollment page: name + webcam capture
  .env.local               # NEXT_PUBLIC_API_URL=http://localhost:8000

api/                        # FastAPI service — owns the DB and all ML inference
  main.py                   # FastAPI app, CORS for the Next.js origin, route registration
  db.py                     # SQLite engine/session setup (SQLModel)
  models.py                 # Person, FaceEmbedding tables
  routers/
    people.py                # GET /people, GET /people/{id}/thumbnail
    enroll.py                 # POST /enroll, POST /people/{id}/embeddings
  data/
    app.db                   # SQLite file — touched only by this process
    thumbnails/<personId>.jpg

face_extraction/, face_recognition/   # existing, reused as-is (imported by api/main.py)
live_pipeline.py, requirements.txt    # existing, untouched
```

**FastAPI is the sole owner of the database.** Next.js never opens the database
or filesystem directly — every read and write, including thumbnail images,
goes through FastAPI's REST API. This means FastAPI is the only thing that
would need to change if SQLite is later swapped for a real DB server
(e.g. Postgres).

## Tech stack

- **Next.js (`web/`):** TypeScript, App Router, Tailwind for styling. Talks to
  FastAPI only via `fetch`, base URL from `NEXT_PUBLIC_API_URL`. No DB client.
- **FastAPI (`api/`):** added to the existing root `requirements.txt`:
  `fastapi`, `uvicorn[standard]`, `python-multipart`, `sqlmodel`. Reuses
  `ultralytics`, `opencv`, `torch`, `onnxruntime` already in the repo, and
  imports detection/alignment/embedding code directly from
  `face_extraction/` and `face_recognition/` — no changes needed to that code.
  `CORSMiddleware` allows the Next.js dev origin.
- **Database:** SQLite, single file at `api/data/app.db`, accessed only by the
  FastAPI process via SQLModel.

**Running it locally** (two processes):

```bash
uvicorn api.main:app --reload --port 8000     # from repo root
cd web && npm run dev                          # port 3000
```

## Database schema

```python
class Person(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: datetime

class FaceEmbedding(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="person.id")
    vector: bytes          # 512-d float32 array, serialized (np.tobytes())
    created_at: datetime
```

A `Person` can have multiple `FaceEmbedding` rows (one per enrolled photo).
Thumbnails are stored as `api/data/thumbnails/{person_id}.jpg`, keyed by
convention rather than a DB column. If a newly enrolled photo is sharper than
the current thumbnail (reusing the sharpness check already implemented in
`face_recognition/face_db.py`), it replaces the file.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/enroll` | POST | `{name, image, force?}` → detect, align, embed, duplicate-check, create `Person` + `FaceEmbedding`. See flow below. |
| `/people` | GET | List all people: `[{id, name, thumbnailUrl}, ...]` |
| `/people/{id}/thumbnail` | GET | Serve the thumbnail JPEG for a person |
| `/people/{id}/embeddings` | POST | `{image}` → add another face sample to an existing person (used by the duplicate-conflict "add photo instead" action) |

## Data flow — enrollment with duplicate detection

```
Browser (/enroll)
  1. User types a name, grants webcam access, captures one snapshot (canvas → JPEG blob)
  2. POST multipart/form-data {name, image} → FastAPI POST /enroll
                                                      │
FastAPI                                               ▼
  3. Decode image → YOLO face detection (reuses face_extraction model)
       no face found        → 422 "No face detected"
       face(s) found        → take the largest/highest-confidence box
  4. Align crop (face_align.py, YuNet) → ArcFace embed (arcface_onnx.py)
  5. Compare new embedding against ALL existing FaceEmbeddings (cosine,
     same match logic as face_recognition/face_db.py already uses)
       best match score >= threshold AND matched person's name != submitted name:
            → 409 Conflict, no DB write yet:
                 { existingPerson: {id, name, thumbnailUrl}, score }
            → frontend shows: "This looks like <name> (score 0.41). Same person?"
                 [Cancel]  [Add photo to <name> instead]  [No, enroll as new person]
       no strong match:
            → proceed to create a brand-new Person
       best match score >= threshold AND matched person's name == submitted name:
            → treat as re-enrolling the same known person: add this embedding to
              the existing Person (via the same path as "add photo to existing
              person" below) instead of creating a duplicate Person row
  6. Insert Person{name} + FaceEmbedding{personId, vector} into SQLite
  7. Save crop to data/thumbnails/<personId>.jpg
  8. Return 201 {id, name, thumbnailUrl}
                                                      │
Browser                                               ▼
  9. Show success, navigate to dashboard (/)
  10. Dashboard GET /people → render name + thumbnail for each person
```

Follow-up actions from the conflict dialog:
- **"Add photo to existing person"** → `POST /people/{id}/embeddings` — adds
  this embedding to the existing person, no new `Person` row.
- **"Enroll as new person anyway"** → re-POST `/enroll` with `force=true`,
  skipping the duplicate check (handles legitimate look-alikes / false positives).
  `force=true` always creates a brand-new `Person`, even if the face would
  otherwise match (by name or by similarity).

Single snapshot per enrollment for this slice (not the old 10-shot gallery
flow) — simplest path to prove the chain works end-to-end. Multi-shot
averaging is a natural follow-up, not needed now.

## UI

**`/` (dashboard):**
- `GET /people` on load
- Grid/list of cards: thumbnail + name per person
- Empty state ("No one enrolled yet")
- "Enroll new person" button → `/enroll`

**`/enroll`:**
- Name text input
- Live `<video>` webcam preview (`getUserMedia`)
- "Capture" button → snapshot to `<canvas>`, preview, retake option
- "Save" button → `POST /enroll`
- On success → redirect to `/`
- On 422 (no face) → inline retry message
- On 409 (conflict) → dialog with the three actions above

## Testing approach

- **FastAPI side** (where the actual logic lives — detection, alignment,
  embedding, duplicate matching): integration tests with `TestClient` against
  fixture face images covering: enroll creates a `Person`; re-enrolling the
  same face under a different name returns 409 with the right payload;
  `force=true` bypasses it; `/people/{id}/embeddings` adds to an existing
  person instead of creating a new one; `/people` lists correctly.
- **Next.js side:** manual click-through for this slice (enroll a face, see it
  appear on the dashboard, trigger the conflict dialog) — no automated
  frontend tests yet, since this layer is presentation-only right now.

## Out of scope (future phases)

- Live continuous webcam recognition view (WebSocket frame streaming, overlay
  boxes/labels) — builds on this foundation but is its own spec.
- Visit/sighting counts and history (the existing CLI pipeline has this; not
  carried into the dashboard yet).
- Multi-user auth — single-user local use for now.
- Swapping SQLite for a networked DB server — possible later without touching
  Next.js, since FastAPI is the sole DB owner.
