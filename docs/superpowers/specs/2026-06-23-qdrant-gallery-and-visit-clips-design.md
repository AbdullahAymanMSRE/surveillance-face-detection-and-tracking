# Design: Qdrant-backed gallery + per-visit video clips

**Date:** 2026-06-23
**Status:** Approved (pre-implementation)

Two independent features for the multi-camera surveillance system:

1. **Qdrant-backed gallery** — replace the in-memory NumPy matrix with Qdrant as
   the authoritative embedding store.
2. **Per-visit video clips** — record a short clip of each visit and surface it
   on the person's timeline.

They are independent and implemented in that order (the vector DB touches the
core matching path; clips are additive).

---

## Feature A — Qdrant gallery (authoritative)

### Goal

Swap the brute-force in-memory matcher (`api/gallery.py:InMemoryGallery`) for a
Qdrant-backed implementation that scales beyond a single process's RAM and
supports per-person payload filtering. Qdrant becomes the **authoritative**
store for face embeddings; the SQLite `FaceEmbedding` table is removed. The
`Person` table stays in SQLite (labels, `best_sharpness`, identity rows).

### Current state (what changes)

- `api/gallery.py` — `GalleryMatcher` Protocol (`match`, `add`) + `InMemoryGallery`
  with extra methods `load`, `count_for_person`, `best_for_person`, `__len__`.
- `api/models.py` — `FaceEmbedding(id, person_id, vector: bytes)` is authoritative.
- `api/main.py` startup — `gallery.load(rows)` rebuilds the matrix from SQLite.
- `api/routers/sightings.py` — uses `match`, `add`, `count_for_person`,
  `best_for_person`, and inserts `FaceEmbedding` rows.
- `api/routers/people.py` (`/search`) — uses `match`.
- `api/consolidate.py` — reassigns `FaceEmbedding` rows on merge, rebuilds gallery.
- `api/matching.py` — dead code (`find_best_match`, never imported).

### New design

**`QdrantGallery`** implements the `GalleryMatcher` protocol plus the extra
methods callers depend on. The `GalleryMatcher` Protocol is extended to declare
`count_for_person` and `best_for_person` so both implementations share one
contract.

| Method | Qdrant operation |
|--------|------------------|
| `match(vec)` | search top-1 in collection `faces`; return `(person_id, score)` if `score >= threshold (0.28)`, else `(None, score)` |
| `add(person_id, vec)` | `upsert` a point: random UUID id, vector = normalized embedding, payload `{person_id, sharpness}` |
| `count_for_person(pid)` | `count` with payload filter `person_id == pid` |
| `best_for_person(pid, vec)` | search filtered to `person_id == pid`, return top-1 score (0.0 if none) |
| `ensure_collection()` | create collection `faces` if missing (replaces `load`) |
| `__len__` | `count` of all points |

- **Collection `faces`:** vector size 512, distance `Cosine`. Embeddings are
  already L2-normalized upstream (ArcFace output), so the existing `0.28`
  threshold transfers unchanged. `QdrantGallery` still normalizes defensively.
- **`match_create_lock` is retained.** The "no match → create a new person"
  decision is a read-then-write that two cameras can interleave; the lock in
  `api/gallery.py` continues to serialize match → (create) → add.
- **Person creation** is unchanged in shape: a `Person` row is inserted in SQLite
  (autoincrement id), and that `person_id` becomes the Qdrant payload value.
- **`api/main.py`:** remove the `gallery.load(...)` rebuild; call
  `gallery.ensure_collection()` at startup. Qdrant persists across restarts, so
  no rebuild from SQLite is needed.
- **`api/consolidate.py`:** on merge, instead of reassigning `FaceEmbedding`
  rows, **set the `person_id` payload** on the loser's Qdrant points to the
  survivor (`set_payload` with a `person_id == loser` filter), then delete the
  loser `Person` row. No gallery rebuild call afterwards.
- **`api/matching.py`:** deleted.
- **`api/models.py`:** `FaceEmbedding` model removed; `Person` and `Sighting`
  retained (`Sighting` gains `has_clip`, see Feature B).

### Gallery singleton / config

- `get_gallery()` returns a process-wide `QdrantGallery`. It connects using
  `QDRANT_URL` (default `http://localhost:6333`) for the running system, or a
  local in-memory client for tests (see Testing).
- `reset_gallery()` (used by tests) drops/recreates the collection.

### Deployment

- Add `qdrant-client` to `requirements.txt`.
- Qdrant server runs via Docker. New Makefile target:
  `make qdrant` → `docker run -p 6333:6333 -v <data>/qdrant:/qdrant/storage qdrant/qdrant`.
- README "Quick start" gains a step: start Qdrant (`make qdrant`) before `make api`.
- Env var `QDRANT_URL` documented alongside `FACE_API_DATA_DIR` etc.

### Testing

- `qdrant-client` supports a local mode (`QdrantClient(":memory:")` or a local
  path) that runs in-process with no server. The test suite uses `:memory:` so it
  stays self-contained (no Docker in CI).
- `conftest.py` wires `get_gallery()` to a fresh in-memory `QdrantGallery` per
  test (via the existing dependency-injection / `reset_gallery` pattern).
- Existing gallery/sighting/search tests are updated to assert against the
  Qdrant-backed gallery (behavior is identical: same `match`/`add`/threshold
  semantics).

### Migration

- One-time `scripts/migrate_to_qdrant.py`: reads existing `FaceEmbedding` rows
  via **raw SQL** (so it does not depend on the soon-removed model), upserts each
  `(person_id, vector)` into Qdrant, and reports counts. Run once before the
  `FaceEmbedding` table is dropped.
- Current data is tiny (1 person, ~6 vectors), so this is low-risk. The
  `app.db.bak` backup from earlier remains a safety net for `Person` rows.

---

## Feature B — per-visit video clips

### Goal

Record a short clip of each visit (one clip per `Sighting`) and show it inline on
the person's appearance timeline. Clip: up to 5 seconds, ~10fps, downscaled to
640px wide.

### Worker (`pipeline_node.py`)

- A `ClipRecorder` is associated with each open sighting (keyed by `track_id`,
  alongside the existing `open_sightings` map).
- In the capture loop, for each open sighting, append the current frame
  (downscaled to 640px wide, aspect preserved) to its recorder when:
  - the recorder holds < `CLIP_MAX_FRAMES` (≈ 5s × 10fps = 50), **and**
  - at least `1/CLIP_FPS` seconds have elapsed since its last appended frame
    (decouples clip fps from source fps).
- Frames are held in memory (≈50 × 640×360×3 ≈ small per active visit).
- On visit end (in `_post_end` / the close path), encode buffered frames to an
  MP4 with `cv2.VideoWriter` (fourcc: try `avc1` for H.264/browser playback, fall
  back to `mp4v`), write to a temp file, and
  `POST /sightings/{id}/clip` (multipart `clip` file). Drop the recorder.
- If a visit has 0 buffered frames (e.g. closed instantly), no clip is posted.

### API

- `POST /sightings/{sighting_id}/clip` — multipart `clip: UploadFile`. Validates
  the sighting exists; saves bytes to `get_clips_dir()/{sighting_id}.mp4`; sets
  `Sighting.has_clip = True`. A max upload size guard rejects oversized files.
- `GET /sightings/{sighting_id}/clip` — returns the MP4 via Starlette
  `FileResponse` (`media_type="video/mp4"`), which supports HTTP Range requests so
  the browser can seek/scrub. 404 if no clip.
- `api/db.py` gains `get_clips_dir()` → `get_data_dir() / "clips"`.
- `api/models.py`: `Sighting.has_clip: bool = False`.
- `api/serializers.py`: `sighting_response` gains `hasClip` and `clipUrl`
  (`/sightings/{id}/clip`).

### Dashboard

- `web/lib/api.ts`: add `clipUrl(sightingId: number)` → absolute
  `/sightings/{id}/clip`; extend the sighting type with `hasClip`.
- `web/app/people/[id]/page.tsx`: each visit row in the timeline that has
  `hasClip` renders an inline `<video controls preload="none" src={clipUrl(...)}>`.
  Visits without a clip render as today.
- Clips live on the **Person detail timeline**, not the Live page (Live shows
  only currently-open sightings, which have no clip yet).

### Testing

- API tests: `POST /sightings/{id}/clip` with small fake MP4 bytes saves the file
  and flips `has_clip`; `GET` returns the bytes with `video/mp4`; the serializer
  includes `clipUrl`/`hasClip`. A `GET` for a sighting with no clip returns 404.

---

## Risks & notes

1. **Qdrant is a hard dependency.** With Qdrant authoritative, the API cannot
   match faces unless Qdrant is running. This is the accepted tradeoff of the
   "authoritative" choice. Vector backups are now Qdrant snapshots, not `app.db`.
2. **Browser codec.** OpenCV's `avc1`/H.264 support depends on the local FFmpeg
   build. If `avc1` is unavailable, `mp4v` (MPEG-4 Part 2) clips may not play in
   all browsers. Verify on the target machine during implementation; the worker
   logs which fourcc it used.
3. **Clip storage growth is unbounded** — one file per visit. Retention/cleanup
   is explicitly out of scope (future work). `make clean-data` clears
   `api/data/` (including `clips/`); Qdrant data lives in its own Docker volume.
4. **`match_create_lock`** remains necessary and unchanged.

## Out of scope (future work)

- Clip retention / rotation policy.
- `DELETE /people/{id}` endpoint (person deletion is still a manual op).
- Annotated (boxes/labels) clips — raw frames only.
- Authentication (pre-existing gap, unchanged here).
