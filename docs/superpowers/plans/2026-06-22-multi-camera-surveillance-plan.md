# Multi-Camera Live Surveillance — Implementation Plan

Implements `specs/2026-06-22-multi-camera-surveillance-design.md`.

Phased so the system is runnable and demoable after **every** phase. Each phase ends
with an explicit verification step. Earlier phases use a **video file** as a camera
(decision 15) so nothing depends on hardware until Phase 7.

Legend: 🟢 = system is end-to-end demoable at this checkpoint.

---

## Phase 0 — Groundwork: extract the shared pipeline core

Goal: get the recognition logic into a reusable, headless, I/O-free module so the rest
of the build has something clean to depend on. No behavior change yet.

1. Create `pipeline/core.py` (new package). Move out of `live_pipeline.py`:
   the YOLO detector wrapper, `IoUTracker`, the FaceProcessor/alignment call, the
   ArcFace embedder call, and the threaded detect→track→embed loop — but with **no**
   data sink and **no** `cv2` window. Expose something like:
   `for event in core.run(source): event = {track_key, box, crop, embedding, sharpness, ts}`.
2. Rewrite `live_pipeline.py` to consume `pipeline/core.py` for its window + local
   file DB (so the standalone debug viewer still works exactly as before).
3. Reuse, don't duplicate: `face_extraction/` (YOLO), `face_recognition/arcface_onnx.py`,
   `face_recognition/face_align.py`.

**Verify:** `python live_pipeline.py` still detects/recognizes from the webcam with a
window, identical to before. Existing `pytest tests/api/` still green.

---

## Phase 1 — Data model + gallery (server foundations) 🟢(API)

Goal: the API can store cameras/persons/sightings and match embeddings centrally.

1. Extend `api/models.py`: rename `Person.name` → `Person.label` (nullable); add
   `Camera` and `Sighting` tables per the spec.
2. `api/gallery.py`: `GalleryMatcher` interface + in-memory NumPy implementation.
   `match(vec)->person_id|None` (cosine, threshold from `api/matching.py`),
   `add(person_id, vec)`. Loaded from `FaceEmbedding` at startup.
3. `api/locks.py` (or inline): a single lock guarding match-or-create.
4. Wire startup in `api/main.py` to build the gallery from the DB.

**Verify:** new unit tests — gallery returns a known match above threshold and `None`
below; concurrent match-or-create on the same new embedding yields exactly one person
(simulate two calls under the lock). `pytest` green.

---

## Phase 2 — Ingestion endpoints (worker → API)

Goal: the API accepts the sighting lifecycle and assigns anonymous identities.

1. `POST /sightings` — match-or-create (locked) → create open `Sighting`, save/refresh
   thumbnail if sharper, return `{person_id, sighting_id}`.
2. `POST /sightings/{id}/heartbeat`, `POST /sightings/{id}/end`.
3. Reaper background task (close sightings with stale `last_seen`); start it on app
   startup with a configurable interval/timeout.

**Verify:** tests drive the lifecycle with fixture faces — first face creates
`person_1` + open sighting; same face again (new camera) matches `person_1`; end sets
`ended_at`; reaper closes a deliberately stale open sighting. `pytest` green.

---

## Phase 3 — Cameras CRUD + supervisor + worker 🟢(end-to-end, file source)

Goal: register a camera, have the server run a worker against it, see sightings appear.

1. `api/supervisor.py`: `start(camera)`, `stop(camera_id)`, `status()`, crash-restart;
   maps `camera_id → subprocess` running `pipeline_node.py`.
2. Camera endpoints: `POST/GET/PATCH/DELETE /cameras`, `POST /cameras/{id}/start|stop`,
   `GET /cameras` includes worker status. On startup, auto-start workers for
   `enabled` cameras.
3. `pipeline_node.py` (headless): `--camera-id --api-url`; `GET /cameras/{id}` for the
   source; open it (loop if file); run `pipeline/core.py`; per track → `POST /sightings`
   (start) → heartbeat while alive → `POST /sightings/{id}/end` on drop.
4. Commit a short sample clip (or document where to drop one) for hardware-free runs.

**Verify:** `POST /cameras` with `source` = the sample clip → `POST /cameras/{id}/start`
→ `GET /sightings/active` shows people; `GET /people` grows; `/end` fires when clips
leave frame; `stop` kills the worker. Confirm via curl/HTTP.

---

## Phase 4 — Read endpoints + Search

1. `GET /people` (thumbnail, visit count, last seen), `GET /people/{id}` (timeline),
   `GET /people/{id}/thumbnail`, `GET /sightings/active`.
2. `POST /search` — image → match → `{person, sightings}` or `{match:null}`; **no
   writes** (assert in a test).
3. `PATCH /people/{id}` to set the optional `label`.

**Verify:** tests for each; search returns a known person without creating rows
(row counts unchanged before/after). `pytest` green.

---

## Phase 5 — Dashboard rebuild 🟢(full product, file source)

Goal: the five screens, polling, against the real API. Reuse the existing dark "HUD"
styling.

1. Extend `web/lib/api.ts`: people, person detail, active sightings, cameras (+ start/
   stop), search, label edit.
2. Pages: `/` **Live/Now** (active sightings grouped by camera, ~1–2s poll);
   `/people`; `/people/[id]` **timeline**; `/cameras` (list + status + add-camera form +
   start/stop); `/search`.
3. Repurpose the old `/enroll` page into `/search`. Make people cards link to detail.

**Verify:** with a worker running off the sample clip, the dashboard shows live "now",
the people list grows, a person's timeline lists visits with camera + times, search
finds a known face, and add-camera + start/stop work from the UI.

---

## Phase 6 — Live preview (server-proxied raw MJPEG, on-demand)

1. `pipeline_node.py` serves raw frames as MJPEG on a local port, encoding **only**
   while a client is attached.
2. `GET /cameras/{id}/preview` proxies that stream (supervisor knows the port).
3. Dashboard: `<img src=".../cameras/{id}/preview">` on the camera view; mount only
   when open so idle cameras don't stream.

**Verify:** opening a camera shows live video in the browser; closing it stops the
encode (CPU drops); works with the file source.

---

## Phase 7 — Real multi-camera (laptops / IP cameras) 🟢(real hardware)

1. `publisher.py` (~30 lines): laptop webcam → MJPEG at `http://0.0.0.0:8090/stream`.
2. On each laptop: run `publisher.py`. On the server dashboard: add a camera with
   `source = http://<laptop-ip>:8090/stream`, start it.
3. Verify cross-camera identity: same person in front of two laptops → **one**
   `person_id`, two sightings on two cameras in the timeline.
4. Confirm `rtsp://…` works unchanged for a cheap IP camera (just another source).

**Verify:** two laptops streaming; dashboard Live/Now shows both; one person appears
under both cameras as a single identity with a merged timeline; preview works for each.

---

## Phase 8 — Orchestration & polish

1. One-command bring-up: a `Makefile`/`docker-compose` (API+supervisor, web) or a
   documented `start.sh`. Loop the sample clip for a zero-hardware demo.
2. README: architecture diagram, run instructions (server, laptops, demo mode), the
   five screens, and explicit future work (auth, annotated preview, vector DB, SSE).
3. Backfill tests for any gaps; confirm full suite green.

**Verify:** fresh checkout → one command → working system on the sample clip with no
hardware; documented path to add laptops.

---

## Risks / watch-items

- **Server CPU** with many streams (spec performance note) — mitigate with frame-skip /
  lower resolution / fewer cameras, or a GPU.
- **`Person.name` → `label` rename** touches `serializers.py`, routers, `web/lib/api.ts`
  and the frontend — do it in Phase 1 and fix call sites together.
- **Preview resource use** — strictly on-demand; never stream all cameras at once.
- **Threshold tuning** for cross-camera identity — reuse the existing 0.28 default;
  expect to tune once real cameras are in (Phase 7).
