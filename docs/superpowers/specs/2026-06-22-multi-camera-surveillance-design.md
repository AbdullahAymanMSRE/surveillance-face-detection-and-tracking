# Multi-Camera Live Surveillance — Design

## Goal

Turn the project from "a recognition pipeline + a manual face-enrollment app" into the
end-to-end surveillance system originally intended: **many cameras → central
detection + recognition + tracking → a web dashboard that shows anonymous detected
persons and, for each, every time they appeared, when, and at which camera/place.**

People are **never named or pre-enrolled** — the system auto-discovers anonymous
identities (`person_017`…) as faces appear, and matches them across cameras.

This supersedes the manual enrollment slice in
`2026-06-21-face-dashboard-foundation-design.md`. The `/enroll` page and its
name-required, conflict-resolution flow are repurposed, not extended.

## Locked design decisions

Each decision below was settled deliberately (see rationale inline). They are the
contract this build implements.

1. **Single source of truth = the API.** The pipeline no longer writes its own
   `face_recognition/face_db/*.pt` files as the system of record. Each camera worker
   reports over **HTTP** to the central FastAPI service, which owns the embedding
   gallery, performs **central** identity matching, and stores cameras + sightings in
   SQLite. *Why: the only model that works when cameras live on other machines, and
   the only way cross-camera identity is possible.*

2. **Anonymous-only auto-detection.** A `Person` is created **solely** by the
   pipeline when an unrecognized face appears. `name` becomes an optional, editable
   **label** an operator may add later (e.g. "janitor") — never required. Plus a
   **read-only image search**: upload a photo → match against the gallery → return the
   matched person + their timeline, or "no record". Search **never writes** (no new
   person/embedding/sighting).

3. **One sighting = one continuous visit.** A `Sighting` row is a single visit to one
   camera, with `started_at` / `ended_at` / `camera_id`. While the tracker holds the
   face it stays the same row; leaving and returning makes a new row. *Why: matches
   what an operator wants to read ("Front Gate 09:14–09:16, Lab 2 09:20–09:31"),
   storage-efficient, and the pipeline already knows visit boundaries via IoU tracking.*

4. **In-memory brute-force matching behind a `GalleryMatcher` interface.** Embeddings
   are held in an in-memory NumPy matrix; matching is a vectorized cosine op (reusing
   `api/matching.py`'s threshold approach), persisted to SQLite. A **vector DB is
   deferred** but the interface (`match()` / `add()`) makes it a drop-in later. The
   **match-or-create** step runs inside a **serialized critical section** (one lock) so
   two cameras seeing a brand-new face simultaneously cannot create duplicate persons.

5. **Cameras live in the DB, managed via admin API.** `POST /cameras`,
   `GET /cameras`; the dashboard has an "add camera" form. No config file as source of
   truth.

6. **Source stored in the `Camera` row + a central supervisor.** Each `Camera` has
   `{id, name, location, source, enabled}`. A **central supervisor** (in the API
   process) spawns and monitors one worker per camera and **restarts crashed workers**.
   The dashboard can **start/stop each camera's worker**.

7. **All workers run on the server; remote machines are stream sources.** A central
   supervisor can only manage local processes, so every recognition worker runs on the
   server. A remote laptop contributes its webcam by **publishing a stream** the server
   pulls. *Why: one supervisor, uniform dashboard control of every camera, no
   per-machine daemon to build.*

8. **Visit lifecycle = open / heartbeat / close + reaper.** Worker POSTs *start* when
   a track is first recognized (row with `ended_at = null` = currently visible),
   heartbeats periodically (`last_seen`), POSTs *end* when the track drops. A
   server-side **reaper** closes any open sighting whose heartbeat went stale (crashed
   worker / dead stream) so nothing hangs open.

9. **Dashboard freshness = polling (~1–2s).** Reuses the existing `fetch` client;
   self-healing. SSE is a later upgrade. (Dashboard only ever reads → no WebSocket.)

10. **Five screens:** Live/Now, People, Person detail (timeline), Cameras
    (manage + start/stop + status), Search.

11. **Pipeline refactored into a shared core + headless node.** The detect → track →
    align → embed logic is extracted into a reusable core module. A new **headless**
    `pipeline_node.py` uses it, gets config by `--camera-id`, sends embeddings to the
    API for central matching, and reports sightings. `live_pipeline.py` stays as an
    optional standalone local-debug viewer (its own window + local file DB).

12. **No auth now, designed to add later.** Runs open on a trusted LAN/localhost for
    the demo; endpoints stay clean so a token/login layer drops in later.

13. **Laptops publish MJPEG over HTTP.** A ~30-line publisher per laptop exposes its
    webcam at `http://<laptop-ip>:8090/stream`. The `source` field is just a string, so
    MJPEG laptops and future `rtsp://…` cameras coexist with no code change.

14. **Live preview = server-proxied raw MJPEG, on-demand.** The worker already decodes
    every frame; it re-emits the **raw** frame (no annotations) as an MJPEG endpoint the
    API proxies to the dashboard `<img>`. Only the camera being viewed streams. An
    annotated (boxes + labels) preview is a future tab on the same plumbing.

15. **Video-file sources supported (looping).** Because `source` is any string, a
    camera can point at a local video file, looping on EOF — lets the whole system run
    and demo on one machine with no hardware.

## Architecture

```
  REMOTE (laptops / IP cameras)            SERVER (one machine)                    BROWSER
  ─────────────────────────────           ──────────────────────────             ─────────
  laptop webcam                            ┌───────────────────────────┐
    └─ publisher.py ──MJPEG──► source ───► │ Supervisor (in API proc)  │
  IP camera ────────RTSP────► source ───► │   spawns/monitors workers │
  video file ───────path────► source ───► │     │                     │
                                          │     ▼                     │
                                          │  pipeline_node.py (per camera, headless)
                                          │   detect→track→align→embed (shared core)
                                          │     │ embedding + crop          ▲ raw MJPEG
                                          │     ▼                           │ (on-demand)
                                          │  ┌─────────────────────────┐    │
                                          │  │ FastAPI                  │    │
                                          │  │  GalleryMatcher (NumPy)  │    │
                                          │  │  match-or-create (locked)│    │
                                          │  │  Person/Camera/Sighting  │◄───┼── polling
                                          │  │  reaper (stale sightings)│    │  (1–2s)
                                          │  │  preview proxy ──────────┼────┘
                                          │  └──────────┬───────────────┘
                                          │             │ SQLite + thumbnails
                                          └─────────────┴─────────────┘
```

## Data model (SQLite / SQLModel)

```
Person
  id              int  PK
  label           str? optional human annotation (was "name"); nullable, editable
  best_sharpness  float
  created_at      datetime

FaceEmbedding            # gallery rows; loaded into the in-memory matrix at startup
  id          int PK
  person_id   int  FK -> person.id
  vector      bytes      # 512-d ArcFace float32, L2-normalized
  created_at  datetime

Camera
  id          int PK
  name        str
  location    str        # the "place" shown on the dashboard
  source      str        # "0" | "http://laptop:8090/stream" | "rtsp://…" | "demo/a.mp4"
  enabled     bool       # whether supervisor should keep a worker running
  created_at  datetime

Sighting                 # one continuous visit
  id          int PK
  person_id   int  FK -> person.id
  camera_id   int  FK -> camera.id
  started_at  datetime
  last_seen   datetime   # heartbeat; used by the reaper
  ended_at    datetime?  # null = currently visible
  best_sharpness float
```

## API surface

**Worker → API (ingestion):**
- `POST /sightings` — body: `{camera_id, track_key, embedding, crop, sharpness}`.
  Runs match-or-create (locked critical section) → returns `{person_id, sighting_id}`
  and opens the sighting. Saves/updates the person thumbnail if the crop is sharper.
- `POST /sightings/{id}/heartbeat` — bumps `last_seen` (and may attach a sharper crop).
- `POST /sightings/{id}/end` — sets `ended_at`.

**Cameras (admin + control):**
- `POST /cameras` · `GET /cameras` (with worker status) · `GET /cameras/{id}` ·
  `PATCH /cameras/{id}` · `DELETE /cameras/{id}`
- `POST /cameras/{id}/start` · `POST /cameras/{id}/stop` — command the supervisor.
- `GET /cameras/{id}/preview` — proxies the worker's raw MJPEG stream (on-demand).

**Dashboard (read):**
- `GET /people` — all anonymous persons (thumbnail, visit count, last seen).
- `GET /people/{id}` — person + full sightings timeline.
- `GET /people/{id}/thumbnail`
- `GET /sightings/active` — currently-open sightings (the Live/Now view).

**Search (read-only):**
- `POST /search` — body: image → `{person, sightings}` or `{match: null}`. No writes.

## Components & responsibilities

- **Shared core** (`pipeline/core.py` or similar): YOLO detector, IoU tracker, YuNet
  alignment, ArcFace embedder, threaded frame loop. No I/O sink, no GUI. Reused by both
  `pipeline_node.py` and `live_pipeline.py`.
- **`pipeline_node.py`** (headless worker): `--camera-id`, `--api-url`. Fetches its
  source from `GET /cameras/{id}`, opens it (loops if file), runs the core, calls the
  ingestion endpoints, and serves raw MJPEG preview on a local port (on-demand encode).
- **Supervisor** (module inside the API process): maps `camera_id → subprocess`;
  start/stop/status/restart; used by the camera control endpoints; tells the preview
  proxy which local port a camera's worker is on.
- **`GalleryMatcher`**: in-memory NumPy gallery; `match(embedding) -> person_id|None`,
  `add(person_id, embedding)`; rebuilt from SQLite at startup. Swappable for FAISS/etc.
- **Reaper**: background task closing open sightings whose `last_seen` is stale.
- **`publisher.py`** (runs on each laptop): webcam → MJPEG over HTTP.
- **Frontend** (`web/`): five pages, polling, preview `<img>` to the proxy endpoint.

## Error handling & edge cases

- **Duplicate-person race** → serialized match-or-create critical section (decision 4).
- **Worker crash mid-visit** → supervisor restarts the worker; reaper closes the
  orphaned open sighting (decision 8).
- **Dead/disconnected stream** → worker exits or stalls; supervisor marks camera
  stopped; reaper closes open sightings.
- **Preview with no viewer** → on-demand: worker encodes MJPEG only while a client is
  attached, so idle cameras cost nothing.
- **EOF on a file source** → reopen from start (loop); EOF on a live source → stop.

## Out of scope (explicit)

Authentication/roles; annotated (boxed) preview tab; per-machine worker agents
(decision 7B); vector DB; SSE/WebSocket push; multi-server/horizontal scaling;
re-identification tuning beyond the existing cosine threshold.

## Performance note

Every worker does its own YOLO + ArcFace on CPU. A single server running many
high-resolution streams will become CPU-bound — fine for a handful of cameras at
modest resolution / frame-skip; a GPU (or fewer/lower-res cameras) is the lever if it
can't keep up. ArcFace itself is light (~5–15 ms/face); YOLO detection across many
streams dominates.
