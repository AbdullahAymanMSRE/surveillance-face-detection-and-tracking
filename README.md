# Surveillance: Multi-Camera Face Detection, Recognition & Tracking

A surveillance system that connects to **many cameras**, runs each through a
**detection → recognition → tracking** pipeline, and presents everything on a
**web dashboard**: the anonymous people it has seen (no names needed), and for
each one **when** and **at which camera/place** they appeared.

People are discovered automatically — `person_001`, `person_002`, … — and matched
**across cameras**, so the same person seen at two cameras is one identity with a
combined timeline.

> CV pipeline internals: see **[PIPELINE.md](PIPELINE.md)**.
> Design decisions & build plan: see **[docs/superpowers/](docs/superpowers/)**.

---

## Architecture

Three parts, connected over HTTP around one source of truth (the API's database):

```
 CAMERAS (anywhere)                 SERVER (one machine)                 BROWSER
 ─────────────────                  ────────────────────                ───────
 laptop webcam ─ publisher.py ─MJPEG┐
 IP camera ────────────────────RTSP─┤ source  ┌──────────────────────┐
 video file ───────────────────path─┘────────►│ Supervisor           │
                                               │  one worker / camera │
                                               │   detect→track→embed │
                                               │      │ embedding      │
                                               │      ▼                │
                                               │  FastAPI              │
                                               │   central matching    │◄── polling ── Dashboard
                                               │   Person/Camera/      │    (1–2s)     (Next.js)
                                               │   Sighting (SQLite)    │
                                               │   + MJPEG preview proxy│──── preview ──►
                                               └───────────────────────┘
```

- **Workers run on the server**, one per camera, spawned and supervised by the
  API. A remote laptop contributes its webcam by running `publisher.py` (an MJPEG
  stream the server pulls); real IP cameras use their `rtsp://` URL directly.
- **Matching is central**: each worker sends embeddings to the API, which assigns
  identities against one shared gallery — that's what makes cross-camera identity
  work.
- **One visit = one `Sighting`** (start / heartbeat / end); a background reaper
  closes visits orphaned by a crashed worker.

---

## Quick start (no hardware needed)

```bash
make install                       # venv + Python deps + web deps  (one time)
```

Then in two terminals (from the repo root):

```bash
make api                           # FastAPI backend + supervisor on :8000
make web                           # Next.js dashboard on :3000
```

Seed a hardware-free demo (faces come from the test fixtures):

```bash
make demo-clips                    # build demo/lobby.mp4 and demo/lab.mp4
make demo-cameras                  # register two cameras (API must be running)
```

Open **http://localhost:3000** — within a few seconds the workers detect the
demo faces and the dashboard fills in.

> First start is slower: each worker loads the YOLO + ArcFace models.

---

## Running with real cameras

**A laptop webcam** — on the laptop (needs only `pip install opencv-python`):

```bash
python publisher.py --source 0 --port 8090
```

Then on the dashboard **Cameras** page, add a camera with source
`http://<laptop-ip>:8090/stream` and press **Start**.

**An IP / CCTV camera** — add a camera whose source is its stream URL, e.g.
`rtsp://user:pass@192.168.1.50:554/stream1`. No extra software; the worker opens
it the same way.

Add as many as the server's CPU can handle. The same person in front of two
cameras shows up as a single identity with a merged timeline.

---

## The dashboard

| Screen | What it shows |
|--------|----------------|
| **Live** (`/`) | Who is currently in view, grouped by camera (updates every ~1–2s). |
| **People** (`/people`) | Every anonymous identity with visit count, last-seen, in-view flag. |
| **Person** (`/people/[id]`) | The appearance timeline — each visit's camera, place and time — plus an optional editable label. |
| **Cameras** (`/cameras`) | Add cameras, start/stop their workers, see status, and watch a **live preview**. |
| **Search** (`/search`) | Upload a face to check whether it's on record (read-only — nothing is saved). |

---

## How it works

1. **Detect** — each worker runs the YOLO face detector and an IoU tracker for
   stable per-face track ids (shared core in `pipeline/core.py`).
2. **Align + embed** — faces are aligned (YuNet landmarks) and turned into 512-d
   ArcFace embeddings; blurry frames are skipped.
3. **Report** — on first recognizing a track the worker opens a sighting
   (`POST /sightings`); it heartbeats while the face stays, and ends the sighting
   when the track drops.
4. **Match centrally** — the API matches each embedding against the in-memory
   gallery (cosine ≥ threshold). A hit reuses that person; otherwise a new
   anonymous `person_NNN` is created — inside a lock so two cameras can't create
   duplicates for the same new face.
5. **Show** — the dashboard polls the API for live state, per-person timelines,
   and camera status, and streams on-demand previews proxied from the workers.

---

## Development

```bash
make test          # API test suite (pytest)
make api           # backend with the supervisor
make web           # dashboard (set API_URL=... to point elsewhere)
```

Key environment variables: `FACE_API_DATA_DIR` (SQLite + thumbnails location),
`FACE_API_SELF_URL` (URL workers use to reach the API), `NEXT_PUBLIC_API_URL`
(dashboard → API base URL; see `web/.env.local.example`).

The standalone local-debug viewer (single webcam, on-screen window, no server)
is still available:

```bash
python live_pipeline.py
```

---

## Future work

Authentication/roles; an annotated (boxes + labels) preview tab; a vector-DB
implementation of the `GalleryMatcher` interface for very large galleries;
server-push (SSE) instead of polling; containerized deployment.

---

## Notes & limitations

- **CPU-bound at scale** — every worker runs YOLO + ArcFace; many high-res
  streams on one machine will saturate CPU. Use fewer/lower-res cameras or a GPU.
- **Threshold tuning** — if two people merge into one identity, raise the match
  threshold; if one person splits into several, lower it.
- The DINO/ViT recognizer is retained as a selectable baseline for the standalone
  viewer (`python live_pipeline.py --recognizer dino`); ArcFace is the default.
