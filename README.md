# Surveillance: Face Detection, Recognition & Tracking

A real-time pipeline that **detects** faces from a webcam (YOLO), **recognizes**
who they are (ArcFace), and **tracks how many times each person appears** — all
running smoothly on a CPU.

It auto-enrolls people as they appear (no manual setup), keeps the same person as
one identity across poses/lighting, and counts each person's visits (re-entries
into the frame).

> Detailed reference: see **[PIPELINE.md](PIPELINE.md)**.

---

## What's in this version (updates)

This builds an end-to-end live system on top of the original detector + ViT
training code. Highlights:

- **Detection → recognition pipeline** (`live_pipeline.py`) wiring YOLO to a face
  recognizer, with both running on **background threads** so the video stays smooth.
- **ArcFace (ONNX Runtime) recognizer** as the default — fast (~5–15 ms/face on
  CPU) and pose-invariant. The original **DINO/ViT** model is kept as an optional
  baseline (`--recognizer dino`).
- **Self-populating vector database** — first time a face is seen it's enrolled;
  later frames match it. No pre-registration needed.
- **Face alignment + quality gating** (YuNet landmarks) so embeddings are stable.
- **Appearance (visit) counter** — increments only when a person leaves the frame
  and returns; staying visible doesn't increment it.
- **Best-shot thumbnails** — each person's thumbnail is automatically their
  sharpest crop, and every crop from every run is archived.
- **`stats.py`** — prints visits/sightings per identity.

### Why ArcFace replaced the ViT for recognition

Measured on crops of the *same person* in different poses:

| | DINO/ViT | ArcFace ONNX |
|---|---|---|
| Same-person similarity across poses | ~0.0 (splits into many IDs) | **+0.3 to +0.6** (one ID) |
| Speed on CPU | ~500 ms/face | **~5–15 ms/face** |

The ViT (`face_recognition/dino_vit/`) is retained as a selectable baseline.

---

## Architecture

```
 webcam ──► [main thread] capture + draw + display        (~30 fps, smooth)
              │ submits latest frame              ▲ overlays boxes + labels
              ▼                                   │
       [DetectionWorker]  YOLO @ imgsz 320  ──► boxes + IoU track ids
              │ aligned face crops (YuNet 112x112)
              ▼
       [RecognitionWorker]  ArcFace ONNX embedding
              │
              ▼
       FaceDatabase.match_or_add
         ├─ cosine ≥ threshold ─► existing identity (running-mean update)
         └─ else                ─► enroll NEW identity (person_NNN)
              │
              ▼
       AppearanceCounter (visits) + sharpest-thumbnail + sighting archive
```

Three threads keep the heavy work off the display loop: capture/draw (main),
detection (YOLO), and recognition (ArcFace).

---

## Repository layout

```
live_pipeline.py            # the real-time app (run this)
stats.py                    # print DB summary (visits, sightings, seen times)
requirements.txt
PIPELINE.md                 # detailed docs

face_extraction/
    last.pt                 # trained YOLO face detector
    live_face_extraction.py # standalone detection demo

face_recognition/           # ACTIVE recognition system (ArcFace)
    arcface_onnx.py         # ArcFace recognizer (ONNX Runtime, CPU)
    face_align.py           # YuNet alignment + sharpness gating
    face_db.py              # self-populating vector database
    models/
        w600k_mbf.onnx                       # ArcFace model
        face_detection_yunet_2023mar.onnx    # landmark model
    face_db/                # runtime DB (git-ignored; created on first run)

    dino_vit/               # DINO/ViT BASELINE (optional, --recognizer dino)
        recognizer.py       # ViT recognizer
        dino_vggface2.py    # training script
        eval_dino_vggface2.py
        enroll.py           # gallery enrollment for the ViT
        best_model/         # trained ViT checkpoint (Git LFS)
        *.ipynb, submit_*.sh
```

---

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, a webcam, and `numpy < 2` (pinned — ultralytics crashes on
numpy 2). The ArcFace and YuNet model files are included under
`face_recognition/models/`. The ViT checkpoint is stored via **Git LFS**; run
`git lfs pull` if you need the `--recognizer dino` baseline.

---

## How to run

```bash
python live_pipeline.py
```

That's it — no enrollment needed. People are auto-enrolled and recognized live.

In the window:
- **Green** box = recognized existing person.
- **Orange** box = just enrolled as a new person.
- **Yellow `...`** = recognition still computing for that face.
- Label format: `person_001 #3 0.45` = **identity**, **visit count**, cosine score.

Press **`q`** to quit (saves the database).

Use the original ViT instead:
```bash
python live_pipeline.py --recognizer dino
```

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--recognizer` | `arcface` | `arcface` (fast+accurate) or `dino` (the ViT baseline) |
| `--threshold` | `0.28` / `0.155` | Same-person cosine threshold (arcface / dino) |
| `--imgsz` | `320` | YOLO input size (smaller = faster) |
| `--cam-width` / `--cam-height` | `640` / `480` | Capture resolution |
| `--min-sharpness` | `40` | Skip blurry faces below this sharpness |
| `--refresh-secs` | `4` | How often to re-recognize a tracked face |
| `--reset-db` | off | Start from an empty database |

---

## How it works

1. **Detection** — YOLO finds face boxes (background thread, `imgsz 320`). A simple
   IoU tracker assigns a stable id to each face across frames.
2. **Alignment** — YuNet detects 5 landmarks; the face is warped to a canonical
   112×112 pose. Blurry frames are skipped.
3. **Recognition** — ArcFace turns the aligned crop into a 512-d embedding
   (background thread).
4. **Matching / enrollment** — the embedding is compared (cosine) to every stored
   identity. Above threshold → that person (their stored embedding is averaged in);
   otherwise a new `person_NNN` is enrolled.
5. **Counting** — a person staying in frame keeps one track id (counted once). When
   they leave (track ages out after ~1.5 s) and return, they get a new track id but
   the same identity → visit count +1.
6. **Storage** — each person's **sharpest** crop becomes their thumbnail; **every**
   crop is archived; counts/metadata persist across runs.

### Database output (`face_recognition/face_db/<backend>/`)

```
embeddings.pt     # identities + embeddings + counts
metadata.json     # per-identity: appearances (visits), sightings, first/last seen
thumbnails/       # sharpest crop per identity
sightings/        # every crop, grouped by identity, kept across runs
```

This folder is git-ignored (runtime + personal data).

---

## Stats

```bash
python stats.py                 # visits / sightings / first-last seen, per person
python stats.py --sort recent   # also: visits, sightings, name
python stats.py --recognizer dino
```

---

## Notes & limitations

- **CPU only here** — works fine; a CUDA GPU would speed detection/recognition
  further but isn't required.
- **Threshold tuning** — if two different people merge into one identity, raise
  `--threshold`; if one person splits, lower it.
- The ViT baseline is kept for reference; ArcFace is recommended for real use.
