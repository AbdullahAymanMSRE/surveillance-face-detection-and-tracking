# Detection → Recognition pipeline

Connects the two models so faces found by the **YOLO detector** are recognized by
the **DINO + ArcFace** model. Detection runs in real time on the main thread;
recognition runs in a **background worker thread**, so the live video never stalls
on the heavier recognition model.

Recognition uses a **self-populating vector database**: each new face embedding is
compared against everything stored. If it matches a stored face it is recognized;
if nothing matches (including the very first face ever seen), it is saved as a new
identity (`person_001`, `person_002`, …). No pre-enrollment is required — the
database fills itself as people appear, and persists between runs.

```
 webcam ─► YOLO detector (main thread, real-time)
              │ face crops + stable track ids (IoU tracker)
              ▼
          queue ─► RecognitionWorker (background thread)
                       │ DINO embedding
                       ▼
                FaceDatabase.match_or_add
                  ├─ best cosine sim >= threshold ─► existing identity (update mean)
                  └─ else                          ─► store as NEW identity
                       ▼
                results ─► drawn back onto the live frame
```

## Files added

| File | Purpose |
|------|---------|
| `face_recognition/arcface_onnx.py` | **Default recognizer** — ArcFace (MobileFaceNet, 512-d) on ONNX Runtime CPU. Fast (~5-15 ms/face) and pose-invariant. |
| `face_recognition/models/w600k_mbf.onnx` | ArcFace model (~14 MB, from InsightFace buffalo_s). |
| `face_recognition/face_align.py` | `FaceProcessor` — landmark-based face alignment (YuNet) + sharpness quality gate. Big recognition-quality lever. |
| `face_recognition/models/face_detection_yunet_2023mar.onnx` | YuNet landmark model (~230 KB) used for alignment. |
| `face_recognition/face_db.py` | `FaceDatabase` — the self-populating vector DB: match-or-add, running-mean update, persistence + thumbnails. |
| `face_recognition/dino_vit/` | The DINO/ViT **baseline** (kept, selectable with `--recognizer dino`): `recognizer.py`, training/eval scripts, `enroll.py`, `best_model/` checkpoint, notebook. Superseded by ArcFace; see below. |
| `live_pipeline.py` | The real-time pipeline: detection + threaded background recognition against the vector DB, with an overlay. |
| `stats.py` | Prints the DB summary: visits, sightings, first/last seen per identity. |

## 1. Install

```bash
pip install -r requirements.txt
```

The first run downloads the DINO backbone architecture from `torch.hub`
(`facebookresearch/dino`), so internet access is needed once.

## 2. Run the live pipeline

```bash
python live_pipeline.py                 # default: ArcFace ONNX recognizer
python live_pipeline.py --recognizer dino   # use the DINO/ViT model instead
```

That's it — no enrollment step needed. As faces appear they are auto-enrolled and
then recognized on later frames.

### Recognizer backends

| Backend | Speed (CPU) | Cross-pose accuracy | Notes |
|---------|-------------|---------------------|-------|
| `arcface` (default) | ~5-15 ms/face | strong | ArcFace ONNX; same person stays one identity across poses |
| `dino` | ~500 ms/face | weak | the trained ViT; kept for reference (`--recognizer dino`) |

Measured on the same person across poses, cross-pose cosine similarity was ~0.0
with the ViT (splits into many identities) vs **+0.31 to +0.63 with ArcFace**
(stays a single identity). Each backend keeps its own DB under
`face_db/<backend>/` (the embedding spaces are not interchangeable).

- **Orange** box = a face just enrolled as a *new* identity.
- **Green** box = recognized as an *existing* identity.
- **Yellow `...`** = recognition still in flight for that face.

Labels read `person_001 #3 0.45` = identity, **visit count**, cosine score. The
visit count increments only when that person **leaves the frame and returns** —
staying continuously visible does not increment it (a ~1.5 s absence, set by the
tracker's `max_age`, debounces brief detection dropouts). Counts persist in the
DB (`metadata.json` → `appearances`) across runs.

The database is written to `face_recognition/face_db/`:

```
face_db/
    embeddings.pt    # labels + embeddings + sighting counts
    metadata.json    # first/last seen + sighting count per identity
    thumbnails/      # the SHARPEST crop per identity (best across all runs)
    sightings/       # EVERY face crop, grouped by identity, kept across all runs
        person_001/  20260618_143701_000000000.jpg ...
        person_002/  ...
```

Every detected face from every run is archived under `sightings/<label>/`
(`<datetime>_<seq>.jpg`); these accumulate across runs and are never overwritten.

Press `q` to quit (the DB is saved on exit and every `--save-every` recognitions).

## 3. (Optional) Seed with named people

If you want real names instead of `person_001`, enroll people first; the pipeline
seeds the DB from the gallery on a fresh start. **Note:** `enroll.py` lives in
`dino_vit/` and builds a DINO (256-d) gallery, so it applies to the
`--recognizer dino` baseline.

**From the webcam** (uses YOLO to auto-crop, SPACE to grab, `q` when done):

```bash
cd face_recognition/dino_vit
python enroll.py --webcam --name alice --num-shots 10
```

**From a folder** organized one subfolder per person:

```
face_recognition/dino_vit/gallery/
    alice/  img1.jpg img2.jpg ...
    bob/    img1.jpg ...
```

```bash
cd face_recognition/dino_vit
python enroll.py --gallery-dir gallery
```

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--threshold` | `0.155` | Min cosine similarity to count as the same person. **Raise it** (e.g. `0.3–0.5`) if distinct people get merged into one identity; lower it if one person spawns many identities. |
| `--reset-db` | off | Start from an empty database (ignore the saved one). |
| `--db-dir` | `face_recognition/face_db` | Where the vector DB lives. |
| `--conf` | `0.3` | YOLO detection confidence. |
| `--refresh-secs` | `2.0` | How often a tracked face is re-recognized. |
| `--min-face` | `40` | Skip faces smaller than this (px) for recognition. |
| `--recognizer` | `arcface` | `arcface` (fast+accurate ONNX) or `dino` (the ViT). |
| `--threshold` | `0.28`/`0.155` | Same-person cosine threshold (arcface/dino defaults). |
| `--detect-every` | `2` | Run YOLO every N frames; ≥2 keeps the video smoother. |
| `--cam-width`/`--cam-height` | `640`/`480` | Capture resolution (lower = smoother). |
| `--save-every` | `50` | Persist the DB every N recognitions. |
| `--camera-index` | `0` | Which camera to open. |
| `--face-frac` | `0.5` | Fraction of the aligned crop the face fills (smaller = more margin). |
| `--margin` | `0.4` | Margin added around the detection box before alignment. |
| `--min-sharpness` | `40` | Skip blurry faces below this Laplacian-variance sharpness. |
| `--no-align` | off | Disable landmark alignment (fall back to square crops). |

## Face alignment (recognition quality)

Raw YOLO boxes vary in framing, scale and tilt, which made the recognition model
give the *same person* near-zero similarity (so it enrolled them as several
different identities). Each face is now normalized before embedding:

1. **YuNet** detects 5 landmarks (eyes, nose, mouth corners).
2. A similarity transform warps the face to a canonical, eyes-level pose at a
   consistent scale with margin (`--face-frac`), matching the model's training
   framing. Falls back to a square padded crop if no landmarks are found.
3. A **sharpness gate** (`--min-sharpness`) skips blurry frames.

Measured on three crops of the same person: average pairwise cosine similarity
rose from **0.04 (unaligned)** to **0.35 (aligned)** — i.e. from "looks like 3
strangers" to comfortably above the 0.155 same-person threshold.

## Notes

- The recognizer preprocesses crops exactly like `dino_vggface2.ImageValData`
  (resize 224, ImageNet normalize), so embeddings match the evaluated model.
- The DB is a simple cosine-similarity store (torch tensor of L2-normalized
  embeddings), equivalent to a FAISS `IndexFlatIP` but with no extra dependency.
  Each recognized hit blends into that identity's running-mean embedding.
- The default threshold `0.155` is the model's verification operating point. For
  **auto-enrollment** it is permissive — if you see different people merged,
  raise `--threshold`.
- Recognition is decoupled from detection by a bounded queue: if recognition
  falls behind, new crops are dropped rather than queued, keeping detection
  real-time. On CPU recognition is slow; a CUDA GPU gives snappier labels.
