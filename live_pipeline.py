#!/usr/bin/env python3
"""Standalone local-debug viewer: live detection + background recognition.

This is the offline development tool. It drives the shared, headless
``pipeline.core`` (detection + IoU tracking + alignment + embedding) and matches
each embedding against a self-populating on-disk vector database
(:class:`FaceDatabase`): if it matches a stored face it is recognized; if nothing
matches (including the very first face), it is saved as a new identity. Results
are drawn back onto the live video as soon as they are ready, so the detection
loop never blocks on the (heavier) recognition model.

The production multi-camera path does not use this file — it uses the headless
``pipeline_node.py`` worker, which reports to the central API instead of writing
local files. This viewer remains handy for testing a webcam without the server.

No setup needed — the database fills itself as people appear. Run from the repo
root:
    python live_pipeline.py
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import torch

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))
sys.path.insert(0, str(_REPO_ROOT / "face_recognition" / "dino_vit"))
from recognizer import FaceRecognizer  # noqa: E402  (DINO/ViT baseline, in dino_vit/)
from arcface_onnx import ArcFaceONNXRecognizer  # noqa: E402
from face_db import FaceDatabase  # noqa: E402
from face_align import FaceProcessor  # noqa: E402

from pipeline.core import PipelineCore  # noqa: E402

DEFAULT_DETECTOR = _REPO_ROOT / "face_extraction" / "last.pt"
DEFAULT_RUN_DIR = _REPO_ROOT / "face_recognition" / "dino_vit" / "best_model"
DEFAULT_GALLERY = _REPO_ROOT / "face_recognition" / "dino_vit" / "gallery" / "gallery.pt"
DEFAULT_DB_DIR = _REPO_ROOT / "face_recognition" / "face_db"

WINDOW = "Surveillance — detection + recognition"


# ── Appearance (re-entry) counter — viewer-only ──────────────────────────────

class AppearanceCounter:
    """Counts how many times each identity *enters* the frame.

    A person who stays visible keeps the same track id and is counted once.
    When they leave, their track ages out of the tracker; when they return they
    get a new track id while keeping the same recognized identity -> +1. The
    tracker's max_age debounces brief detection dropouts so a blink doesn't
    double-count. Counts persist in the DB (db.meta[label]["appearances"]).
    """

    def __init__(self, db: FaceDatabase):
        self.db = db
        self.counts: Dict[str, int] = {
            label: meta.get("appearances", 0) for label, meta in db.meta.items()
        }
        self.present: Dict[str, set] = {}   # label -> live track ids showing it
        self.counted: set = set()           # track ids already counted

    def update(self, live: Dict[int, Optional[str]]) -> None:
        """live: current {track_id -> label or None} for on-screen faces."""
        for tid, label in live.items():
            if label is None or tid in self.counted:
                continue
            if not self.present.get(label):     # identity not currently on screen
                self.counts[label] = self.counts.get(label, 0) + 1
                self.db.meta.setdefault(label, {})["appearances"] = self.counts[label]
            self.present.setdefault(label, set()).add(tid)
            self.counted.add(tid)

        # Drop tracks that left the frame; when a label has none left it's "absent".
        live_ids = set(live)
        for label in list(self.present):
            dead = self.present[label] - live_ids
            for gone in dead:
                self.counted.discard(gone)
            self.present[label] -= dead
            if not self.present[label]:
                del self.present[label]

    def get(self, label: str) -> int:
        return self.counts.get(label, 0)


def main() -> int:
    p = argparse.ArgumentParser(description="Live detection + background recognition")
    p.add_argument("--recognizer", choices=["arcface", "dino"], default="arcface",
                   help="arcface = ArcFace ONNX (fast + pose-invariant); dino = the ViT")
    p.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--db-dir", default=None,
                   help="Vector DB directory (default: face_db/<recognizer>)")
    p.add_argument("--gallery", default=str(DEFAULT_GALLERY),
                   help="Optional gallery cache (.pt) from enroll.py to seed named people")
    p.add_argument("--threshold", type=float, default=None,
                   help="Min cosine similarity to count as the same person "
                        "(default: 0.30 arcface, 0.155 dino)")
    p.add_argument("--reset-db", action="store_true",
                   help="Start from an empty database (ignore any saved one)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--cam-width", type=int, default=640, help="Capture width")
    p.add_argument("--cam-height", type=int, default=480, help="Capture height")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO input size (smaller = much faster on CPU)")
    p.add_argument("--conf", type=float, default=0.3, help="Detector confidence")
    p.add_argument("--refresh-secs", type=float, default=4.0,
                   help="How often to re-recognize an already-identified tracked face")
    p.add_argument("--min-face", type=int, default=60,
                   help="Skip faces smaller than this (pixels) for recognition")
    p.add_argument("--save-every", type=int, default=50,
                   help="Persist the database every N recognitions")
    p.add_argument("--threads", type=int, default=None,
                   help="Torch CPU threads (default: torch's own choice)")
    # Alignment + quality gating (big recognition-quality lever).
    p.add_argument("--no-align", action="store_true",
                   help="Disable landmark-based face alignment")
    p.add_argument("--face-frac", type=float, default=0.5,
                   help="Fraction of the aligned crop the face fills (smaller=more margin)")
    p.add_argument("--margin", type=float, default=0.4,
                   help="Margin added around the detection box before alignment")
    p.add_argument("--min-sharpness", type=float, default=40.0,
                   help="Skip blurry faces below this Laplacian-variance sharpness")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"Torch CPU threads: {torch.get_num_threads()}")

    from ultralytics import YOLO

    detector_path = Path(args.detector)
    if not detector_path.exists():
        print(f"Detector not found: {detector_path}", file=sys.stderr)
        return 1

    print(f"Loading detector from {detector_path} ...")
    detector = YOLO(str(detector_path))

    # Backend-appropriate default threshold.
    threshold = args.threshold
    if threshold is None:
        threshold = 0.28 if args.recognizer == "arcface" else 0.155

    if args.recognizer == "arcface":
        print("Loading ArcFace ONNX recognizer ...")
        recognizer = ArcFaceONNXRecognizer(threshold=threshold)
        processor = FaceProcessor(template="arcface", min_sharpness=args.min_sharpness,
                                  yunet_path=None if not args.no_align else "")
    else:
        print(f"Loading DINO/ViT recognizer from {args.run_dir} ...")
        recognizer = FaceRecognizer(args.run_dir, threshold=threshold)
        processor = FaceProcessor(out_size=224, margin=args.margin,
                                  face_frac=args.face_frac,
                                  min_sharpness=args.min_sharpness,
                                  yunet_path=None if not args.no_align else "")
    if args.no_align:
        processor.detector = None
    aligning = processor.detector is not None
    print(f"Recognizer: {args.recognizer} (dim={recognizer.embed_dim}, thr={threshold}) | "
          f"alignment: {'ON' if aligning else 'OFF'} | min_sharpness={args.min_sharpness}")

    db_dir = Path(args.db_dir) if args.db_dir else DEFAULT_DB_DIR / args.recognizer
    db = FaceDatabase(threshold=threshold,
                      db_dir=None if args.reset_db else db_dir,
                      embed_dim=recognizer.embed_dim)
    db.db_dir = db_dir  # ensure saves land here even with --reset-db
    if len(db):
        print(f"Loaded vector DB with {len(db)} identities from {db_dir}")
    else:
        print(f"Starting with an empty vector DB (will fill as faces appear): {db_dir}")

    # Optionally seed with named people enrolled via enroll.py.
    gallery_path = Path(args.gallery)
    if gallery_path.exists() and len(db) == 0:
        data = torch.load(str(gallery_path), map_location="cpu", weights_only=False)
        db.seed_from_gallery(list(data["names"]), data["embeds"])
        print(f"Seeded DB with {len(data['names'])} named identities: "
              f"{', '.join(data['names'])}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}.", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    core = PipelineCore(detector, recognizer, processor, conf=args.conf,
                        imgsz=args.imgsz, refresh_secs=args.refresh_secs,
                        min_face=args.min_face)
    core.start()
    counter = AppearanceCounter(db)
    # track_id -> (label, score, is_new); drives what we draw across frames.
    track_labels: Dict[int, Tuple[str, float, bool]] = {}
    recog_count = 0

    print("Running. Press 'q' to quit.")
    fps_t0, fps_n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed; stopping.", file=sys.stderr)
                break

            boxes, track_ids, events = core.step(frame)

            # Match each finished embedding against the local DB (main thread).
            for ev in events:
                label, score, is_new = db.match_or_add(ev.embedding, ev.crop)
                if is_new:
                    print(f"[db] new identity enrolled: {label} "
                          f"(best prior score {score:.2f})")
                track_labels[ev.track_id] = (label, score, is_new)
                recog_count += 1

            # Update the appearance (re-entry) counter from current tracks.
            live: Dict[int, Optional[str]] = {
                tid: (track_labels[tid][0] if tid in track_labels else None)
                for tid in track_ids
            }
            counter.update(live)

            display = frame.copy()
            for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
                res = track_labels.get(tid)
                if res is None:
                    label, color = "...", (0, 200, 255)          # still processing
                else:
                    name, score, is_new = res
                    label = f"{name} #{counter.get(name)} {score:.2f}"
                    color = (0, 165, 255) if is_new else (0, 255, 0)  # orange=new, green=known

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Persist the growing database periodically.
            if recog_count // max(args.save_every, 1) > 0 and recog_count % args.save_every == 0:
                db.save()

            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0
            cv2.putText(display, f"FPS {fps:.1f}  faces {len(boxes)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(WINDOW, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        core.stop()
        cap.release()
        cv2.destroyAllWindows()
        db.save()
        print(f"Saved vector DB ({len(db)} identities) to {db.db_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
