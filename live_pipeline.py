#!/usr/bin/env python3
"""Real-time face detection + background face recognition pipeline.

The YOLO detector runs in the main thread at camera frame rate. Detected face
crops are handed to a background worker thread that runs the DINO+ArcFace
recognizer. Each embedding is matched against a self-populating vector database
(FaceDatabase): if it matches a stored face it is recognized; if nothing matches
(including the very first face), it is saved as a new identity. Results are drawn
back onto the live video as soon as they are ready, so the detection loop never
blocks on the (heavier) recognition model.

No setup needed — the database fills itself as people appear. Optionally seed it
with named people first:
    cd face_recognition
    python enroll.py --webcam --name alice

Run the live pipeline from the repo root:
    python live_pipeline.py
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))
sys.path.insert(0, str(_REPO_ROOT / "face_recognition" / "dino_vit"))
from recognizer import FaceRecognizer  # noqa: E402  (DINO/ViT baseline, in dino_vit/)
from arcface_onnx import ArcFaceONNXRecognizer  # noqa: E402
from face_db import FaceDatabase  # noqa: E402
from face_align import FaceProcessor  # noqa: E402

DEFAULT_DETECTOR = _REPO_ROOT / "face_extraction" / "last.pt"
DEFAULT_RUN_DIR = _REPO_ROOT / "face_recognition" / "dino_vit" / "best_model"
DEFAULT_GALLERY = _REPO_ROOT / "face_recognition" / "dino_vit" / "gallery" / "gallery.pt"
DEFAULT_DB_DIR = _REPO_ROOT / "face_recognition" / "face_db"

WINDOW = "Surveillance — detection + recognition"


# ── Minimal IoU tracker (gives faces stable ids across frames) ────────────────

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}  # id -> {"box", "age"}

    def update(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """Assign a track id to each detection box (same order as input)."""
        for tr in self.tracks.values():
            tr["age"] += 1

        assigned: List[int] = [-1] * len(boxes)
        used_tracks = set()
        for i, box in enumerate(boxes):
            best_id, best_iou = -1, self.iou_threshold
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                score = _iou(box, tr["box"])
                if score >= best_iou:
                    best_id, best_iou = tid, score
            if best_id >= 0:
                assigned[i] = best_id
                used_tracks.add(best_id)
                self.tracks[best_id] = {"box": box, "age": 0}
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "age": 0}
                assigned[i] = tid

        # Drop stale tracks.
        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]
        return assigned


# ── Background recognition worker ─────────────────────────────────────────────

class RecognitionWorker(threading.Thread):
    def __init__(self, recognizer: FaceRecognizer, db: FaceDatabase,
                 max_queue: int = 8, max_batch: int = 8):
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.db = db
        self.max_batch = max_batch
        self.queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(max_queue)
        # track_id -> (label, score, ts, is_new)
        self.results: Dict[int, Tuple[str, float, float, bool]] = {}
        self.inflight: set = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit(self, track_id: int, crop: np.ndarray) -> None:
        """Queue a crop for recognition; drop it if the worker is busy (stay real-time)."""
        with self._lock:
            if track_id in self.inflight:
                return
            self.inflight.add(track_id)
        try:
            self.queue.put_nowait((track_id, crop))
        except queue.Full:
            with self._lock:
                self.inflight.discard(track_id)

    def get_result(self, track_id: int) -> Optional[Tuple[str, float, bool]]:
        with self._lock:
            r = self.results.get(track_id)
        return (r[0], r[1], r[3]) if r else None

    def needs_recognition(self, track_id: int, refresh_secs: float) -> bool:
        with self._lock:
            if track_id in self.inflight:
                return False
            r = self.results.get(track_id)
        return r is None or (time.time() - r[2]) > refresh_secs

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            # Drain whatever else is queued so the (heavy) ViT forward pass runs
            # on a batch of faces at once instead of one at a time.
            batch = [first]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break

            track_ids = [b[0] for b in batch]
            crops = [b[1] for b in batch]
            try:
                embeds = self.recognizer.embed_batch(crops)  # [B, embed_dim]
                now = time.time()
                for track_id, emb, crop in zip(track_ids, embeds, crops):
                    # Compare against the vector DB; enroll a new identity if none match.
                    label, score, is_new = self.db.match_or_add(emb, crop)
                    if is_new:
                        print(f"[db] new identity enrolled: {label} "
                              f"(best prior score {score:.2f})")
                    with self._lock:
                        self.results[track_id] = (label, score, now, is_new)
            finally:
                with self._lock:
                    for track_id in track_ids:
                        self.inflight.discard(track_id)

    def stop(self) -> None:
        self._stop.set()


# ── Background detection worker (keeps YOLO off the display thread) ────────────

class DetectionWorker(threading.Thread):
    """Runs YOLO on the most recent frame in the background, so the capture/
    display loop runs at camera speed instead of blocking on detection."""

    def __init__(self, detector, tracker: "IoUTracker", conf: float, imgsz: int):
        super().__init__(daemon=True)
        self.detector = detector
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz
        self._frame = None
        self._boxes: List[Tuple[int, int, int, int]] = []
        self._track_ids: List[int] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame  # keep only the latest; older frames are dropped

    def get_boxes(self) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        with self._lock:
            return list(self._boxes), list(self._track_ids)

    def run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.003)
                continue
            results = self.detector.predict(frame, conf=self.conf,
                                            imgsz=self.imgsz, verbose=False)
            h, w = frame.shape[:2]
            boxes: List[Tuple[int, int, int, int]] = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w), min(y2, h)
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
            track_ids = self.tracker.update(boxes)
            with self._lock:
                self._boxes, self._track_ids = boxes, track_ids

    def stop(self) -> None:
        self._stop.set()


# ── Appearance (re-entry) counter ─────────────────────────────────────────────

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


# ── Main loop ─────────────────────────────────────────────────────────────────

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

    worker = RecognitionWorker(recognizer, db)
    worker.start()
    tracker = IoUTracker()
    detector_worker = DetectionWorker(detector, tracker, args.conf, args.imgsz)
    detector_worker.start()
    counter = AppearanceCounter(db)
    recog_count = 0

    print("Running. Press 'q' to quit.")
    fps_t0, fps_n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed; stopping.", file=sys.stderr)
                break

            # Detection runs in the background; we just use its latest boxes.
            detector_worker.submit_frame(frame)
            boxes, track_ids = detector_worker.get_boxes()

            # Update the appearance (re-entry) counter from current tracks.
            live = {tid: (r[0] if (r := worker.get_result(tid)) else None)
                    for tid in track_ids}
            counter.update(live)

            display = frame.copy()
            for (x1, y1, x2, y2), tid in zip(boxes, track_ids):
                # Hand off to the recognition worker (non-blocking).
                if (min(x2 - x1, y2 - y1) >= args.min_face
                        and worker.needs_recognition(tid, args.refresh_secs)):
                    # Align + quality-gate before recognition.
                    crop, sharpness, _ = processor.process(frame, (x1, y1, x2, y2))
                    if processor.is_sharp(sharpness):
                        worker.submit(tid, crop)

                res = worker.get_result(tid)
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
            with worker._lock:
                done = len(worker.results)
            if done - recog_count >= args.save_every:
                recog_count = done
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
        worker.stop()
        detector_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        db.save()
        print(f"Saved vector DB ({len(db)} identities) to {db.db_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
