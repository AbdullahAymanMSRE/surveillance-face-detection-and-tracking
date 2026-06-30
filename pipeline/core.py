"""Shared, headless face pipeline core: detect -> track -> align -> embed.

This module holds the recognition machinery that is common to both consumers:

  * ``live_pipeline.py`` — the standalone local-debug viewer, which matches each
    embedding against its own on-disk :class:`FaceDatabase` and draws a window.
  * ``pipeline_node.py`` — the headless camera worker, which reports each
    embedding to the central API for matching and sighting tracking.

The core deliberately has **no data sink and no GUI**. It runs detection,
IoU tracking, alignment and embedding on background threads, and the consumer
drives it one frame at a time via :meth:`PipelineCore.step`, deciding what to do
with the embeddings it produces.
"""

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))
from arcface_onnx import ArcFaceONNXRecognizer  # noqa: E402
from face_align import FaceProcessor  # noqa: E402

DEFAULT_DETECTOR = _REPO_ROOT / "face_extraction" / "last.pt"

Box = Tuple[int, int, int, int]


def _black_border_frac(crop: np.ndarray, thresh: int = 12) -> float:
    """Fraction of (near-)black pixels in a crop.

    Landmark warping fills out-of-image areas with black, so a partial or
    frame-edge face produces an aligned crop with a large black wedge. A high
    fraction means the alignment had little real face to work with -> skip it."""
    if crop.size == 0:
        return 1.0
    return float((crop.max(axis=2) < thresh).mean())


# ── Minimal IoU tracker (gives faces stable ids across frames) ────────────────

def _iou(a: Box, b: Box) -> float:
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

    def update(self, boxes: List[Box]) -> List[int]:
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


# ── Recognition event (what the core hands back to its consumer) ──────────────

@dataclass
class RecognitionEvent:
    """A completed embedding for one tracked face, ready for a sink to consume."""
    track_id: int
    box: Box
    embedding: torch.Tensor   # 1-D, L2-normalized, CPU (matches recognizer output)
    crop: np.ndarray          # aligned BGR crop the embedding was computed from
    sharpness: float
    ts: float


# ── Background detection worker (keeps YOLO off the caller's thread) ──────────

class DetectionWorker(threading.Thread):
    """Runs YOLO on the most recent frame in the background, so the caller's
    capture loop runs at stream speed instead of blocking on detection."""

    def __init__(self, detector, tracker: IoUTracker, conf: float, imgsz: int):
        super().__init__(daemon=True)
        self.detector = detector
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz
        self._frame = None
        self._boxes: List[Box] = []
        self._track_ids: List[int] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame  # keep only the latest; older frames are dropped

    def get_boxes(self) -> Tuple[List[Box], List[int]]:
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
            boxes: List[Box] = []
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


# ── Background embedding worker ───────────────────────────────────────────────

class EmbeddingWorker(threading.Thread):
    """Embeds aligned face crops off the caller's thread.

    Unlike a recognition worker, this performs **no matching** — it only turns
    crops into embeddings and hands them back as :class:`RecognitionEvent`s. The
    consumer decides how to match them (local DB vs. central API). Crops are
    batched so the embedding model runs on several faces at once; submissions are
    dropped when the worker is busy so the pipeline stays real-time.
    """

    def __init__(self, recognizer, max_queue: int = 8, max_batch: int = 8):
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.max_batch = max_batch
        # queued work: (track_id, box, crop, sharpness)
        self.queue: "queue.Queue[Tuple[int, Box, np.ndarray, float]]" = queue.Queue(max_queue)
        self._last_ts: Dict[int, float] = {}   # track_id -> last completed embed ts
        self._inflight: set = set()
        self._new: List[RecognitionEvent] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def needs_recognition(self, track_id: int, refresh_secs: float) -> bool:
        with self._lock:
            if track_id in self._inflight:
                return False
            ts = self._last_ts.get(track_id)
        return ts is None or (time.time() - ts) > refresh_secs

    def submit(self, track_id: int, box: Box, crop: np.ndarray, sharpness: float) -> None:
        """Queue a crop for embedding; drop it if the worker is busy (stay real-time)."""
        with self._lock:
            if track_id in self._inflight:
                return
            self._inflight.add(track_id)
        try:
            self.queue.put_nowait((track_id, box, crop, sharpness))
        except queue.Full:
            with self._lock:
                self._inflight.discard(track_id)

    def drain_new(self) -> List[RecognitionEvent]:
        """Return (and clear) embeddings completed since the last call."""
        with self._lock:
            out, self._new = self._new, []
        return out

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            # Drain whatever else is queued so the embedding forward pass runs on
            # a batch of faces at once instead of one at a time.
            batch = [first]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break

            track_ids = [b[0] for b in batch]
            crops = [b[2] for b in batch]
            try:
                embeds = self.recognizer.embed_batch(crops)  # [B, embed_dim]
                now = time.time()
                events = [
                    RecognitionEvent(track_id=tid, box=box, embedding=emb.clone(),
                                     crop=crop, sharpness=sharp, ts=now)
                    for (tid, box, crop, sharp), emb in zip(batch, embeds)
                ]
                with self._lock:
                    for tid in track_ids:
                        self._last_ts[tid] = now
                    self._new.extend(events)
            finally:
                with self._lock:
                    for tid in track_ids:
                        self._inflight.discard(tid)

    def stop(self) -> None:
        self._stop.set()


# ── The core: detection + tracking + alignment + embedding ───────────────────

class PipelineCore:
    """Drives detection, tracking, alignment and embedding for one stream.

    The consumer calls :meth:`step` once per captured frame and receives the
    current boxes/track ids plus any embeddings that finished since the last
    call. Matching and persistence are entirely the consumer's responsibility.
    """

    def __init__(self, detector, recognizer, processor: FaceProcessor, *,
                 conf: float = 0.3, imgsz: int = 320, refresh_secs: float = 4.0,
                 min_face: int = 60, max_black_frac: float = 0.30,
                 require_landmarks: bool = True):
        self.processor = processor
        self.refresh_secs = refresh_secs
        self.min_face = min_face
        self.max_black_frac = max_black_frac
        # Only embed when YuNet actually finds facial landmarks in the box. This
        # rejects the detector's non-face false positives (a shoulder, the back
        # of a head) that YOLO boxed but that contain no real face.
        self.require_landmarks = require_landmarks and (processor.detector is not None)
        self.tracker = IoUTracker()
        self.det_worker = DetectionWorker(detector, self.tracker, conf, imgsz)
        self.embed_worker = EmbeddingWorker(recognizer)

    def start(self) -> None:
        self.det_worker.start()
        self.embed_worker.start()

    def step(self, frame: np.ndarray
             ) -> Tuple[List[Box], List[int], List[RecognitionEvent]]:
        """Process one frame; return (boxes, track_ids, new_events)."""
        self.det_worker.submit_frame(frame)
        boxes, track_ids = self.det_worker.get_boxes()
        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            if (min(x2 - x1, y2 - y1) >= self.min_face
                    and self.embed_worker.needs_recognition(tid, self.refresh_secs)):
                crop, sharpness, aligned = self.processor.process(frame, box)
                if ((aligned or not self.require_landmarks)
                        and self.processor.is_sharp(sharpness)
                        and _black_border_frac(crop) <= self.max_black_frac):
                    self.embed_worker.submit(tid, box, crop, sharpness)
        return boxes, track_ids, self.embed_worker.drain_new()

    def stop(self) -> None:
        self.det_worker.stop()
        self.embed_worker.stop()


def build_arcface_core(detector_path: str | Path = DEFAULT_DETECTOR,
                       threshold: float = 0.35, *, conf: float = 0.5,
                       imgsz: int = 320, refresh_secs: float = 4.0,
                       min_face: int = 90, min_sharpness: float = 60.0,
                       align: bool = True, require_landmarks: bool = True
                       ) -> Tuple[PipelineCore, ArcFaceONNXRecognizer]:
    """Build a ready-to-run ArcFace pipeline core (used by the headless worker)."""
    from ultralytics import YOLO

    detector_path = Path(detector_path)
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector not found: {detector_path}")
    detector = YOLO(str(detector_path))
    recognizer = ArcFaceONNXRecognizer(threshold=threshold)
    processor = FaceProcessor(template="arcface", min_sharpness=min_sharpness,
                              yunet_path=None if align else "")
    if not align:
        processor.detector = None
    core = PipelineCore(detector, recognizer, processor, conf=conf, imgsz=imgsz,
                        refresh_secs=refresh_secs, min_face=min_face,
                        require_landmarks=require_landmarks and align)
    return core, recognizer
