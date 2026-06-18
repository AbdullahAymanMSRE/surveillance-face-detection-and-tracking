"""Face alignment + quality gating to stabilize recognition embeddings.

Raw YOLO bounding-box crops vary wildly in framing, scale and in-plane rotation,
which makes the recognition model produce very different embeddings for the same
person. This module normalizes each face before it reaches the recognizer:

  - Detects 5 facial landmarks (eyes, nose, mouth corners) with OpenCV's YuNet.
  - Warps the face to a canonical pose via a similarity transform, so eyes are
    level and the face sits at a consistent scale/position (with margin, to match
    how the recognition model's training images were framed).
  - Falls back to a square, margin-padded crop when no landmarks are found.
  - Reports a quality score (sharpness) so callers can skip blurry frames.

Needs only opencv (YuNet model file in models/), no extra pip dependency.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_YUNET = _THIS_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# Canonical ArcFace 5-point template defined on a 112x112 face.
# Order matches YuNet's landmarks: right eye, left eye, nose, right mouth, left mouth.
_REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _build_template(out_size: int, face_frac: float) -> np.ndarray:
    """Scale/center the 112-template into an out_size canvas.

    face_frac controls how much of the output the face fills (smaller -> more
    margin around the face, to mimic loosely-cropped training images).
    """
    t = _REFERENCE_5PTS * (out_size * face_frac / 112.0)
    t[:, 0] += out_size / 2 - t[:, 0].mean()
    t[:, 1] += out_size / 2 - t[:, 1].mean()
    return t


def laplacian_sharpness(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — higher is sharper, low means blurry."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def square_pad_crop(frame: np.ndarray, box: Tuple[int, int, int, int],
                    out_size: int, margin: float) -> np.ndarray:
    """Expand a box to a square with margin, clamp to frame, resize to out_size."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    h, w = frame.shape[:2]
    sx1, sy1 = max(int(cx - half), 0), max(int(cy - half), 0)
    sx2, sy2 = min(int(cx + half), w), min(int(cy + half), h)
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


class FaceProcessor:
    """Aligns + quality-checks faces. Degrades gracefully if YuNet is missing."""

    def __init__(self, out_size: int = 224, margin: float = 0.4,
                 face_frac: float = 0.5, min_sharpness: float = 40.0,
                 yunet_path: Optional[str] = None, det_score: float = 0.6,
                 template: str = "dino"):
        self.margin = margin
        self.min_sharpness = min_sharpness
        self.template_kind = template
        if template == "arcface":
            # Standard ArcFace 112 template (equivalent to insightface norm_crop).
            self.out_size = 112
            self.template = _REFERENCE_5PTS.copy()
            self._border = cv2.BORDER_CONSTANT
        else:
            self.out_size = out_size
            self.template = _build_template(out_size, face_frac)
            self._border = cv2.BORDER_REPLICATE

        path = Path(yunet_path) if yunet_path else DEFAULT_YUNET
        self.detector = None
        if path.exists():
            self.detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold=det_score)

    def _landmarks(self, region: np.ndarray) -> Optional[np.ndarray]:
        """Return the largest face's 5 landmarks (5x2) in region coords, or None."""
        if self.detector is None:
            return None
        h, w = region.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(region)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])  # largest by w*h
        return face[4:14].reshape(5, 2).astype(np.float32)

    def process(self, frame: np.ndarray, box: Tuple[int, int, int, int]
                ) -> Tuple[np.ndarray, float, bool]:
        """Return (aligned_crop, sharpness, aligned?) for one detection box.

        Aligns via landmarks when possible; otherwise returns a square padded
        crop. `aligned` is True only when landmark-based warping was applied.
        """
        # Work on a margin-expanded region so the warp has real context.
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * (1 + self.margin) / 2
        h, w = frame.shape[:2]
        rx1, ry1 = max(int(cx - half), 0), max(int(cy - half), 0)
        rx2, ry2 = min(int(cx + half), w), min(int(cy + half), h)
        region = frame[ry1:ry2, rx1:rx2]
        if region.size == 0:
            blank = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)
            return blank, 0.0, False

        pts = self._landmarks(region)
        if pts is None:
            crop = square_pad_crop(frame, box, self.out_size, self.margin)
            return crop, laplacian_sharpness(crop), False

        M, _ = cv2.estimateAffinePartial2D(pts, self.template, method=cv2.LMEDS)
        if M is None:
            crop = square_pad_crop(frame, box, self.out_size, self.margin)
            return crop, laplacian_sharpness(crop), False

        aligned = cv2.warpAffine(region, M, (self.out_size, self.out_size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=self._border)
        return aligned, laplacian_sharpness(aligned), True

    def is_sharp(self, sharpness: float) -> bool:
        return sharpness >= self.min_sharpness
