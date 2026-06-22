#!/usr/bin/env python3
"""Make a short demo video from a still face image.

Lets the whole system run with no camera hardware: point a camera's ``source``
at the generated clip (workers loop file sources on EOF). Faces drift slightly
frame to frame so the detector/tracker behave like a real stream.

    python scripts/make_demo_clip.py tests/api/fixtures/face_known_1.jpg demo/lobby.mp4
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def make_clip(image_path: str, out_path: str, *, seconds: int = 12,
              fps: int = 15, size=(640, 480)) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read image: {image_path}")
    w, h = size
    face = cv2.resize(img, (int(w * 0.5), int(h * 0.5)))
    fh, fw = face.shape[:2]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = seconds * fps
    for i in range(n):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        # Gentle drift so boxes move a little, like a real scene.
        dx = int(20 * np.sin(i / 12))
        dy = int(12 * np.cos(i / 15))
        x = (w - fw) // 2 + dx
        y = (h - fh) // 2 + dy
        frame[y:y + fh, x:x + fw] = face
        writer.write(frame)
    writer.release()
    print(f"wrote {out_path} ({n} frames, {seconds}s @ {fps}fps)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tests/api/fixtures/face_known_1.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "demo/lobby.mp4"
    make_clip(src, dst)
