"""Per-visit clip recorder for the headless worker.

Buffers downscaled frames at a capped cadence/length during a visit, then
encodes them to an MP4 the worker uploads to the API on visit end.
"""

from typing import List

import cv2
import numpy as np


class ClipRecorder:
    def __init__(self, max_frames: int = 50, fps: int = 10, width: int = 640):
        self.max_frames = max_frames
        self.fps = fps
        self.width = width
        self._frames: List[np.ndarray] = []
        self._last_ts: float = -1e9

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.width:
            return frame
        new_h = int(round(h * self.width / w))
        return cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_AREA)

    def maybe_add(self, frame: np.ndarray, now: float) -> None:
        if len(self._frames) >= self.max_frames:
            return
        if (now - self._last_ts) < (1.0 / self.fps):
            return
        self._last_ts = now
        self._frames.append(self._downscale(frame))

    def encode(self, path: str) -> bool:
        if not self._frames:
            return False
        h, w = self._frames[0].shape[:2]
        for fourcc_name in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(path, fourcc, float(self.fps), (w, h))
            if not writer.isOpened():
                writer.release()
                continue
            for f in self._frames:
                writer.write(f)
            writer.release()
            print(f"[clip] encoded {len(self._frames)} frames with {fourcc_name} -> {path}", flush=True)
            return True
        return False
