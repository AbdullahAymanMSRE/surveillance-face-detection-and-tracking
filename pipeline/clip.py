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

    def maybe_add(self, frame: np.ndarray, now: float,
                  box=None, label=None) -> None:
        if len(self._frames) >= self.max_frames:
            return
        if (now - self._last_ts) < (1.0 / self.fps):
            return
        self._last_ts = now
        src_w = frame.shape[1]
        out = self._downscale(frame)
        if box is not None:
            if out is frame:          # no resize happened -> don't mutate caller's frame
                out = out.copy()
            self._draw(out, src_w, box, label)
        self._frames.append(out)

    def _draw(self, img: np.ndarray, src_width: int, box, label) -> None:
        scale = img.shape[1] / src_width
        x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not label:
            return
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bar_h = th + base + 6
        by1, by2 = y1 - bar_h, y1        # bar sits just above the box...
        if by1 < 0:                      # ...unless the box hugs the top edge
            by1, by2 = y1, y1 + bar_h
        cv2.rectangle(img, (x1, by1), (x1 + tw + 8, by2), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 4, by2 - base - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
