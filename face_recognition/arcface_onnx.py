"""ArcFace face recognizer running on ONNX Runtime (CPU).

A drop-in alternative to the DINO/ViT `FaceRecognizer` with the same interface
(`embed_dim`, `embed`, `embed_batch`). It runs a pretrained InsightFace ArcFace
model (w600k_mbf, MobileFaceNet, 512-d) through ONNX Runtime — fast on CPU
(~5-15 ms/face) and trained for pose/lighting invariance, unlike the ViT.

Expects an already-aligned 112x112 BGR face crop (produced by
FaceProcessor(template="arcface")). Preprocessing matches InsightFace's
recognition model: BGR->RGB, normalized to [-1, 1], NCHW float32.

Needs only onnxruntime (no insightface package).
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import torch

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = _THIS_DIR / "models" / "w600k_mbf.onnx"

INPUT_SIZE = 112


class ArcFaceONNXRecognizer:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 threshold: float = 0.28, num_threads: int = 2):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace ONNX model not found: {model_path}\n"
                f"Download buffalo_s and extract w600k_mbf.onnx into models/.")

        self.threshold = threshold
        self.device = "cpu"
        # Cap threads so recognition doesn't starve the camera/display loop.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.embed_dim = self.session.get_outputs()[0].shape[1]  # 512

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            crop_bgr = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        return rgb.transpose(2, 0, 1)  # HWC -> CHW

    def embed_batch(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        """Return L2-normalized embeddings [B, 512] as a CPU torch tensor."""
        if not crops_bgr:
            return torch.empty(0, self.embed_dim)
        blob = np.stack([self._preprocess(c) for c in crops_bgr]).astype(np.float32)
        out = self.session.run(None, {self.input_name: blob})[0]  # [B, 512]
        emb = torch.from_numpy(np.ascontiguousarray(out)).float()
        return torch.nn.functional.normalize(emb, dim=1)

    def embed(self, crop_bgr: np.ndarray) -> torch.Tensor:
        return self.embed_batch([crop_bgr])[0]
