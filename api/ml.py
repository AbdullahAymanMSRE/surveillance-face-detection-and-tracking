import sys
from pathlib import Path

import cv2
import numpy as np

_API_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))

from arcface_onnx import ArcFaceONNXRecognizer  # noqa: E402
from face_align import FaceProcessor  # noqa: E402

DETECTOR_PATH = _REPO_ROOT / "face_extraction" / "last.pt"

_detector = None
_processor = None
_recognizer = None


class NoFaceDetected(Exception):
    pass


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(str(DETECTOR_PATH))
    return _detector


def get_processor():
    global _processor
    if _processor is None:
        _processor = FaceProcessor(template="arcface")
    return _processor


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = ArcFaceONNXRecognizer()
    return _recognizer


def detect_and_embed(image_bgr: np.ndarray):
    """Detect the largest face, align it, and embed it.

    Returns (vector_bytes, crop_bgr, sharpness). Raises NoFaceDetected if no
    face is found.
    """
    detector = get_detector()
    results = detector.predict(image_bgr, conf=0.3, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    if not boxes:
        raise NoFaceDetected()

    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    processor = get_processor()
    crop, sharpness, _aligned = processor.process(image_bgr, box)

    recognizer = get_recognizer()
    embedding = recognizer.embed(crop)
    vector_bytes = embedding.numpy().astype(np.float32).tobytes()
    return vector_bytes, crop, sharpness
