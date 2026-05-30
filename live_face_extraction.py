"""Live face extraction from the MacBook webcam using a trained YOLO model."""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "last.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.3
SAVE_EVERY_N = 5
OUTPUT_DIR = "output/cropped_faces"
LATEST_FACE_DISPLAY_SIZE = 256

LIVE_WINDOW = "Live"
FACE_WINDOW = "Latest Face"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_PATH
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading model from {model_path} ...")
    model = YOLO(str(model_path))
    print("Model loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(
            f"Could not open camera index {CAMERA_INDEX}.\n"
            "On macOS, grant camera access to your terminal/IDE in\n"
            "System Settings -> Privacy & Security -> Camera, then retry.",
            file=sys.stderr,
        )
        return 2

    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        print(
            "Camera opened but returned no frames. This usually means macOS\n"
            "denied camera access to the host process. Grant access in\n"
            "System Settings -> Privacy & Security -> Camera, then retry.",
            file=sys.stderr,
        )
        return 3

    print("Camera ready. Press 'q' in either window to quit.")

    latest_face = np.zeros(
        (LATEST_FACE_DISPLAY_SIZE, LATEST_FACE_DISPLAY_SIZE, 3), dtype=np.uint8
    )
    frame_idx = 0
    saved_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed; stopping.", file=sys.stderr)
                break

            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            display = frame.copy()
            should_save = frame_idx % SAVE_EVERY_N == 0

            face_idx = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        f"Face {conf:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    h, w = frame.shape[:2]
                    cx1, cy1 = max(x1, 0), max(y1, 0)
                    cx2, cy2 = min(x2, w), min(y2, h)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue

                    latest_face = cv2.resize(
                        crop,
                        (LATEST_FACE_DISPLAY_SIZE, LATEST_FACE_DISPLAY_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )

                    if should_save:
                        out_path = (
                            output_dir
                            / f"face_{frame_idx:06d}_{face_idx:02d}.jpg"
                        )
                        cv2.imwrite(str(out_path), crop)
                        saved_count += 1

                    face_idx += 1

            cv2.imshow(LIVE_WINDOW, display)
            cv2.imshow(FACE_WINDOW, latest_face)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved {saved_count} face crops to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
