#!/usr/bin/env python3
"""Build the recognition gallery of known people.

Two modes:

1. From a folder of images already organized by person:
       gallery/
           alice/  img1.jpg img2.jpg ...
           bob/    img1.jpg ...
   Run:
       python enroll.py --gallery-dir gallery --out gallery/gallery.pt

2. Capture a new person live from the webcam (uses the YOLO detector to crop
   faces automatically):
       python enroll.py --webcam --name alice --gallery-dir gallery --out gallery/gallery.pt

Either way, the result is a gallery cache (.pt) that live_pipeline.py loads.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from recognizer import FaceRecognizer

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent  # repo root (dino_vit/ is two levels down)
DEFAULT_DETECTOR = _REPO_ROOT / "face_extraction" / "last.pt"
DEFAULT_RUN_DIR = _THIS_DIR / "best_model"


def capture_from_webcam(name: str, gallery_dir: Path, detector_path: Path,
                        num_shots: int, conf: float, camera_index: int) -> None:
    from ultralytics import YOLO

    person_dir = gallery_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector from {detector_path} ...")
    detector = YOLO(str(detector_path))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {camera_index}.")

    print(f"Capturing {num_shots} face crops for '{name}'.")
    print("Press SPACE to grab the largest detected face, 'q' to stop early.")

    saved = 0
    while saved < num_shots:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame grab failed; stopping.", file=sys.stderr)
            break

        results = detector.predict(frame, conf=conf, verbose=False)
        best_box, best_area = None, 0
        display = frame.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area, best_box = area, (x1, y1, x2, y2)

        cv2.putText(display, f"{name}: {saved}/{num_shots}  [SPACE]=save [q]=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Enroll", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and best_box is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = best_box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_path = person_dir / f"{name}_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved += 1
            print(f"  saved {out_path.name}  ({saved}/{num_shots})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {saved} crops into {person_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Enroll people into the recognition gallery")
    p.add_argument("--gallery-dir", default=str(_THIS_DIR / "gallery"),
                   help="Folder with one subfolder of images per person")
    p.add_argument("--out", default=None,
                   help="Output cache path (default: <gallery-dir>/gallery.pt)")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                   help="Recognition checkpoint dir (run_summary.json + .pt)")
    p.add_argument("--threshold", type=float, default=0.155)
    # webcam capture mode
    p.add_argument("--webcam", action="store_true",
                   help="Capture a new person from the webcam first")
    p.add_argument("--name", default=None, help="Person name (required with --webcam)")
    p.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    p.add_argument("--num-shots", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--camera-index", type=int, default=0)
    args = p.parse_args()

    gallery_dir = Path(args.gallery_dir)
    out_path = Path(args.out) if args.out else gallery_dir / "gallery.pt"

    if args.webcam:
        if not args.name:
            sys.exit("--webcam requires --name")
        capture_from_webcam(args.name, gallery_dir, Path(args.detector),
                            args.num_shots, args.conf, args.camera_index)

    print(f"Loading recognizer from {args.run_dir} ...")
    recognizer = FaceRecognizer(args.run_dir, threshold=args.threshold)

    print(f"Building gallery from {gallery_dir} ...")
    count = recognizer.load_gallery(gallery_dir)
    if count == 0:
        sys.exit(f"No identities found under {gallery_dir}. Add <name>/*.jpg folders.")

    recognizer.save_gallery(out_path)
    print(f"Enrolled {count} identities: {', '.join(recognizer.gallery_names)}")
    print(f"Saved gallery cache to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
