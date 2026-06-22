#!/usr/bin/env python3
"""Headless camera worker — the production recognition node.

Spawned by the API supervisor, one per camera. It fetches its source from the
central API, runs the shared ``pipeline.core`` (detect -> track -> align ->
embed), and reports the sighting lifecycle to the API, which performs central
identity matching. No GUI and no local database.

    python pipeline_node.py --camera-id 3 --api-url http://127.0.0.1:8000

A video-file source loops on EOF so a clip can stand in for a live camera. The
preview port is reserved for the on-demand MJPEG preview added in a later phase.
"""

import argparse
import http.server
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import httpx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))

from pipeline.core import build_arcface_core  # noqa: E402

# How long a track may be absent before we close its visit. Larger than the
# tracker's max_age debounce so a brief detection dropout doesn't end a visit.
END_GRACE_SECS = 2.0

_PREVIEW_BOUNDARY = "frame"


class PreviewState:
    """Holds the latest raw frame so the preview server can serve it."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def snapshot(self):
        with self._lock:
            return self._frame


def start_preview_server(state: PreviewState, port: int):
    """Serve the latest raw frame as MJPEG. Encoding happens only while a client
    is connected (on-demand), so an unwatched camera costs nothing extra."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence access logs
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_PREVIEW_BOUNDARY}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    frame = state.snapshot()
                    if frame is not None:
                        ok, buf = cv2.imencode(".jpg", frame)
                        if ok:
                            data = buf.tobytes()
                            self.wfile.write(f"--{_PREVIEW_BOUNDARY}\r\n".encode())
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                    time.sleep(0.1)  # ~10 fps preview
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (the API proxy) went away

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _open_capture(source: str) -> cv2.VideoCapture:
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def _is_file_source(source: str) -> bool:
    return not str(source).isdigit() and Path(source).exists()


def _fetch_camera(client: httpx.Client, api_url: str, camera_id: int) -> Optional[dict]:
    """Fetch camera config, retrying while the API/camera is not yet ready."""
    for _ in range(60):
        try:
            resp = client.get(f"{api_url}/cameras/{camera_id}", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                print(f"[node] camera {camera_id} not found; exiting", flush=True)
                return None
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    print(f"[node] gave up fetching camera {camera_id}", flush=True)
    return None


def _encode(ev) -> dict:
    emb_bytes = ev.embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    ok, buf = cv2.imencode(".jpg", ev.crop)
    crop_bytes = buf.tobytes() if ok else b""
    return {
        "data": {"sharpness": float(ev.sharpness)},
        "files": {
            "embedding": ("e.bin", emb_bytes, "application/octet-stream"),
            "crop": ("c.jpg", crop_bytes, "image/jpeg"),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Headless camera recognition worker")
    p.add_argument("--camera-id", type=int, required=True)
    p.add_argument("--api-url", default=os.environ.get("FACE_API_SELF_URL",
                                                       "http://127.0.0.1:8000"))
    p.add_argument("--preview-port", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.28)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--refresh-secs", type=float, default=4.0)
    p.add_argument("--min-face", type=int, default=90)
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Stop after N seconds (0 = run forever; for testing)")
    args = p.parse_args()

    client = httpx.Client()
    camera = _fetch_camera(client, args.api_url, args.camera_id)
    if camera is None:
        return 1
    source = camera["source"]
    is_file = _is_file_source(source)
    print(f"[node] camera {args.camera_id} '{camera['name']}' source={source} "
          f"(file={is_file})", flush=True)

    core, _recognizer = build_arcface_core(
        threshold=args.threshold, conf=args.conf, imgsz=args.imgsz,
        refresh_secs=args.refresh_secs, min_face=args.min_face)
    core.start()

    cap = _open_capture(source)
    if not cap.isOpened():
        print(f"[node] could not open source: {source}", flush=True)
        return 2

    preview = PreviewState()
    if args.preview_port:
        try:
            start_preview_server(preview, args.preview_port)
            print(f"[node] preview on :{args.preview_port}/stream", flush=True)
        except OSError as e:
            print(f"[node] preview server failed: {e}", flush=True)

    api = args.api_url
    camera_id = args.camera_id
    # track_id -> sighting_id (open visits); last time each track was on screen.
    open_sightings: Dict[int, int] = {}
    last_present: Dict[int, float] = {}
    started = time.time()

    def _post_open(ev) -> Optional[int]:
        try:
            payload = _encode(ev)
            payload["data"]["camera_id"] = camera_id
            r = client.post(f"{api}/sightings", timeout=10, **payload)
            if r.status_code == 201:
                return r.json()["sightingId"]
        except httpx.HTTPError as e:
            print(f"[node] open failed: {e}", flush=True)
        return None

    def _post_heartbeat(sighting_id: int, ev) -> None:
        try:
            payload = _encode(ev)
            client.post(f"{api}/sightings/{sighting_id}/heartbeat",
                        timeout=10, **payload)
        except httpx.HTTPError:
            pass

    def _post_end(sighting_id: int) -> None:
        try:
            client.post(f"{api}/sightings/{sighting_id}/end", timeout=10)
        except httpx.HTTPError:
            pass

    print("[node] running", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    cap.release()
                    cap = _open_capture(source)  # loop the clip
                    continue
                time.sleep(0.3)  # live stream hiccup; retry
                if not cap.isOpened():
                    cap = _open_capture(source)
                continue

            preview.set_frame(frame)  # raw frame for the on-demand preview
            boxes, track_ids, events = core.step(frame)
            now = time.time()
            for tid in track_ids:
                last_present[tid] = now

            for ev in events:
                if ev.track_id in open_sightings:
                    _post_heartbeat(open_sightings[ev.track_id], ev)
                else:
                    sid = _post_open(ev)
                    if sid is not None:
                        open_sightings[ev.track_id] = sid

            # Close visits whose track has been gone past the grace window.
            for tid in list(open_sightings):
                if now - last_present.get(tid, 0.0) > END_GRACE_SECS:
                    _post_end(open_sightings.pop(tid))
                    last_present.pop(tid, None)

            if args.max_seconds and (now - started) > args.max_seconds:
                break
    finally:
        for sid in open_sightings.values():
            _post_end(sid)
        core.stop()
        cap.release()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
