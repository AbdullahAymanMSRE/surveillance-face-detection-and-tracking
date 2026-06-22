#!/usr/bin/env python3
"""Laptop webcam publisher — exposes a local camera as an MJPEG stream.

Run this on each laptop that should act as a camera. It serves the webcam over
HTTP; on the server, add a camera whose ``source`` is this stream URL and the
recognition worker will pull it like any other source. Real IP cameras skip this
entirely (use their ``rtsp://`` URL directly).

Only needs OpenCV (`pip install opencv-python`) — not the full recognition stack.

    python publisher.py                 # publish webcam 0 on :8090
    python publisher.py --source 1 --port 8091
    python publisher.py --source clip.mp4   # publish a video file (loops)

Then on the server dashboard add a camera with source:
    http://<this-laptop-ip>:8090/stream
"""

import argparse
import http.server
import threading
import time

import cv2

BOUNDARY = "frame"


class Camera:
    """Reads frames in the background and holds the latest one."""

    def __init__(self, source: str, width: int, height: int):
        self.source = source
        self.width = width
        self.height = height
        self.is_file = not str(source).isdigit()
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _open(self):
        cap = cv2.VideoCapture(int(self.source) if not self.is_file else self.source)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cap = self._open()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file:           # loop the clip
                    cap.release()
                    cap = self._open()
                    continue
                time.sleep(0.2)            # webcam hiccup
                continue
            with self._lock:
                self._frame = frame
        cap.release()

    def latest(self):
        with self._lock:
            return self._frame


def main() -> int:
    p = argparse.ArgumentParser(description="Publish a local camera as MJPEG over HTTP")
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=15.0, help="max stream frame rate")
    args = p.parse_args()

    camera = Camera(args.source, args.width, args.height)
    camera.start()
    delay = 1.0 / max(args.fps, 1.0)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path not in ("/stream", "/"):
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    frame = camera.latest()
                    if frame is not None:
                        ok, buf = cv2.imencode(".jpg", frame)
                        if ok:
                            data = buf.tobytes()
                            self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                    time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Publishing source={args.source} at http://0.0.0.0:{args.port}/stream "
          f"(Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
