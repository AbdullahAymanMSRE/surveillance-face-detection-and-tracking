"""Worker supervisor: one recognition subprocess per enabled camera.

The supervisor runs inside the API process. It spawns ``pipeline_node.py`` per
camera, monitors the children, and restarts crashed workers. The dashboard's
start/stop controls flip ``Camera.enabled`` (the desired run state) and the
monitor loop reconciles reality to it, so a manual stop stays stopped while a
crash gets restarted.

Disabled under tests via ``FACE_API_ENABLE_SUPERVISOR=0`` so the test suite never
spawns real subprocesses.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

from sqlmodel import Session, select

from .db import get_engine
from .models import Camera

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODE = _REPO_ROOT / "pipeline_node.py"


class Supervisor:
    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or os.environ.get(
            "FACE_API_SELF_URL", "http://127.0.0.1:8000")
        self.active = False           # only spawns when active (off under tests)
        self.interval = 5.0
        self._procs: Dict[int, subprocess.Popen] = {}
        self._ports: Dict[int, int] = {}     # camera_id -> preview port
        self._next_port = 8100
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None

    # ── lifecycle control (used by the cameras router) ───────────────────────

    def start_camera(self, camera_id: int) -> None:
        if not self.active:
            return
        with self._lock:
            proc = self._procs.get(camera_id)
            if proc is not None and proc.poll() is None:
                return  # already running
            port = self._ports.get(camera_id)
            if port is None:
                port = self._next_port
                self._next_port += 1
                self._ports[camera_id] = port
            args = [sys.executable, str(_NODE),
                    "--camera-id", str(camera_id),
                    "--api-url", self.api_url,
                    "--preview-port", str(port)]
            # Optional detector overrides (e.g. larger imgsz / lower conf for
            # small or high-angle faces). Unset -> pipeline_node.py defaults.
            det_imgsz = os.environ.get("FACE_DET_IMGSZ")
            det_conf = os.environ.get("FACE_DET_CONF")
            det_min_face = os.environ.get("FACE_DET_MIN_FACE")
            det_min_sharp = os.environ.get("FACE_DET_MIN_SHARPNESS")
            if det_imgsz:
                args += ["--imgsz", det_imgsz]
            if det_conf:
                args += ["--conf", det_conf]
            if det_min_face:
                args += ["--min-face", det_min_face]
            if det_min_sharp:
                args += ["--min-sharpness", det_min_sharp]
            self._procs[camera_id] = subprocess.Popen(args)

    def stop_camera(self, camera_id: int) -> None:
        with self._lock:
            proc = self._procs.pop(camera_id, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def is_running(self, camera_id: int) -> bool:
        with self._lock:
            proc = self._procs.get(camera_id)
            return proc is not None and proc.poll() is None

    def preview_port(self, camera_id: int) -> Optional[int]:
        with self._lock:
            if self.is_running(camera_id):
                return self._ports.get(camera_id)
            return None

    # ── monitor: reconcile running workers to enabled cameras ────────────────

    def reconcile(self) -> None:
        if not self.active:
            return
        with Session(get_engine()) as session:
            cameras = session.exec(select(Camera)).all()
        enabled_ids = {c.id for c in cameras if c.enabled}
        for camera_id in enabled_ids:
            self.start_camera(camera_id)          # (re)start enabled, incl. crashes
        with self._lock:
            running = list(self._procs.keys())
        for camera_id in running:
            if camera_id not in enabled_ids:        # disabled or deleted -> stop
                self.stop_camera(camera_id)

    def start_monitor(self, interval: float = 5.0) -> None:
        self.active = True
        self.interval = interval
        self.reconcile()  # bring enabled cameras up immediately at startup
        self._monitor = threading.Thread(target=self._loop, daemon=True)
        self._monitor.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.reconcile()
            except Exception:
                pass  # transient DB/engine churn; next tick recovers

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            ids = list(self._procs.keys())
        for camera_id in ids:
            self.stop_camera(camera_id)


_supervisor: Optional[Supervisor] = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor


def reset_supervisor() -> None:
    global _supervisor
    _supervisor = None
