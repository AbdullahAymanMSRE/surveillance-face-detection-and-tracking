"""Camera admin + worker control.

Cameras live in the DB (decision B). Creating one and flipping its ``enabled``
flag via start/stop is how the dashboard brings recognition workers up and down;
the supervisor reconciles running workers to the enabled set.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import Camera
from ..supervisor import get_supervisor

router = APIRouter()


class CameraCreate(BaseModel):
    name: str
    location: str = ""
    source: str
    enabled: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    enabled: Optional[bool] = None


def camera_response(camera: Camera) -> dict:
    return {
        "id": camera.id,
        "name": camera.name,
        "location": camera.location,
        "source": camera.source,
        "enabled": camera.enabled,
        "status": "running" if get_supervisor().is_running(camera.id) else "stopped",
    }


@router.post("/cameras", status_code=201)
def create_camera(body: CameraCreate, session: Session = Depends(get_session)):
    camera = Camera(name=body.name, location=body.location,
                    source=body.source, enabled=body.enabled)
    session.add(camera)
    session.commit()
    session.refresh(camera)
    if camera.enabled:
        get_supervisor().start_camera(camera.id)
    return camera_response(camera)


@router.get("/cameras")
def list_cameras(session: Session = Depends(get_session)):
    cameras = session.exec(select(Camera)).all()
    return [camera_response(c) for c in cameras]


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: int, session: Session = Depends(get_session)):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, detail="Camera not found")
    return camera_response(camera)


@router.patch("/cameras/{camera_id}")
def update_camera(camera_id: int, body: CameraUpdate,
                  session: Session = Depends(get_session)):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, detail="Camera not found")
    data = body.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(camera, field, value)
    session.add(camera)
    session.commit()
    session.refresh(camera)
    # A source/enabled change needs the worker restarted to take effect.
    sup = get_supervisor()
    sup.stop_camera(camera.id)
    if camera.enabled:
        sup.start_camera(camera.id)
    return camera_response(camera)


@router.delete("/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: int, session: Session = Depends(get_session)):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, detail="Camera not found")
    get_supervisor().stop_camera(camera_id)
    session.delete(camera)
    session.commit()


@router.post("/cameras/{camera_id}/start")
def start_camera(camera_id: int, session: Session = Depends(get_session)):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, detail="Camera not found")
    camera.enabled = True
    session.add(camera)
    session.commit()
    session.refresh(camera)
    get_supervisor().start_camera(camera_id)
    return camera_response(camera)


@router.get("/cameras/{camera_id}/preview")
def preview(camera_id: int, session: Session = Depends(get_session)):
    """Proxy the worker's on-demand raw MJPEG stream to the browser."""
    if session.get(Camera, camera_id) is None:
        raise HTTPException(404, detail="Camera not found")
    port = get_supervisor().preview_port(camera_id)
    if port is None:
        raise HTTPException(503, detail="Camera worker not running")
    upstream = f"http://127.0.0.1:{port}/stream"

    def relay():
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", upstream) as resp:
                    for chunk in resp.iter_raw():
                        yield chunk
        except httpx.HTTPError:
            return  # worker stopped or unreachable

    return StreamingResponse(
        relay(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: int, session: Session = Depends(get_session)):
    camera = session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, detail="Camera not found")
    camera.enabled = False
    session.add(camera)
    session.commit()
    session.refresh(camera)
    get_supervisor().stop_camera(camera_id)
    return camera_response(camera)
