from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Person(SQLModel, table=True):
    """An anonymous, auto-discovered identity.

    Persons are created by the recognition pipeline when an unrecognized face
    appears — never by a human. ``label`` is an optional annotation an operator
    may add later (e.g. "janitor"); it is never required.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    label: Optional[str] = Field(default=None)
    best_sharpness: float = -1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Camera(SQLModel, table=True):
    """A video source the server pulls and runs a recognition worker against.

    ``source`` is any OpenCV-openable string: a device index ("0"), an MJPEG
    URL ("http://laptop:8090/stream"), an RTSP URL, or a video file path.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    location: str = ""
    source: str
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Sighting(SQLModel, table=True):
    """One continuous visit of a person to a camera.

    Opened (``ended_at`` null = currently visible) when a tracked face is first
    recognized, kept fresh by heartbeats (``last_seen``), and closed when the
    track drops or the reaper finds the heartbeat stale.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="person.id", index=True)
    camera_id: int = Field(foreign_key="camera.id", index=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = Field(default=None)
    best_sharpness: float = -1.0
    has_clip: bool = False
