from datetime import datetime, timedelta

import numpy as np
from sqlmodel import Session, select

from api.db import get_engine
from api.models import Camera, Person, Sighting
from api.reaper import close_stale_sightings


def _emb(idx: int) -> bytes:
    """A 512-d one-hot float32 embedding; different idx => orthogonal => no match."""
    v = np.zeros(512, dtype=np.float32)
    v[idx] = 1.0
    return v.tobytes()


def _make_camera(name="A", location="Gate", source="0") -> int:
    with Session(get_engine()) as s:
        cam = Camera(name=name, location=location, source=source)
        s.add(cam)
        s.commit()
        s.refresh(cam)
        return cam.id


def _open(client, camera_id: int, emb: bytes, sharpness: float = 50.0):
    return client.post(
        "/sightings",
        data={"camera_id": camera_id, "sharpness": sharpness},
        files={
            "embedding": ("e.bin", emb, "application/octet-stream"),
            "crop": ("c.jpg", b"fake-jpeg-bytes", "image/jpeg"),
        },
    )


def test_open_sighting_creates_new_person(client):
    cam = _make_camera()
    resp = _open(client, cam, _emb(0))
    assert resp.status_code == 201
    body = resp.json()
    assert body["isNew"] is True
    assert body["personId"] is not None
    assert body["sightingId"] is not None


def test_same_face_matches_across_cameras(client):
    cam_a = _make_camera(name="A")
    cam_b = _make_camera(name="B")
    first = _open(client, cam_a, _emb(0)).json()
    second = _open(client, cam_b, _emb(0)).json()
    assert second["isNew"] is False
    assert second["personId"] == first["personId"]
    # Two distinct sightings (one per camera) for the same identity.
    assert second["sightingId"] != first["sightingId"]


def test_different_face_creates_distinct_person(client):
    cam = _make_camera()
    p1 = _open(client, cam, _emb(0)).json()
    p2 = _open(client, cam, _emb(1)).json()
    assert p2["isNew"] is True
    assert p2["personId"] != p1["personId"]


def test_unknown_camera_rejected(client):
    resp = _open(client, 999, _emb(0))
    assert resp.status_code == 404


def test_end_sets_ended_at(client):
    cam = _make_camera()
    sighting_id = _open(client, cam, _emb(0)).json()["sightingId"]
    resp = client.post(f"/sightings/{sighting_id}/end")
    assert resp.status_code == 200
    with Session(get_engine()) as s:
        assert s.get(Sighting, sighting_id).ended_at is not None


def test_heartbeat_updates_last_seen(client):
    cam = _make_camera()
    sighting_id = _open(client, cam, _emb(0)).json()["sightingId"]
    with Session(get_engine()) as s:
        before = s.get(Sighting, sighting_id).last_seen
    resp = client.post(f"/sightings/{sighting_id}/heartbeat", data={"sharpness": 10.0})
    assert resp.status_code == 200
    with Session(get_engine()) as s:
        assert s.get(Sighting, sighting_id).last_seen >= before


def test_reaper_closes_stale_open_sighting(session):
    person = Person()
    camera = Camera(name="A", source="0")
    session.add(person)
    session.add(camera)
    session.commit()
    session.refresh(person)
    session.refresh(camera)

    stale = Sighting(
        person_id=person.id,
        camera_id=camera.id,
        last_seen=datetime.utcnow() - timedelta(seconds=100),
    )
    fresh = Sighting(person_id=person.id, camera_id=camera.id)
    session.add(stale)
    session.add(fresh)
    session.commit()

    closed = close_stale_sightings(session, timeout_secs=30)
    assert closed == 1

    open_rows = session.exec(
        select(Sighting).where(Sighting.ended_at == None)  # noqa: E711
    ).all()
    assert len(open_rows) == 1  # only the fresh one remains open


def test_sighting_response_includes_clip_fields():
    from api.models import Sighting
    from api.serializers import sighting_response
    s = Sighting(id=5, person_id=1, camera_id=1, has_clip=True)
    body = sighting_response(s, None)
    assert body["hasClip"] is True
    assert body["clipUrl"] == "/sightings/5/clip"
    s2 = Sighting(id=6, person_id=1, camera_id=1, has_clip=False)
    assert sighting_response(s2, None)["clipUrl"] is None
