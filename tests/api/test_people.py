import numpy as np
from sqlmodel import Session

from api.db import get_engine
from api.models import Camera


def _emb(idx: int) -> bytes:
    v = np.zeros(512, dtype=np.float32)
    v[idx] = 1.0
    return v.tobytes()


def _make_camera(name="Cam", location="Place", source="0") -> int:
    with Session(get_engine()) as s:
        cam = Camera(name=name, location=location, source=source)
        s.add(cam)
        s.commit()
        s.refresh(cam)
        return cam.id


def _open(client, camera_id, emb, sharpness=50.0):
    return client.post(
        "/sightings",
        data={"camera_id": camera_id, "sharpness": sharpness},
        files={
            "embedding": ("e.bin", emb, "application/octet-stream"),
            "crop": ("c.jpg", b"jpeg", "image/jpeg"),
        },
    ).json()


def test_list_people_includes_visit_summary(client):
    cam = _make_camera()
    _open(client, cam, _emb(0))
    people = client.get("/people").json()
    assert len(people) == 1
    p = people[0]
    assert p["displayName"] == "person_001"
    assert p["visits"] == 1
    assert p["active"] is True            # sighting still open
    assert p["lastSeen"] is not None


def test_person_detail_has_timeline_across_cameras(client):
    cam_a = _make_camera(name="A", location="Gate")
    cam_b = _make_camera(name="B", location="Lab")
    pid = _open(client, cam_a, _emb(0))["personId"]
    _open(client, cam_b, _emb(0))  # same face, second camera

    detail = client.get(f"/people/{pid}").json()
    assert detail["visits"] == 2
    locations = {entry["location"] for entry in detail["timeline"]}
    assert locations == {"Gate", "Lab"}


def test_patch_person_label(client):
    cam = _make_camera()
    pid = _open(client, cam, _emb(0))["personId"]
    resp = client.patch(f"/people/{pid}", json={"label": "Janitor"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "Janitor"
    assert client.get(f"/people/{pid}").json()["displayName"] == "Janitor"


def test_active_sightings_endpoint(client):
    cam = _make_camera(name="Lobby", location="Lobby")
    open_sid = _open(client, cam, _emb(0))["sightingId"]
    other = _open(client, cam, _emb(1))["sightingId"]
    client.post(f"/sightings/{other}/end")

    active = client.get("/sightings/active").json()
    ids = {a["id"] for a in active}
    assert open_sid in ids
    assert other not in ids
    assert active[0]["person"] is not None


def test_get_missing_person_404(client):
    assert client.get("/people/999").status_code == 404
