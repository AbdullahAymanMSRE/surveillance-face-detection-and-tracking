import numpy as np
from sqlmodel import Session, select

from api import ml
from api.db import get_engine
from api.gallery import get_gallery
from api.models import FaceEmbedding, Person


def _seed_person_from_image(image_bytes: bytes) -> int:
    """Enroll a real face into DB + gallery the way the pipeline would."""
    frame = ml.decode_image(image_bytes)
    vector_bytes, _crop, sharpness = ml.detect_and_embed(frame)
    with Session(get_engine()) as s:
        person = Person(best_sharpness=sharpness)
        s.add(person)
        s.commit()
        s.refresh(person)
        s.add(FaceEmbedding(person_id=person.id, vector=vector_bytes))
        s.commit()
        person_id = person.id
    get_gallery().add(person_id, np.frombuffer(vector_bytes, dtype=np.float32))
    return person_id


def test_search_no_face_returns_422(client, no_face_bytes):
    resp = client.post("/search", files={"image": ("x.jpg", no_face_bytes, "image/jpeg")})
    assert resp.status_code == 422


def test_search_no_match_on_empty_gallery(client, face_known_1_bytes):
    resp = client.post(
        "/search", files={"image": ("x.jpg", face_known_1_bytes, "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["match"] is None


def test_search_finds_enrolled_person(client, face_known_1_bytes, face_known_2_bytes):
    pid = _seed_person_from_image(face_known_1_bytes)
    # query with a *different* photo of the same person
    resp = client.post(
        "/search", files={"image": ("x.jpg", face_known_2_bytes, "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["match"] is not None
    assert body["match"]["id"] == pid
    assert body["score"] >= 0.28
    assert "timeline" in body["match"]


def test_search_does_not_write(client, face_known_1_bytes):
    _seed_person_from_image(face_known_1_bytes)
    with Session(get_engine()) as s:
        persons_before = len(s.exec(select(Person)).all())
        embeds_before = len(s.exec(select(FaceEmbedding)).all())

    client.post("/search", files={"image": ("x.jpg", face_known_1_bytes, "image/jpeg")})

    with Session(get_engine()) as s:
        assert len(s.exec(select(Person)).all()) == persons_before
        assert len(s.exec(select(FaceEmbedding)).all()) == embeds_before
