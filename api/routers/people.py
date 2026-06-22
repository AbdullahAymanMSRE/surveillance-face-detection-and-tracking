"""People read endpoints + read-only image search.

The dashboard reads anonymous identities and their appearance timelines here.
``POST /search`` looks a face up against the gallery without ever writing — no
new person, embedding, or sighting is created.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from .. import ml
from ..db import get_session, get_thumbnails_dir
from ..gallery import get_gallery
from ..models import Camera, Person, Sighting
from ..serializers import _iso, person_response, sighting_response

router = APIRouter()


def _camera_map(session: Session) -> dict:
    return {c.id: c for c in session.exec(select(Camera)).all()}


def _person_summary(person: Person, sightings: list) -> dict:
    last_seen = max((s.last_seen for s in sightings), default=None)
    data = person_response(person)
    data.update({
        "visits": len(sightings),
        "lastSeen": _iso(last_seen),
        "active": any(s.ended_at is None for s in sightings),
    })
    return data


@router.get("/people")
def list_people(session: Session = Depends(get_session)):
    people = session.exec(select(Person)).all()
    by_person: dict = {}
    for s in session.exec(select(Sighting)).all():
        by_person.setdefault(s.person_id, []).append(s)
    out = [_person_summary(p, by_person.get(p.id, [])) for p in people]
    out.sort(key=lambda d: (d["lastSeen"] or ""), reverse=True)
    return out


@router.get("/people/{person_id}")
def get_person(person_id: int, session: Session = Depends(get_session)):
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(404, detail="Person not found")
    sightings = session.exec(
        select(Sighting).where(Sighting.person_id == person_id)
        .order_by(Sighting.started_at.desc())
    ).all()
    cameras = _camera_map(session)
    data = _person_summary(person, sightings)
    data["timeline"] = [sighting_response(s, cameras.get(s.camera_id)) for s in sightings]
    return data


class PersonUpdate(BaseModel):
    label: Optional[str] = None


@router.patch("/people/{person_id}")
def update_person(person_id: int, body: PersonUpdate,
                  session: Session = Depends(get_session)):
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(404, detail="Person not found")
    person.label = body.label
    session.add(person)
    session.commit()
    session.refresh(person)
    return person_response(person)


@router.get("/people/{person_id}/thumbnail")
def get_thumbnail(person_id: int, session: Session = Depends(get_session)):
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(404, detail="Person not found")
    path = get_thumbnails_dir() / f"{person_id}.jpg"
    if not path.exists():
        raise HTTPException(404, detail="No thumbnail for this person")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/search")
def search(image: UploadFile = File(...), session: Session = Depends(get_session)):
    """Look an uploaded face up against the gallery. Never writes."""
    frame = ml.decode_image(image.file.read())
    try:
        vector_bytes, _crop, _sharp = ml.detect_and_embed(frame)
    except ml.NoFaceDetected:
        raise HTTPException(422, detail="No face detected")

    vec = np.frombuffer(vector_bytes, dtype=np.float32)
    person_id, score = get_gallery().match(vec)
    if person_id is None:
        return {"match": None, "score": score}

    person = session.get(Person, person_id)
    if person is None:  # gallery/DB drift; treat as no match
        return {"match": None, "score": score}
    sightings = session.exec(
        select(Sighting).where(Sighting.person_id == person_id)
        .order_by(Sighting.started_at.desc())
    ).all()
    cameras = _camera_map(session)
    match = _person_summary(person, sightings)
    match["timeline"] = [sighting_response(s, cameras.get(s.camera_id)) for s in sightings]
    return {"match": match, "score": score}
