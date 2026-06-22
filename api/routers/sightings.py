"""Ingestion endpoints: the camera workers' write path.

A worker reports the visit lifecycle here:
  * ``POST /sightings``            — open a visit (match-or-create the identity)
  * ``POST /sightings/{id}/heartbeat`` — keep an open visit alive
  * ``POST /sightings/{id}/end``   — close a visit

Identity matching is central (decision A): the worker sends the embedding, the
server matches it against the in-memory gallery, and creates a new anonymous
person only when nothing matches — inside a serialized critical section so two
cameras seeing the same new face cannot create duplicate persons.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from sqlmodel import select

from ..db import get_session, get_thumbnails_dir
from ..gallery import get_gallery, match_create_lock
from ..models import Camera, FaceEmbedding, Person, Sighting
from ..serializers import person_response, sighting_response

router = APIRouter()

# Multi-exemplar enrollment: a person accumulates several pose exemplars (e.g.
# frontal + profile) instead of a single frozen reference. Within one continuous
# track every heartbeat is already bound to that track's person, so as the head
# turns we capture the new pose under the *same* identity — which is what keeps a
# side profile from spawning its own person.
EXEMPLAR_MAX = 12            # cap exemplars per person (bounds the gallery)
EXEMPLAR_NOVELTY = 0.50      # only store a frame whose pose isn't already covered
EXEMPLAR_MIN_SHARPNESS = 60.0  # never enroll a blurry crop as a reference


def _maybe_add_exemplar(session: Session, person_id: int, emb_bytes: bytes,
                        vec: np.ndarray, sharpness: float) -> None:
    """Store ``vec`` as a new pose exemplar for ``person_id`` when it is sharp,
    genuinely novel, and the per-person cap isn't reached."""
    if sharpness < EXEMPLAR_MIN_SHARPNESS:
        return
    gallery = get_gallery()
    if gallery.count_for_person(person_id) >= EXEMPLAR_MAX:
        return
    if gallery.best_for_person(person_id, vec) >= EXEMPLAR_NOVELTY:
        return  # this pose is already well represented
    session.add(FaceEmbedding(person_id=person_id, vector=emb_bytes))
    session.commit()
    gallery.add(person_id, vec)


@router.get("/sightings/active")
def active_sightings(session: Session = Depends(get_session)):
    """Currently-visible people (open sightings) — the Live/Now view."""
    rows = session.exec(
        select(Sighting).where(Sighting.ended_at == None)  # noqa: E711
        .order_by(Sighting.started_at.desc())
    ).all()
    cameras = {c.id: c for c in session.exec(select(Camera)).all()}
    out = []
    for s in rows:
        person = session.get(Person, s.person_id)
        entry = sighting_response(s, cameras.get(s.camera_id))
        entry["person"] = person_response(person) if person else None
        out.append(entry)
    return out


def _save_thumbnail(person_id: int, jpg_bytes: bytes) -> None:
    thumbs = get_thumbnails_dir()
    thumbs.mkdir(parents=True, exist_ok=True)
    (thumbs / f"{person_id}.jpg").write_bytes(jpg_bytes)


@router.post("/sightings", status_code=201)
def open_sighting(
    camera_id: int = Form(...),
    sharpness: float = Form(-1.0),
    embedding: UploadFile = File(...),
    crop: Optional[UploadFile] = File(None),
    session: Session = Depends(get_session),
):
    if session.get(Camera, camera_id) is None:
        raise HTTPException(404, detail="Camera not found")

    emb_bytes = embedding.file.read()
    vec = np.frombuffer(emb_bytes, dtype=np.float32)
    if vec.size == 0:
        raise HTTPException(422, detail="Empty embedding")
    crop_bytes = crop.file.read() if crop is not None else None

    gallery = get_gallery()
    thumb: Optional[bytes] = None
    # Serialize the match-or-create decision so concurrent reports of a brand-new
    # face produce exactly one person.
    with match_create_lock:
        person_id, score = gallery.match(vec)
        is_new = person_id is None
        if is_new:
            person = Person(best_sharpness=sharpness)
            session.add(person)
            session.commit()
            session.refresh(person)
            person_id = person.id
            session.add(FaceEmbedding(person_id=person_id, vector=emb_bytes))
            session.commit()
            gallery.add(person_id, vec)
            if crop_bytes:
                thumb = crop_bytes
        else:
            person = session.get(Person, person_id)
            if crop_bytes and sharpness > person.best_sharpness:
                person.best_sharpness = sharpness
                session.add(person)
                session.commit()
                thumb = crop_bytes
            # Re-matched an existing identity: capture this pose if it's new.
            _maybe_add_exemplar(session, person_id, emb_bytes, vec, sharpness)

        sighting = Sighting(person_id=person_id, camera_id=camera_id,
                            best_sharpness=sharpness)
        session.add(sighting)
        session.commit()
        session.refresh(sighting)
        sighting_id = sighting.id

    if thumb is not None:
        _save_thumbnail(person_id, thumb)

    return {"personId": person_id, "sightingId": sighting_id,
            "score": score, "isNew": is_new}


@router.post("/sightings/{sighting_id}/heartbeat")
def heartbeat(
    sighting_id: int,
    sharpness: float = Form(-1.0),
    embedding: Optional[UploadFile] = File(None),
    crop: Optional[UploadFile] = File(None),
    session: Session = Depends(get_session),
):
    sighting = session.get(Sighting, sighting_id)
    if sighting is None:
        raise HTTPException(404, detail="Sighting not found")
    if sighting.ended_at is not None:
        raise HTTPException(409, detail="Sighting already ended")

    sighting.last_seen = datetime.utcnow()
    thumb: Optional[tuple] = None
    if crop is not None and sharpness > sighting.best_sharpness:
        sighting.best_sharpness = sharpness
        person = session.get(Person, sighting.person_id)
        if sharpness > person.best_sharpness:
            person.best_sharpness = sharpness
            session.add(person)
            thumb = (person.id, crop.file.read())
    session.add(sighting)
    session.commit()
    # Grow this person's pose coverage from frames seen during the same visit.
    # This is what lets a profile turn enroll under the existing identity.
    if embedding is not None:
        emb_bytes = embedding.file.read()
        vec = np.frombuffer(emb_bytes, dtype=np.float32)
        if vec.size:
            _maybe_add_exemplar(session, sighting.person_id, emb_bytes, vec, sharpness)
    if thumb is not None:
        _save_thumbnail(*thumb)
    return {"ok": True}


@router.post("/sightings/{sighting_id}/end")
def end_sighting(sighting_id: int, session: Session = Depends(get_session)):
    sighting = session.get(Sighting, sighting_id)
    if sighting is None:
        raise HTTPException(404, detail="Sighting not found")
    if sighting.ended_at is None:
        sighting.ended_at = datetime.utcnow()
        session.add(sighting)
        session.commit()
    return {"ok": True}
