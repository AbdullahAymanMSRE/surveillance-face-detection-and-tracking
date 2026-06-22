import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from .. import ml
from ..db import get_session, get_thumbnails_dir
from ..matching import find_best_match
from ..models import FaceEmbedding, Person
from ..serializers import person_response

router = APIRouter()


def _decode_and_embed(image: UploadFile):
    """Decode an uploaded image and run detect+align+embed.

    Returns (vector_bytes, crop_bgr, sharpness). Raises HTTPException(422) if
    no face is found — shared by /enroll and /people/{id}/embeddings."""
    image_bytes = image.file.read()
    frame = ml.decode_image(image_bytes)
    try:
        return ml.detect_and_embed(frame)
    except ml.NoFaceDetected:
        raise HTTPException(422, detail="No face detected")


def _save_thumbnail(person_id: int, crop_bgr) -> None:
    thumbs_dir = get_thumbnails_dir()
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumbs_dir / f"{person_id}.jpg"), crop_bgr)


def _create_person(
    session: Session, name: str, vector_bytes: bytes, crop_bgr, sharpness: float
) -> Person:
    person = Person(name=name, best_sharpness=sharpness)
    session.add(person)
    session.commit()
    session.refresh(person)
    session.add(FaceEmbedding(person_id=person.id, vector=vector_bytes))
    session.commit()
    _save_thumbnail(person.id, crop_bgr)
    return person


def _add_embedding(
    session: Session, person: Person, vector_bytes: bytes, crop_bgr, sharpness: float
) -> Person:
    session.add(FaceEmbedding(person_id=person.id, vector=vector_bytes))
    if sharpness > person.best_sharpness:
        person.best_sharpness = sharpness
        session.add(person)
        _save_thumbnail(person.id, crop_bgr)
    session.commit()
    session.refresh(person)
    return person


@router.post("/enroll", status_code=201)
def enroll(
    name: str = Form(...),
    image: UploadFile = File(...),
    force: bool = Form(False),
    session: Session = Depends(get_session),
):
    vector_bytes, crop, sharpness = _decode_and_embed(image)

    if not force:
        matched_person, score = find_best_match(session, vector_bytes)
        if matched_person is not None:
            if matched_person.name == name:
                person = _add_embedding(
                    session, matched_person, vector_bytes, crop, sharpness
                )
                return person_response(person)
            raise HTTPException(
                409,
                detail={
                    "existingPerson": person_response(matched_person),
                    "score": score,
                },
            )

    person = _create_person(session, name, vector_bytes, crop, sharpness)
    return person_response(person)


@router.post("/people/{person_id}/embeddings", status_code=201)
def add_embedding_route(
    person_id: int,
    image: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(404, detail="Person not found")

    vector_bytes, crop, sharpness = _decode_and_embed(image)
    person = _add_embedding(session, person, vector_bytes, crop, sharpness)
    return person_response(person)
