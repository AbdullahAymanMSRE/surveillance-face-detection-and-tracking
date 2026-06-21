from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..db import get_session, get_thumbnails_dir
from ..models import Person
from ..serializers import person_response

router = APIRouter()


@router.get("/people")
def list_people(session: Session = Depends(get_session)):
    people = session.exec(select(Person)).all()
    return [person_response(p) for p in people]


@router.get("/people/{person_id}/thumbnail")
def get_thumbnail(person_id: int, session: Session = Depends(get_session)):
    person = session.get(Person, person_id)
    if person is None:
        raise HTTPException(404, detail="Person not found")
    path = get_thumbnails_dir() / f"{person_id}.jpg"
    if not path.exists():
        raise HTTPException(404, detail="No thumbnail for this person")
    return FileResponse(str(path), media_type="image/jpeg")
