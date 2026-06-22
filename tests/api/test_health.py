from sqlmodel import Session

from api import db
from api.models import Person


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_person_roundtrip(session):
    person = Person(label="Alice")
    session.add(person)
    session.commit()
    session.refresh(person)
    assert person.id is not None

    loaded = session.get(Person, person.id)
    assert loaded.label == "Alice"
