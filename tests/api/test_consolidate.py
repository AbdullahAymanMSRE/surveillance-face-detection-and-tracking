import numpy as np
from sqlmodel import Session, select

from api.consolidate import consolidate_identities
from api.db import get_engine
from api.gallery import get_gallery
from api.models import Person, Sighting


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_consolidate_merges_similar_persons(client):
    gallery = get_gallery()
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    near = base.copy()
    near[1] = 0.5  # cosine with base ~0.89, well above MERGE_THRESHOLD (0.35)

    with Session(get_engine()) as s:
        p1, p2 = Person(), Person()
        s.add(p1); s.add(p2); s.commit(); s.refresh(p1); s.refresh(p2)
        id1, id2 = p1.id, p2.id
        # p1 has more visits -> it is the survivor
        s.add(Sighting(person_id=id1, camera_id=1))
        s.add(Sighting(person_id=id1, camera_id=1))
        s.add(Sighting(person_id=id2, camera_id=1))
        s.commit()

    gallery.add(id1, _unit(base))
    gallery.add(id2, _unit(near))

    with Session(get_engine()) as s:
        removed = consolidate_identities(s)

    assert removed == 1
    with Session(get_engine()) as s:
        remaining = {p.id for p in s.exec(select(Person)).all()}
    assert remaining == {id1}
    # both vectors now belong to the survivor
    assert gallery.count_for_person(id1) == 2
    assert gallery.count_for_person(id2) == 0
