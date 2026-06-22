from typing import Optional, Tuple

import numpy as np
from sqlmodel import Session, select

from .models import FaceEmbedding, Person


def find_best_match(
    session: Session, embedding_bytes: bytes, threshold: float = 0.28
) -> Tuple[Optional[Person], float]:
    """Return (person, score) for the closest stored embedding, or (None,
    score) if nothing stored is within ``threshold``. ``score`` is the best
    cosine similarity found even when below threshold (0.0 if the DB is
    empty)."""
    query_vec = np.frombuffer(embedding_bytes, dtype=np.float32)
    rows = session.exec(select(FaceEmbedding)).all()

    best_score = 0.0
    best_person_id = None
    for row in rows:
        stored_vec = np.frombuffer(row.vector, dtype=np.float32)
        score = float(np.dot(query_vec, stored_vec))
        if score > best_score:
            best_score = score
            best_person_id = row.person_id

    if best_person_id is not None and best_score >= threshold:
        return session.get(Person, best_person_id), best_score
    return None, best_score
