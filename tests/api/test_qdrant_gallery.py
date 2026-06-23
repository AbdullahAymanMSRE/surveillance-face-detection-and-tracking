import numpy as np
from qdrant_client import QdrantClient

from api.gallery import QdrantGallery


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _gallery() -> QdrantGallery:
    return QdrantGallery(QdrantClient(location=":memory:"))


def test_empty_gallery_returns_no_match():
    g = _gallery()
    person_id, score = g.match(_vec(1))
    assert person_id is None
    assert score == 0.0
    assert len(g) == 0


def test_add_then_exact_match():
    g = _gallery()
    v = _vec(2)
    g.add(7, v)
    person_id, score = g.match(v)
    assert person_id == 7
    assert score > 0.99
    assert len(g) == 1


def test_below_threshold_returns_none_with_score():
    g = _gallery()
    g.add(1, _vec(10))
    # An (almost) orthogonal vector scores below 0.28.
    person_id, score = g.match(_vec(99))
    assert person_id is None
    assert score < 0.28


def test_count_and_best_for_person():
    g = _gallery()
    g.add(3, _vec(4))
    g.add(3, _vec(5))
    g.add(4, _vec(6))
    assert g.count_for_person(3) == 2
    assert g.count_for_person(4) == 1
    assert g.best_for_person(3, _vec(4)) > 0.99
    assert g.best_for_person(99, _vec(4)) == 0.0


def test_all_vectors_by_person_and_reassign():
    g = _gallery()
    g.add(1, _vec(11))
    g.add(2, _vec(12))
    g.reassign_person(2, 1)
    by_person = g.all_vectors_by_person()
    assert set(by_person) == {1}
    assert by_person[1].shape == (2, 512)
