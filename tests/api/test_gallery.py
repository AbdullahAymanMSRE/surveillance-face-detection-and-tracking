import threading

import numpy as np

from api.gallery import InMemoryGallery, match_create_lock


def _vec(*xs) -> bytes:
    v = np.array(xs, dtype=np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


def test_match_above_threshold_returns_person():
    g = InMemoryGallery(threshold=0.5)
    g.load([(7, _vec(1.0, 0.0, 0.0))])
    pid, score = g.match(np.frombuffer(_vec(0.99, 0.01, 0.0), dtype=np.float32))
    assert pid == 7
    assert score > 0.5


def test_no_match_below_threshold():
    g = InMemoryGallery(threshold=0.9)
    g.load([(7, _vec(1.0, 0.0, 0.0))])
    pid, score = g.match(np.frombuffer(_vec(0.0, 1.0, 0.0), dtype=np.float32))
    assert pid is None
    assert score < 0.9


def test_empty_gallery_returns_none():
    g = InMemoryGallery()
    pid, score = g.match(np.frombuffer(_vec(1.0, 0.0), dtype=np.float32))
    assert pid is None
    assert score == 0.0


def test_add_makes_future_matches_hit():
    g = InMemoryGallery(threshold=0.5)
    assert g.match(np.frombuffer(_vec(1.0, 0.0), dtype=np.float32))[0] is None
    g.add(3, np.frombuffer(_vec(1.0, 0.0), dtype=np.float32))
    assert g.match(np.frombuffer(_vec(1.0, 0.0), dtype=np.float32))[0] == 3


def test_concurrent_match_or_create_yields_one_person():
    """Two threads see the same brand-new face at once; the serialized
    match-or-create section must create exactly one identity, not two."""
    g = InMemoryGallery(threshold=0.5)
    created = []
    next_id = [0]
    query = np.frombuffer(_vec(1.0, 0.0, 0.0), dtype=np.float32)

    def ingest():
        with match_create_lock:
            pid, _ = g.match(query)
            if pid is None:
                next_id[0] += 1
                pid = next_id[0]
                g.add(pid, query)
                created.append(pid)

    threads = [threading.Thread(target=ingest) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created) == 1
    assert len(g) == 1
