"""In-memory face embedding gallery + central matching.

The authoritative store is SQLite (``FaceEmbedding`` rows); this module keeps a
hot in-memory NumPy matrix of those vectors so identity matching is a single
vectorized cosine op. It is hidden behind the :class:`GalleryMatcher` protocol so
a vector-DB-backed implementation (FAISS/Qdrant/…) can be dropped in later
without touching callers.

Embeddings are L2-normalized, so cosine similarity is a plain dot product.
"""

import threading
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np

DEFAULT_THRESHOLD = 0.28


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class GalleryMatcher(Protocol):
    def match(self, vector: np.ndarray) -> Tuple[Optional[int], float]:
        """Return (person_id, score) of the closest embedding, or (None, score)
        when nothing is within threshold (score is the best similarity seen)."""
        ...

    def add(self, person_id: int, vector: np.ndarray) -> None:
        """Add one embedding for a person to the gallery."""
        ...


class InMemoryGallery:
    """Brute-force NumPy implementation of :class:`GalleryMatcher`."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._ids: List[int] = []
        self._matrix: Optional[np.ndarray] = None  # [N, D] float32, L2-normalized
        self._lock = threading.Lock()

    def load(self, rows: Iterable[Tuple[int, bytes]]) -> None:
        """Replace gallery contents from (person_id, vector_bytes) rows."""
        ids: List[int] = []
        vecs: List[np.ndarray] = []
        for person_id, vector_bytes in rows:
            ids.append(person_id)
            vecs.append(_normalize(np.frombuffer(vector_bytes, dtype=np.float32)))
        with self._lock:
            self._ids = ids
            self._matrix = np.stack(vecs).astype(np.float32) if vecs else None

    def match(self, vector: np.ndarray) -> Tuple[Optional[int], float]:
        v = _normalize(vector)
        with self._lock:
            if self._matrix is None or not self._ids:
                return None, 0.0
            sims = self._matrix @ v  # [N]
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            if score >= self.threshold:
                return self._ids[idx], score
            return None, score

    def add(self, person_id: int, vector: np.ndarray) -> None:
        v = _normalize(vector)[None, :]
        with self._lock:
            self._ids.append(person_id)
            self._matrix = v if self._matrix is None else np.vstack([self._matrix, v])

    def count_for_person(self, person_id: int) -> int:
        """How many exemplars this person currently holds."""
        with self._lock:
            return sum(1 for pid in self._ids if pid == person_id)

    def best_for_person(self, person_id: int, vector: np.ndarray) -> float:
        """Best cosine of ``vector`` against this person's own exemplars.

        Used to decide whether a new frame shows a pose this person doesn't
        already cover (low value -> a genuinely new view worth storing)."""
        v = _normalize(vector)
        with self._lock:
            if self._matrix is None:
                return 0.0
            mask = np.array([pid == person_id for pid in self._ids])
            if not mask.any():
                return 0.0
            sims = self._matrix[mask] @ v
        return float(sims.max())

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)


# Process-wide gallery + the lock guarding match-or-create.
#
# The gallery has its own internal lock for matrix thread-safety, but the
# *decision* "no match found -> create a new person" is a read-then-write that
# two cameras can interleave. Callers wrap match -> (create) -> add in
# ``match_create_lock`` so a brand-new face seen by two cameras at once yields
# exactly one person.
_gallery: Optional[InMemoryGallery] = None
match_create_lock = threading.Lock()


def get_gallery() -> InMemoryGallery:
    global _gallery
    if _gallery is None:
        _gallery = InMemoryGallery()
    return _gallery


def reset_gallery() -> None:
    global _gallery
    _gallery = None
