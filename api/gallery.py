"""Qdrant-backed face embedding gallery + central matching.

Embeddings are L2-normalized, so cosine similarity is a plain dot product.
The gallery is hidden behind the :class:`GalleryMatcher` protocol so the
implementation can be swapped without touching callers.
"""

import os
import threading
import uuid
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
from qdrant_client import QdrantClient, models

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

    def count_for_person(self, person_id: int) -> int:
        """How many exemplars this person currently holds."""
        ...

    def best_for_person(self, person_id: int, vector: np.ndarray) -> float:
        """Best cosine of ``vector`` against this person's own exemplars."""
        ...


class QdrantGallery:
    """Qdrant-backed :class:`GalleryMatcher`. Authoritative store for embeddings.

    Points carry a ``person_id`` payload; matching is a top-1 cosine search and
    per-person operations are payload-filtered queries.
    """

    def __init__(self, client: QdrantClient, threshold: float = DEFAULT_THRESHOLD,
                 collection: str = "faces", dim: int = 512):
        self.client = client
        self.threshold = threshold
        self.collection = collection
        self.dim = dim
        self.ensure_collection()

    def _person_filter(self, person_id: int) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(
            key="person_id", match=models.MatchValue(value=int(person_id)))])

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE),
            )

    def match(self, vector: np.ndarray) -> Tuple[Optional[int], float]:
        v = _normalize(vector)
        res = self.client.query_points(
            self.collection, query=v.tolist(), limit=1, with_payload=True)
        if not res.points:
            return None, 0.0
        top = res.points[0]
        score = float(top.score)
        if score >= self.threshold:
            return int(top.payload["person_id"]), score
        return None, score

    def add(self, person_id: int, vector: np.ndarray) -> None:
        v = _normalize(vector)
        self.client.upsert(self.collection, points=[models.PointStruct(
            id=str(uuid.uuid4()), vector=v.tolist(),
            payload={"person_id": int(person_id)})])

    def count_for_person(self, person_id: int) -> int:
        return self.client.count(
            self.collection, count_filter=self._person_filter(person_id),
            exact=True).count

    def best_for_person(self, person_id: int, vector: np.ndarray) -> float:
        v = _normalize(vector)
        res = self.client.query_points(
            self.collection, query=v.tolist(), limit=1,
            query_filter=self._person_filter(person_id))
        return float(res.points[0].score) if res.points else 0.0

    def all_vectors_by_person(self) -> Dict[int, np.ndarray]:
        by_person: Dict[int, list] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset,
                with_payload=True, with_vectors=True)
            for p in points:
                pid = int(p.payload["person_id"])
                by_person.setdefault(pid, []).append(
                    np.asarray(p.vector, dtype=np.float32))
            if offset is None:
                break
        return {pid: np.stack(vs) for pid, vs in by_person.items()}

    def reassign_person(self, from_id: int, to_id: int) -> None:
        self.client.set_payload(
            self.collection, payload={"person_id": int(to_id)},
            points=self._person_filter(from_id))

    def __len__(self) -> int:
        return self.client.count(self.collection, exact=True).count


# Process-wide gallery + the lock guarding match-or-create.
#
# The gallery has its own internal lock for matrix thread-safety, but the
# *decision* "no match found -> create a new person" is a read-then-write that
# two cameras can interleave. Callers wrap match -> (create) -> add in
# ``match_create_lock`` so a brand-new face seen by two cameras at once yields
# exactly one person.
_gallery: Optional[QdrantGallery] = None
match_create_lock = threading.Lock()


def _make_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url)


def get_gallery() -> QdrantGallery:
    global _gallery
    if _gallery is None:
        _gallery = QdrantGallery(_make_client())
    return _gallery


def reset_gallery() -> None:
    global _gallery
    _gallery = None
