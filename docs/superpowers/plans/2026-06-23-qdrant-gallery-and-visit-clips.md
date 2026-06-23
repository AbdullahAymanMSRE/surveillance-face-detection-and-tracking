# Qdrant Gallery + Per-Visit Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-memory NumPy gallery with an authoritative Qdrant vector store, and record a short video clip of each visit shown on the person's timeline.

**Architecture:** Feature A swaps `InMemoryGallery` for a `QdrantGallery` implementing the same `GalleryMatcher` contract; Qdrant becomes authoritative and the `FaceEmbedding` SQLite table is removed. Feature B has the camera worker record a ≤5s downscaled clip per visit and upload it to the API on visit end, served back to the dashboard.

**Tech Stack:** FastAPI, SQLModel/SQLite, Qdrant (`qdrant-client`), OpenCV (`cv2.VideoWriter`), Next.js/React.

## Global Constraints

- Python deps via the existing `.venv` (built with **python3.12** — do NOT run `make install`). Run python as `.venv/bin/python`, pytest as `.venv/bin/python -m pytest`.
- `numpy>=1.26,<2` (already pinned) — do not introduce numpy 2 APIs.
- Embeddings are 512-d, float32, L2-normalized; cosine distance throughout.
- Match threshold stays `0.28` (`gallery.DEFAULT_THRESHOLD`); merge threshold stays `0.35` (`consolidate.MERGE_THRESHOLD`).
- Tests must run with **no external services**: the gallery uses `qdrant-client`'s in-process `:memory:` mode under tests (`QDRANT_URL=:memory:`).
- Running system uses a Qdrant server (Docker) at `QDRANT_URL` (default `http://localhost:6333`).
- Follow existing patterns: routers under `api/routers/`, serializers in `api/serializers.py`, frequent commits, TDD.

---

## Task 1: QdrantGallery class

**Files:**
- Modify: `requirements.txt`
- Modify: `api/gallery.py`
- Test: `tests/api/test_qdrant_gallery.py` (Create)

**Interfaces:**
- Consumes: nothing (foundational).
- Produces:
  - `class QdrantGallery` with: `__init__(self, client, threshold=DEFAULT_THRESHOLD, collection="faces", dim=512)`, `ensure_collection() -> None`, `match(vector: np.ndarray) -> Tuple[Optional[int], float]`, `add(person_id: int, vector: np.ndarray) -> None`, `count_for_person(person_id: int) -> int`, `best_for_person(person_id: int, vector: np.ndarray) -> float`, `all_vectors_by_person() -> Dict[int, np.ndarray]`, `reassign_person(from_id: int, to_id: int) -> None`, `__len__() -> int`.
  - Extended `GalleryMatcher` Protocol declaring `match`, `add`, `count_for_person`, `best_for_person`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add after the `sqlmodel` line:

```
qdrant-client>=1.12.0
```

Install it into the existing venv:

Run: `.venv/bin/pip install "qdrant-client>=1.12.0"`
Expected: `Successfully installed qdrant-client-...`

- [ ] **Step 2: Write the failing test**

Create `tests/api/test_qdrant_gallery.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/api/test_qdrant_gallery.py -q`
Expected: FAIL with `ImportError: cannot import name 'QdrantGallery'`.

- [ ] **Step 4: Implement QdrantGallery**

In `api/gallery.py`, add imports at the top (after the existing `import threading` / numpy import):

```python
import os
import uuid
from typing import Dict

from qdrant_client import QdrantClient, models
```

Extend the Protocol (replace the existing `GalleryMatcher` class body):

```python
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
```

Add the new class below `InMemoryGallery` (keep `InMemoryGallery` for now — it and its unit tests `tests/api/test_gallery.py` are removed in Task 5; do not delete in this task):

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/api/test_qdrant_gallery.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt api/gallery.py tests/api/test_qdrant_gallery.py
git commit -m "feat(gallery): add QdrantGallery matcher backed by qdrant-client"
```

---

## Task 2: Wire QdrantGallery as the process gallery

**Files:**
- Modify: `api/gallery.py` (singleton factory)
- Modify: `api/main.py` (startup)
- Modify: `tests/api/conftest.py` (set `QDRANT_URL=:memory:`)

**Interfaces:**
- Consumes: `QdrantGallery` (Task 1).
- Produces: `get_gallery() -> QdrantGallery` (process-wide singleton, connects per `QDRANT_URL`); startup calls `get_gallery()` instead of `.load(...)`.

- [ ] **Step 1: Update the singleton factory**

In `api/gallery.py`, replace the `get_gallery()` / `reset_gallery()` block at the bottom:

```python
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
```

- [ ] **Step 2: Update conftest to use in-memory Qdrant**

In `tests/api/conftest.py`, inside the `client` fixture, add the env var alongside the existing `setenv` calls (after the `FACE_API_ENABLE_SUPERVISOR` line):

```python
    monkeypatch.setenv("QDRANT_URL", ":memory:")
```

- [ ] **Step 3: Update startup**

In `api/main.py`, remove the `FaceEmbedding` import (line 10) and the `select` import if now unused (keep `Session` only if still used — after this change it is not, so remove `from sqlmodel import Session, select`). Replace the `on_startup` body's gallery block:

```python
@app.on_event("startup")
def on_startup() -> None:
    db.init_db()
    # Connect to Qdrant and ensure the 'faces' collection exists. Qdrant is the
    # authoritative embedding store, so there is nothing to rebuild from SQLite.
    get_gallery()
    if os.environ.get("FACE_API_ENABLE_REAPER", "1") == "1":
        start_reaper(
            interval_secs=float(os.environ.get("FACE_API_REAPER_INTERVAL", "5")),
            timeout_secs=float(os.environ.get("FACE_API_REAPER_TIMEOUT", "15")),
        )
    if os.environ.get("FACE_API_ENABLE_SUPERVISOR", "1") == "1":
        get_supervisor().start_monitor(
            interval=float(os.environ.get("FACE_API_SUPERVISOR_INTERVAL", "5")),
        )
```

(Leave the rest of `main.py` unchanged.)

- [ ] **Step 4: Run the health/people tests**

Run: `.venv/bin/python -m pytest tests/api/test_health.py tests/api/test_people.py -q`
Expected: PASS. (`sightings.py` still inserts `FaceEmbedding` rows AND calls `gallery.add`; that double-write is harmless and removed in Task 3. The startup no longer reads `FaceEmbedding`.)

- [ ] **Step 5: Commit**

```bash
git add api/gallery.py api/main.py tests/api/conftest.py
git commit -m "feat(gallery): make Qdrant the process gallery; tests use in-memory mode"
```

---

## Task 3: Drop FaceEmbedding writes from the ingestion path

**Files:**
- Modify: `api/routers/sightings.py`

**Interfaces:**
- Consumes: `get_gallery().add/match/count_for_person/best_for_person` (Tasks 1–2).
- Produces: `_maybe_add_exemplar(person_id, vec, sharpness)` — `session` and `emb_bytes` parameters removed; embeddings now go only to the gallery.

**Note on tests:** No test file counts `FaceEmbedding` rows for the ingestion path — `test_sightings.py` asserts behavior through the API, so it stays green unchanged. Only `tests/api/test_search.py` inserts `FaceEmbedding`, and it keeps working until Task 5 (the model still exists). Do not edit tests in this task; just confirm the suite is green.

- [ ] **Step 1: Update `_maybe_add_exemplar`**

In `api/routers/sightings.py`, replace the function:

```python
def _maybe_add_exemplar(person_id: int, vec: np.ndarray, sharpness: float) -> None:
    """Store ``vec`` as a new pose exemplar for ``person_id`` when it is sharp,
    genuinely novel, and the per-person cap isn't reached."""
    if sharpness < EXEMPLAR_MIN_SHARPNESS:
        return
    gallery = get_gallery()
    if gallery.count_for_person(person_id) >= EXEMPLAR_MAX:
        return
    if gallery.best_for_person(person_id, vec) >= EXEMPLAR_NOVELTY:
        return  # this pose is already well represented
    gallery.add(person_id, vec)
```

- [ ] **Step 2: Update the new-person branch in `open_sighting`**

Replace these lines in the `is_new` branch:

```python
            session.add(FaceEmbedding(person_id=person_id, vector=emb_bytes))
            session.commit()
            gallery.add(person_id, vec)
```

with:

```python
            gallery.add(person_id, vec)
```

And update the re-matched branch call site:

```python
            _maybe_add_exemplar(person_id, vec, sharpness)
```

- [ ] **Step 3: Update the heartbeat call site**

In `heartbeat`, replace:

```python
        if vec.size:
            _maybe_add_exemplar(session, sighting.person_id, emb_bytes, vec, sharpness)
```

with:

```python
        if vec.size:
            _maybe_add_exemplar(sighting.person_id, vec, sharpness)
```

- [ ] **Step 4: Drop the FaceEmbedding import**

Change the models import to drop `FaceEmbedding`:

```python
from ..models import Camera, Person, Sighting
```

- [ ] **Step 5: Run the affected suites (no test edits expected)**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py tests/api/test_gallery.py tests/api/test_search.py -q`
Expected: PASS unchanged. `test_sightings.py` exercises the API and does not inspect `FaceEmbedding`; `test_search.py` still inserts `FaceEmbedding` (the model exists until Task 5) and still passes. If any assertion genuinely depends on the removed insert, stop and report — do not invent test changes here.

- [ ] **Step 6: Commit**

```bash
git add api/routers/sightings.py
git commit -m "refactor(sightings): store embeddings only in Qdrant, not FaceEmbedding"
```

---

## Task 4: Consolidation against Qdrant

**Files:**
- Modify: `api/consolidate.py`
- Create: `tests/api/test_consolidate.py` (no consolidation test exists yet — add one)

**Interfaces:**
- Consumes: `QdrantGallery.all_vectors_by_person()`, `reassign_person()` (Task 1).
- Produces: `consolidate_identities(session, merge_threshold=MERGE_THRESHOLD) -> int` (unchanged signature; now reads vectors from Qdrant).

- [ ] **Step 1: Rewrite the data-access + merge to use the gallery**

In `api/consolidate.py`:

Change the imports:

```python
from .db import get_thumbnails_dir
from .gallery import get_gallery
from .models import Person, Sighting
```

Delete `_vectors_by_person` entirely. Keep `_max_cross_similarity`. At the top of `consolidate_identities`, replace `mats = _vectors_by_person(session)` with:

```python
    gallery = get_gallery()
    mats = gallery.all_vectors_by_person()
```

In the per-loser loop, replace the `FaceEmbedding` reassignment block:

```python
            for e in session.exec(
                    select(FaceEmbedding).where(FaceEmbedding.person_id == loser)).all():
                e.person_id = keeper
                session.add(e)
```

with:

```python
            gallery.reassign_person(loser, keeper)
```

Replace the tail (rebuild) block:

```python
    if removed:
        session.commit()
        rows = session.exec(select(FaceEmbedding)).all()
        get_gallery().load((r.person_id, r.vector) for r in rows)
    return removed
```

with:

```python
    if removed:
        session.commit()
    return removed
```

- [ ] **Step 2: Add a consolidation test (seeded via the gallery)**

Create `tests/api/test_consolidate.py`. It seeds two persons whose exemplars are similar enough to merge (cross cosine ≥ 0.35), runs `consolidate_identities`, and asserts the loser's person row is gone and its vectors were reassigned. The `client` fixture wires the in-memory Qdrant gallery and a temp DB.

```python
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
```

- [ ] **Step 3: Run the consolidation test**

Run: `.venv/bin/python -m pytest tests/api/test_consolidate.py -q`
Expected: PASS. (Write this test first and watch it fail against the old `FaceEmbedding`-based `consolidate.py` before applying Step 1, per TDD; if you apply Step 1 first, it should pass directly.)

- [ ] **Step 4: Commit**

```bash
git add api/consolidate.py tests/api/test_consolidate.py
git commit -m "refactor(consolidate): merge identities via Qdrant payload reassignment"
```

---

## Task 5: Remove FaceEmbedding model, InMemoryGallery, dead matcher; add migration

**Files:**
- Modify: `api/models.py` (remove `FaceEmbedding`)
- Modify: `api/gallery.py` (remove `InMemoryGallery`)
- Modify: `tests/api/test_search.py` (seed via gallery, count gallery not table)
- Delete: `api/matching.py`, `tests/api/test_gallery.py`
- Create: `scripts/migrate_to_qdrant.py`

**Interfaces:**
- Consumes: `get_gallery().add` (Task 1).
- Produces: a runnable one-time migration `scripts/migrate_to_qdrant.py`. After this task the only `GalleryMatcher` implementation is `QdrantGallery`; the `GalleryMatcher` Protocol stays.

- [ ] **Step 1: Update `tests/api/test_search.py` to drop FaceEmbedding**

In `tests/api/test_search.py`, change the import to drop `FaceEmbedding`:

```python
from api.models import Person
```

In `_seed_person_from_image`, remove the `FaceEmbedding` insert (keep the `Person` row and the `get_gallery().add`):

```python
def _seed_person_from_image(image_bytes: bytes) -> int:
    """Enroll a real face into DB + gallery the way the pipeline would."""
    frame = ml.decode_image(image_bytes)
    vector_bytes, _crop, sharpness = ml.detect_and_embed(frame)
    with Session(get_engine()) as s:
        person = Person(best_sharpness=sharpness)
        s.add(person)
        s.commit()
        s.refresh(person)
        person_id = person.id
    get_gallery().add(person_id, np.frombuffer(vector_bytes, dtype=np.float32))
    return person_id
```

In `test_search_does_not_write`, count gallery points instead of `FaceEmbedding` rows:

```python
def test_search_does_not_write(client, face_known_1_bytes):
    _seed_person_from_image(face_known_1_bytes)
    gallery = get_gallery()
    with Session(get_engine()) as s:
        persons_before = len(s.exec(select(Person)).all())
    embeds_before = len(gallery)

    client.post("/search", files={"image": ("x.jpg", face_known_1_bytes, "image/jpeg")})

    with Session(get_engine()) as s:
        assert len(s.exec(select(Person)).all()) == persons_before
    assert len(gallery) == embeds_before
```

- [ ] **Step 2: Remove the model, InMemoryGallery, and dead code**

In `api/models.py`, delete the entire `FaceEmbedding` class. In `api/gallery.py`, delete the entire `InMemoryGallery` class (keep `_normalize`, the `GalleryMatcher` Protocol, `QdrantGallery`, and the singleton block). Remove any now-unused imports. Delete the dead matcher and the InMemoryGallery unit tests:

```bash
git rm api/matching.py tests/api/test_gallery.py
```

Then verify nothing still references the removed names:

Run: `grep -rn "FaceEmbedding\|InMemoryGallery\|find_best_match" api/ tests/ --include=*.py`
Expected: no output (zero references remain).

- [ ] **Step 3: Add the migration script**

Create `scripts/migrate_to_qdrant.py`:

```python
#!/usr/bin/env python3
"""One-time: copy existing FaceEmbedding rows from app.db into Qdrant.

Run once against a database created before the Qdrant migration, with
QDRANT_URL pointing at the running Qdrant server. Reads the legacy table via
raw SQL so it does not depend on the (now-removed) FaceEmbedding model.

    QDRANT_URL=http://localhost:6333 .venv/bin/python scripts/migrate_to_qdrant.py
"""
import sqlite3
import sys

import numpy as np

from api.db import get_db_path
from api.gallery import get_gallery


def main() -> int:
    db_path = get_db_path()
    if not db_path.exists():
        print(f"no database at {db_path}; nothing to migrate")
        return 0
    con = sqlite3.connect(str(db_path))
    try:
        names = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'").fetchall()}
        if "faceembedding" not in names:
            print("no faceembedding table; nothing to migrate")
            return 0
        rows = con.execute("select person_id, vector from faceembedding").fetchall()
    finally:
        con.close()

    gallery = get_gallery()
    for person_id, vector in rows:
        vec = np.frombuffer(vector, dtype=np.float32)
        gallery.add(int(person_id), vec)
    print(f"migrated {len(rows)} embeddings into Qdrant ({len(gallery)} total points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full API suite**

Run: `.venv/bin/python -m pytest tests/api -q`
Expected: PASS (all tests green; `FaceEmbedding` fully gone).

- [ ] **Step 5: Commit**

```bash
git add api/models.py api/gallery.py tests/api/test_search.py scripts/migrate_to_qdrant.py
git rm api/matching.py tests/api/test_gallery.py
git commit -m "refactor: remove FaceEmbedding, InMemoryGallery, dead matcher; add Qdrant migration"
```

---

## Task 6: Qdrant deployment wiring (Makefile + docs)

**Files:**
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `web/.env.local.example` (no change needed) — instead document `QDRANT_URL` near other env vars in README

**Interfaces:** none (ops/docs only).

- [ ] **Step 1: Add Makefile targets**

In `Makefile`, add to `.PHONY` (`qdrant qdrant-stop`) and add the targets:

```make
qdrant:
	docker run -d --name face-qdrant -p 6333:6333 \
	  -v $(PWD)/api/data/qdrant:/qdrant/storage qdrant/qdrant

qdrant-stop:
	-docker rm -f face-qdrant
```

Add their help lines under `help:`:

```make
	@echo "make qdrant        - start the Qdrant vector DB (Docker, port 6333)"
```

- [ ] **Step 2: Document the new run step**

In `README.md` "Quick start", add Qdrant before `make api`:

```markdown
make qdrant                        # start the Qdrant vector store (Docker)
make api                           # FastAPI backend + supervisor on :8000
```

In the env-vars paragraph, add: `` `QDRANT_URL` (vector store; default `http://localhost:6333`, `:memory:` for tests) ``. Update the "Future work" line that promised a vector-DB GalleryMatcher (it is now implemented).

- [ ] **Step 3: Smoke-test against a real Qdrant**

Run:
```bash
make qdrant
sleep 3
QDRANT_URL=http://localhost:6333 .venv/bin/python scripts/migrate_to_qdrant.py
```
Expected: `migrated N embeddings into Qdrant (...)` (N may be 0 if the dev DB has no legacy rows — that is fine).

- [ ] **Step 4: Commit**

```bash
git add Makefile README.md
git commit -m "ops: add make qdrant target and document QDRANT_URL"
```

---

## Task 7: Clip storage plumbing (model + serializer)

**Files:**
- Modify: `api/db.py` (`get_clips_dir`)
- Modify: `api/models.py` (`Sighting.has_clip`)
- Modify: `api/serializers.py` (`hasClip`, `clipUrl`)
- Test: `tests/api/test_sightings.py` (serializer assertion)

**Interfaces:**
- Produces: `get_clips_dir() -> Path`; `Sighting.has_clip: bool`; `sighting_response` includes `hasClip: bool` and `clipUrl: Optional[str]`.

- [ ] **Step 1: Write the failing serializer test**

Add to `tests/api/test_sightings.py`:

```python
def test_sighting_response_includes_clip_fields():
    from api.models import Sighting
    from api.serializers import sighting_response
    s = Sighting(id=5, person_id=1, camera_id=1, has_clip=True)
    body = sighting_response(s, None)
    assert body["hasClip"] is True
    assert body["clipUrl"] == "/sightings/5/clip"
    s2 = Sighting(id=6, person_id=1, camera_id=1, has_clip=False)
    assert sighting_response(s2, None)["clipUrl"] is None
```

- [ ] **Step 2: Run it to verify failure**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py::test_sighting_response_includes_clip_fields -q`
Expected: FAIL (`Sighting` has no `has_clip` / KeyError `hasClip`).

- [ ] **Step 3: Add the storage dir and model field**

In `api/db.py`, after `get_thumbnails_dir`:

```python
def get_clips_dir() -> Path:
    return get_data_dir() / "clips"
```

In `api/models.py`, add to `Sighting`:

```python
    has_clip: bool = False
```

- [ ] **Step 4: Extend the serializer**

In `api/serializers.py`, add to the dict returned by `sighting_response`:

```python
        "hasClip": sighting.has_clip,
        "clipUrl": f"/sightings/{sighting.id}/clip" if sighting.has_clip else None,
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py::test_sighting_response_includes_clip_fields -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/db.py api/models.py api/serializers.py tests/api/test_sightings.py
git commit -m "feat(sightings): add has_clip field, clips dir, and clip serializer fields"
```

---

## Task 8: Clip upload + download endpoints

**Files:**
- Modify: `api/routers/sightings.py`
- Test: `tests/api/test_sightings.py`

**Interfaces:**
- Consumes: `get_clips_dir()` (Task 7).
- Produces: `POST /sightings/{id}/clip` (multipart `clip`), `GET /sightings/{id}/clip` (`video/mp4`).

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_sightings.py` (the file uses the `client` fixture; follow its existing helpers for creating a camera + sighting):

```python
def test_clip_upload_and_download(client):
    # Create a camera and an open sighting via the ingestion path used elsewhere
    # in this file (reuse the existing helper that opens a sighting).
    cam = client.post("/cameras", json={"name": "c", "source": "0"}).json()
    sighting_id = _open_a_sighting(client, cam["id"])  # existing helper in this file

    fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
    up = client.post(f"/sightings/{sighting_id}/clip",
                     files={"clip": ("v.mp4", fake_mp4, "video/mp4")})
    assert up.status_code == 200

    got = client.get(f"/sightings/{sighting_id}/clip")
    assert got.status_code == 200
    assert got.headers["content-type"] == "video/mp4"
    assert got.content == fake_mp4

    # 404 for a sighting with no clip
    assert client.get("/sightings/999999/clip").status_code == 404
```

If no `_open_a_sighting` helper exists, open one inline using the same multipart shape the other tests in this file use for `POST /sightings` (a real embedding + crop from the fixtures), and read `sightingId` from the response.

- [ ] **Step 2: Run it to verify failure**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py::test_clip_upload_and_download -q`
Expected: FAIL (404/405 — endpoints not defined).

- [ ] **Step 3: Implement the endpoints**

In `api/routers/sightings.py`, add imports:

```python
from fastapi.responses import FileResponse
from ..db import get_session, get_thumbnails_dir, get_clips_dir
```

(Merge `get_clips_dir` into the existing `..db` import line.)

Add a size cap constant near the other constants:

```python
MAX_CLIP_BYTES = 20 * 1024 * 1024  # 20 MB ceiling per visit clip
```

Add the endpoints at the end of the file:

```python
@router.post("/sightings/{sighting_id}/clip")
def upload_clip(
    sighting_id: int,
    clip: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    sighting = session.get(Sighting, sighting_id)
    if sighting is None:
        raise HTTPException(404, detail="Sighting not found")
    data = clip.file.read()
    if len(data) > MAX_CLIP_BYTES:
        raise HTTPException(413, detail="Clip too large")
    clips = get_clips_dir()
    clips.mkdir(parents=True, exist_ok=True)
    (clips / f"{sighting_id}.mp4").write_bytes(data)
    sighting.has_clip = True
    session.add(sighting)
    session.commit()
    return {"ok": True}


@router.get("/sightings/{sighting_id}/clip")
def get_clip(sighting_id: int):
    path = get_clips_dir() / f"{sighting_id}.mp4"
    if not path.exists():
        raise HTTPException(404, detail="No clip for this sighting")
    return FileResponse(path, media_type="video/mp4")
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py::test_clip_upload_and_download -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/sightings.py tests/api/test_sightings.py
git commit -m "feat(sightings): upload and serve per-visit MP4 clips"
```

---

## Task 9: Worker-side clip recording

**Files:**
- Create: `pipeline/clip.py`
- Modify: `pipeline_node.py`
- Test: `tests/test_clip_recorder.py` (Create)

**Interfaces:**
- Produces: `class ClipRecorder(max_frames=50, fps=10, width=640)` with `maybe_add(frame: np.ndarray, now: float) -> None`, `encode(path: str) -> bool`, `frame_count -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clip_recorder.py`:

```python
import os

import numpy as np

from pipeline.clip import ClipRecorder


def _frame(w=1280, h=720):
    return (np.ones((h, w, 3), dtype=np.uint8) * 127)


def test_respects_fps_cadence():
    r = ClipRecorder(max_frames=50, fps=10, width=640)
    r.maybe_add(_frame(), now=100.0)
    r.maybe_add(_frame(), now=100.02)  # too soon (<0.1s) -> dropped
    r.maybe_add(_frame(), now=100.2)   # ok
    assert r.frame_count == 2


def test_caps_frame_count():
    r = ClipRecorder(max_frames=3, fps=10, width=640)
    t = 0.0
    for _ in range(20):
        r.maybe_add(_frame(), now=t)
        t += 0.2
    assert r.frame_count == 3


def test_downscales_to_width():
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    r.maybe_add(_frame(1280, 720), now=0.0)
    assert r._frames[0].shape[1] == 640  # width
    assert r._frames[0].shape[0] == 360  # height preserves aspect


def test_encode_writes_a_playable_file(tmp_path):
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    t = 0.0
    for _ in range(5):
        r.maybe_add(_frame(), now=t)
        t += 0.2
    out = str(tmp_path / "clip.mp4")
    assert r.encode(out) is True
    assert os.path.getsize(out) > 0


def test_encode_empty_returns_false(tmp_path):
    r = ClipRecorder()
    assert r.encode(str(tmp_path / "x.mp4")) is False
```

- [ ] **Step 2: Run it to verify failure**

Run: `.venv/bin/python -m pytest tests/test_clip_recorder.py -q`
Expected: FAIL (`ModuleNotFoundError: pipeline.clip`).

- [ ] **Step 3: Implement ClipRecorder**

Create `pipeline/clip.py`:

```python
"""Per-visit clip recorder for the headless worker.

Buffers downscaled frames at a capped cadence/length during a visit, then
encodes them to an MP4 the worker uploads to the API on visit end.
"""

from typing import List

import cv2
import numpy as np


class ClipRecorder:
    def __init__(self, max_frames: int = 50, fps: int = 10, width: int = 640):
        self.max_frames = max_frames
        self.fps = fps
        self.width = width
        self._frames: List[np.ndarray] = []
        self._last_ts: float = -1e9

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.width:
            return frame
        new_h = int(round(h * self.width / w))
        return cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_AREA)

    def maybe_add(self, frame: np.ndarray, now: float) -> None:
        if len(self._frames) >= self.max_frames:
            return
        if (now - self._last_ts) < (1.0 / self.fps):
            return
        self._last_ts = now
        self._frames.append(self._downscale(frame))

    def encode(self, path: str) -> bool:
        if not self._frames:
            return False
        h, w = self._frames[0].shape[:2]
        for fourcc_name in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(path, fourcc, float(self.fps), (w, h))
            if not writer.isOpened():
                writer.release()
                continue
            for f in self._frames:
                writer.write(f)
            writer.release()
            return True
        return False
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_clip_recorder.py -q`
Expected: PASS. (If `avc1` is unavailable on this machine, `encode` silently falls through to `mp4v` and still passes.)

- [ ] **Step 5: Wire the recorder into the worker**

In `pipeline_node.py`:

Add the import near the top (after `from pipeline.core import build_arcface_core`):

```python
from pipeline.clip import ClipRecorder
```

Add constants near `END_GRACE_SECS`:

```python
CLIP_SECS = 5.0
CLIP_FPS = 10
CLIP_WIDTH = 640
```

Inside `main()`, alongside `open_sightings`/`last_present`, add:

```python
    recorders: Dict[int, ClipRecorder] = {}
```

Add a clip-upload helper next to `_post_end`:

```python
    def _post_clip(sighting_id: int, recorder: ClipRecorder) -> None:
        import tempfile, os
        if recorder.frame_count == 0:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            if not recorder.encode(tmp.name):
                return
            with open(tmp.name, "rb") as fh:
                client.post(f"{api}/sightings/{sighting_id}/clip", timeout=30,
                            files={"clip": ("clip.mp4", fh.read(), "video/mp4")})
        except (OSError, httpx.HTTPError) as e:
            print(f"[node] clip upload failed: {e}", flush=True)
        finally:
            os.unlink(tmp.name)
```

In the per-event loop, when a NEW sighting opens, create a recorder. Replace:

```python
                else:
                    sid = _post_open(ev)
                    if sid is not None:
                        open_sightings[ev.track_id] = sid
```

with:

```python
                else:
                    sid = _post_open(ev)
                    if sid is not None:
                        open_sightings[ev.track_id] = sid
                        recorders[ev.track_id] = ClipRecorder(
                            max_frames=int(CLIP_SECS * CLIP_FPS),
                            fps=CLIP_FPS, width=CLIP_WIDTH)
```

After the `for tid in track_ids: last_present[tid] = now` loop, feed frames to active recorders:

```python
            for tid in track_ids:
                rec = recorders.get(tid)
                if rec is not None:
                    rec.maybe_add(frame, now)
```

In the close-visit loop, upload before dropping. Replace:

```python
            for tid in list(open_sightings):
                if now - last_present.get(tid, 0.0) > END_GRACE_SECS:
                    _post_end(open_sightings.pop(tid))
                    last_present.pop(tid, None)
```

with:

```python
            for tid in list(open_sightings):
                if now - last_present.get(tid, 0.0) > END_GRACE_SECS:
                    sid = open_sightings.pop(tid)
                    rec = recorders.pop(tid, None)
                    if rec is not None:
                        _post_clip(sid, rec)
                    _post_end(sid)
                    last_present.pop(tid, None)
```

In the `finally:` block, upload any still-open recorders before ending:

```python
    finally:
        for tid, sid in list(open_sightings.items()):
            rec = recorders.pop(tid, None)
            if rec is not None:
                _post_clip(sid, rec)
            _post_end(sid)
        core.stop()
        cap.release()
        client.close()
```

(Replace the existing `for sid in open_sightings.values(): _post_end(sid)` loop with the above.)

- [ ] **Step 6: Run the worker test + full suite**

Run: `.venv/bin/python -m pytest tests/test_clip_recorder.py tests/api -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/clip.py pipeline_node.py tests/test_clip_recorder.py
git commit -m "feat(worker): record and upload a short clip per visit"
```

---

## Task 10: Dashboard — show clips on the person timeline

**Files:**
- Modify: `web/lib/api.ts`
- Modify: `web/app/people/[id]/page.tsx`

**Interfaces:**
- Consumes: `GET /sightings/{id}/clip`, serializer `hasClip`/`clipUrl` (Tasks 7–8).
- Produces: `clipUrl(sightingId: number): string`; timeline rows render `<video>` when `hasClip`.

- [ ] **Step 1: Add the client helper + type field**

In `web/lib/api.ts`, near `thumbnailUrl`/`previewUrl`:

```typescript
export function clipUrl(sightingId: number): string {
  return absUrl(`/sightings/${sightingId}/clip`);
}
```

In the sighting/appearance TypeScript type used by the person detail page, add:

```typescript
  hasClip: boolean;
```

(Match the existing type's location and naming; the field name is `hasClip` to match the serializer.)

- [ ] **Step 2: Render the clip in the timeline**

In `web/app/people/[id]/page.tsx`, import `clipUrl`:

```typescript
import { /* existing imports */, clipUrl } from "@/lib/api";
```

In each timeline/appearance row, after the camera/time details, conditionally render the video:

```tsx
{appearance.hasClip && (
  <video
    controls
    preload="none"
    src={clipUrl(appearance.id)}
    className="mt-2 w-full max-w-md rounded border border-white/10"
  />
)}
```

(Use the existing variable name for the per-visit object — `appearance.id` must be the **sighting** id; confirm the timeline maps over sightings, which carry `id` + `hasClip`.)

- [ ] **Step 3: Verify the build compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no type errors. (If the timeline type lives in a shared interface, ensure `hasClip` was added there so `.hasClip` type-checks.)

- [ ] **Step 4: Manual end-to-end verification**

With Qdrant + API + web running (API on the chosen port with `FACE_API_SELF_URL` and the dashboard's `NEXT_PUBLIC_API_URL` matching), let a face appear and leave so a visit closes. Open the person's page; the closed visit should show an inline, playable clip. If it does not play, check the worker log for the fourcc it used (`avc1` vs `mp4v`) — note the codec in the PR if `mp4v` was the fallback.

- [ ] **Step 5: Commit**

```bash
git add web/lib/api.ts web/app/people/[id]/page.tsx
git commit -m "feat(web): show per-visit clip on the person timeline"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Qdrant `QdrantGallery` + protocol → Task 1. Singleton/startup → Task 2. Ingestion stops writing `FaceEmbedding` → Task 3. Consolidation via Qdrant (+ new `test_consolidate.py`) → Task 4. Remove `FaceEmbedding` + `InMemoryGallery` + `matching.py` + delete `test_gallery.py` + update `test_search.py` + migration → Task 5. Docker/Makefile/README/`QDRANT_URL` → Task 6.
- **Pre-flight resolutions (user-approved):** `InMemoryGallery` and its `test_gallery.py` are removed in Task 5 (single authoritative matcher). No consolidation test existed, so Task 4 adds one. Only `test_search.py` referenced `FaceEmbedding`; it is updated in Task 5.
- Clips: `has_clip`/clips dir/serializer → Task 7. Upload+serve endpoints (range via `FileResponse`) → Task 8. Worker `ClipRecorder` (≤5s/10fps/640px, `avc1`→`mp4v`, upload on end) → Task 9. Person-timeline `<video>` → Task 10.
- Risks (hard Qdrant dependency, codec fallback, storage growth, `match_create_lock` retained) reflected in Global Constraints / Task 9 Step 4 / Task 8 size cap.

**Placeholder scan:** No TBD/TODO; every code step shows full code. Two steps reference existing test helpers (`_open_a_sighting`, the consolidation test) by instruction because their exact names live in the repo; both include a fallback ("if no helper exists, …").

**Type consistency:** `hasClip`/`clipUrl` consistent across serializer (Task 7), endpoint (Task 8), client (Task 10). `_maybe_add_exemplar(person_id, vec, sharpness)` signature consistent across its definition and both call sites (Task 3). `ClipRecorder` API (`maybe_add`, `encode`, `frame_count`) consistent between Task 9 test and implementation and worker wiring.
