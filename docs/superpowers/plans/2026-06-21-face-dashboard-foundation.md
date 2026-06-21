# Face Dashboard Foundation + Enrollment Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Next.js dashboard + FastAPI backend on top of the existing
face detection/recognition code, and ship one working vertical slice:
enrolling a person by name + webcam photo, with duplicate-face detection.

**Architecture:** FastAPI (`api/`) is the sole owner of a SQLite database and
all ML inference (YOLO detection, YuNet alignment, ArcFace embedding — all
reused as-is from `face_extraction/` and `face_recognition/`). Next.js
(`web/`) is a pure frontend: it never touches the database directly, only
calls FastAPI's REST API.

**Tech Stack:** FastAPI, SQLModel (SQLite), pytest + httpx for backend tests;
Next.js (TypeScript, App Router, Tailwind) for the frontend.

## Global Constraints

- Backend requires **Python 3.10+** (existing code already uses `str | Path`
  union syntax). The system `python3` on this machine is 3.9.6 — install and
  use a 3.10+ interpreter for the venv in Task 1 (e.g. `brew install
  python@3.11`, already done on this machine at `/opt/homebrew/bin/python3.11`).
- FastAPI is the **sole owner of the database** — Next.js never opens SQLite
  or the filesystem directly; every read/write goes through FastAPI's REST API.
- ArcFace embeddings are 512-d, L2-normalized; cosine similarity is a plain
  dot product. Default match threshold: **0.28** (matches the existing
  pipeline's `--recognizer arcface` default in `live_pipeline.py`).
- `force=true` on `POST /enroll` always creates a new `Person`, bypassing both
  the conflict check and the same-name auto-merge.
- Single snapshot per enrollment for this slice (no multi-shot averaging).
- No automated frontend tests for this slice — verify Next.js pages by manual
  click-through, per the spec's testing approach.
- Test fixtures already exist and are verified against the real pipeline:
  `tests/api/fixtures/face_known_1.jpg` and `face_known_2.jpg` (two different
  real photos of the same person — YOLO detects + ArcFace embeds both
  successfully, cosine similarity 0.447, comfortably above the 0.28
  threshold) and `no_face.jpg` (a blank image YOLO finds zero boxes in).

---

## File Structure

```
requirements.txt                # MODIFY: add fastapi, uvicorn, sqlmodel, etc.
pytest.ini                      # CREATE: pythonpath=. so `import api` works
.gitignore                      # MODIFY: add api/data/

api/
  __init__.py                   # CREATE: empty
  main.py                       # CREATE: FastAPI app, CORS, startup, /health
  db.py                         # CREATE: engine/session management
  models.py                     # CREATE: Person, FaceEmbedding tables
  serializers.py                # CREATE: shared Person -> dict response shape
  ml.py                         # CREATE: detection+alignment+embedding wrapper
  matching.py                   # CREATE: cosine-similarity duplicate matching
  routers/
    __init__.py                 # CREATE: empty
    people.py                   # CREATE: GET /people, GET /people/{id}/thumbnail
    enroll.py                   # CREATE: POST /enroll, POST /people/{id}/embeddings

tests/api/
  __init__.py                   # CREATE: empty
  conftest.py                   # CREATE: client/session fixtures, fixture bytes
  fixtures/                     # ALREADY EXISTS (committed in a prior step)
    face_known_1.jpg
    face_known_2.jpg
    no_face.jpg
  test_health.py                # CREATE
  test_ml.py                    # CREATE
  test_matching.py               # CREATE
  test_enroll.py                 # CREATE
  test_embeddings.py             # CREATE
  test_people.py                  # CREATE

web/                             # CREATE: Next.js app (via create-next-app)
  lib/api.ts                     # CREATE: typed fetch helpers
  app/page.tsx                   # CREATE: dashboard
  app/enroll/page.tsx             # CREATE: enrollment page
  .env.local                      # CREATE: NEXT_PUBLIC_API_URL (gitignored by default)
  .env.local.example              # CREATE: committed template
```

---

### Task 1: Backend scaffold — FastAPI app, DB models, health check

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Create: `api/__init__.py`
- Create: `api/db.py`
- Create: `api/models.py`
- Create: `api/main.py`
- Create: `api/routers/__init__.py`
- Create: `api/routers/people.py` (stub router only)
- Create: `api/routers/enroll.py` (stub router only)
- Create: `tests/api/__init__.py`
- Create: `tests/api/conftest.py`
- Test: `tests/api/test_health.py`

**Interfaces:**
- Produces: `api.db.get_engine() -> Engine`, `api.db.reset_engine() -> None`,
  `api.db.init_db() -> None`, `api.db.get_session() -> Iterator[Session]`,
  `api.db.get_data_dir() -> Path`, `api.db.get_thumbnails_dir() -> Path`.
  `api.models.Person(id, name, best_sharpness, created_at)`,
  `api.models.FaceEmbedding(id, person_id, vector, created_at)`.
  `api.main.app` (FastAPI instance). Test fixtures `client` and `session`
  (both isolate each test in its own `tmp_path` via `FACE_API_DATA_DIR`).

- [ ] **Step 1: Set up a Python 3.10+ virtual environment**

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python --version   # must print 3.11.x
```

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Append to the end of the existing file:

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
sqlmodel>=0.0.16
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: installs without errors (numpy stays `<2` per the existing pin).

- [ ] **Step 4: Create `pytest.ini` at repo root**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 5: Create `api/__init__.py`** (empty file)

- [ ] **Step 6: Write `api/models.py`**

```python
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Person(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    best_sharpness: float = -1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FaceEmbedding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="person.id")
    vector: bytes
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

- [ ] **Step 7: Write `api/db.py`**

```python
import os
from pathlib import Path
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

_engine = None


def get_data_dir() -> Path:
    return Path(os.environ.get(
        "FACE_API_DATA_DIR",
        Path(__file__).resolve().parent / "data",
    ))


def get_db_path() -> Path:
    return get_data_dir() / "app.db"


def get_thumbnails_dir() -> Path:
    return get_data_dir() / "thumbnails"


def get_engine():
    global _engine
    if _engine is None:
        get_data_dir().mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{get_db_path()}",
            connect_args={"check_same_thread": False},
        )
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None


def init_db() -> None:
    from . import models  # noqa: F401  (registers tables on SQLModel.metadata)
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
```

- [ ] **Step 8: Create `api/routers/__init__.py`** (empty file)

- [ ] **Step 9: Write stub `api/routers/people.py`**

```python
from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 10: Write stub `api/routers/enroll.py`**

```python
from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 11: Write `api/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .routers import enroll, people

app = FastAPI(title="Face Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(people.router)
app.include_router(enroll.router)
```

- [ ] **Step 12: Create `tests/api/__init__.py`** (empty file)

- [ ] **Step 13: Write `tests/api/conftest.py`**

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from api import db
from api.main import app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_API_DATA_DIR", str(tmp_path))
    db.reset_engine()
    with TestClient(app) as c:
        yield c
    db.reset_engine()


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_API_DATA_DIR", str(tmp_path))
    db.reset_engine()
    db.init_db()
    with Session(db.get_engine()) as s:
        yield s
    db.reset_engine()


@pytest.fixture
def face_known_1_bytes() -> bytes:
    return (FIXTURES_DIR / "face_known_1.jpg").read_bytes()


@pytest.fixture
def face_known_2_bytes() -> bytes:
    return (FIXTURES_DIR / "face_known_2.jpg").read_bytes()


@pytest.fixture
def no_face_bytes() -> bytes:
    return (FIXTURES_DIR / "no_face.jpg").read_bytes()
```

- [ ] **Step 14: Write the failing test `tests/api/test_health.py`**

```python
from sqlmodel import Session

from api import db
from api.models import Person


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_person_roundtrip(session):
    person = Person(name="Alice")
    session.add(person)
    session.commit()
    session.refresh(person)
    assert person.id is not None

    loaded = session.get(Person, person.id)
    assert loaded.name == "Alice"
```

- [ ] **Step 15: Run the test**

```bash
pytest tests/api/test_health.py -v
```

Expected: both tests PASS.

- [ ] **Step 16: Commit**

```bash
git add requirements.txt pytest.ini api/ tests/api/__init__.py tests/api/conftest.py tests/api/test_health.py
git commit -m "Add FastAPI backend scaffold with health check and DB models"
```

---

### Task 2: ML wrapper — detect, align, embed a face from raw image bytes

**Files:**
- Create: `api/ml.py`
- Test: `tests/api/test_ml.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `api.ml.decode_image(data: bytes) -> np.ndarray`,
  `api.ml.detect_and_embed(image_bgr: np.ndarray) -> tuple[bytes, np.ndarray, float]`
  (returns `(vector_bytes, crop_bgr, sharpness)`), `api.ml.NoFaceDetected`
  (exception raised when no face is found).

- [ ] **Step 1: Write the failing test `tests/api/test_ml.py`**

```python
import numpy as np
import pytest

from api import ml


def test_detect_and_embed_finds_face(face_known_1_bytes):
    frame = ml.decode_image(face_known_1_bytes)
    vector_bytes, crop, sharpness = ml.detect_and_embed(frame)

    vector = np.frombuffer(vector_bytes, dtype=np.float32)
    assert vector.shape == (512,)
    assert crop.shape[:2] == (112, 112)
    assert sharpness > 0


def test_detect_and_embed_raises_on_no_face(no_face_bytes):
    frame = ml.decode_image(no_face_bytes)
    with pytest.raises(ml.NoFaceDetected):
        ml.detect_and_embed(frame)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_ml.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.ml'`.

- [ ] **Step 3: Write `api/ml.py`**

```python
import sys
from pathlib import Path

import cv2
import numpy as np

_API_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "face_recognition"))

from arcface_onnx import ArcFaceONNXRecognizer  # noqa: E402
from face_align import FaceProcessor  # noqa: E402

DETECTOR_PATH = _REPO_ROOT / "face_extraction" / "last.pt"

_detector = None
_processor = None
_recognizer = None


class NoFaceDetected(Exception):
    pass


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        _detector = YOLO(str(DETECTOR_PATH))
    return _detector


def get_processor():
    global _processor
    if _processor is None:
        _processor = FaceProcessor(template="arcface")
    return _processor


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = ArcFaceONNXRecognizer()
    return _recognizer


def detect_and_embed(image_bgr: np.ndarray):
    """Detect the largest face, align it, and embed it.

    Returns (vector_bytes, crop_bgr, sharpness). Raises NoFaceDetected if no
    face is found.
    """
    detector = get_detector()
    results = detector.predict(image_bgr, conf=0.3, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    if not boxes:
        raise NoFaceDetected()

    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    processor = get_processor()
    crop, sharpness, _aligned = processor.process(image_bgr, box)

    recognizer = get_recognizer()
    embedding = recognizer.embed(crop)
    vector_bytes = embedding.numpy().astype(np.float32).tobytes()
    return vector_bytes, crop, sharpness
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_ml.py -v
```

Expected: both tests PASS. (First run is slower — it loads the YOLO and
ONNX models from disk.)

- [ ] **Step 5: Commit**

```bash
git add api/ml.py tests/api/test_ml.py
git commit -m "Add ML wrapper: detect, align, embed a face from image bytes"
```

---

### Task 3: Duplicate-face matching logic

**Files:**
- Create: `api/matching.py`
- Test: `tests/api/test_matching.py`

**Interfaces:**
- Consumes: `api.models.Person`, `api.models.FaceEmbedding` (Task 1).
- Produces: `api.matching.find_best_match(session: Session, embedding_bytes: bytes, threshold: float = 0.28) -> tuple[Person | None, float]`.

- [ ] **Step 1: Write the failing test `tests/api/test_matching.py`**

```python
import numpy as np

from api.matching import find_best_match
from api.models import FaceEmbedding, Person


def _vec_bytes(values):
    return np.array(values, dtype=np.float32).tobytes()


def test_no_match_on_empty_db(session):
    person, score = find_best_match(session, _vec_bytes([1, 0, 0, 0]))
    assert person is None
    assert score == 0.0


def test_match_above_threshold(session):
    alice = Person(name="Alice")
    session.add(alice)
    session.commit()
    session.refresh(alice)
    session.add(FaceEmbedding(person_id=alice.id, vector=_vec_bytes([1, 0, 0, 0])))
    session.commit()

    person, score = find_best_match(
        session, _vec_bytes([0.9, 0.1, 0, 0]), threshold=0.28
    )
    assert person is not None
    assert person.id == alice.id
    assert score >= 0.28


def test_no_match_below_threshold(session):
    alice = Person(name="Alice")
    session.add(alice)
    session.commit()
    session.refresh(alice)
    session.add(FaceEmbedding(person_id=alice.id, vector=_vec_bytes([1, 0, 0, 0])))
    session.commit()

    person, score = find_best_match(
        session, _vec_bytes([0, 1, 0, 0]), threshold=0.28
    )
    assert person is None
    assert score < 0.28
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_matching.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.matching'`.

- [ ] **Step 3: Write `api/matching.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_matching.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/matching.py tests/api/test_matching.py
git commit -m "Add cosine-similarity duplicate-face matching"
```

---

### Task 4: `POST /enroll` with duplicate detection

**Files:**
- Create: `api/serializers.py`
- Modify: `api/routers/enroll.py`
- Test: `tests/api/test_enroll.py`

**Interfaces:**
- Consumes: `api.ml.decode_image`, `api.ml.detect_and_embed`, `api.ml.NoFaceDetected`
  (Task 2); `api.matching.find_best_match` (Task 3); `api.db.get_session`,
  `api.db.get_thumbnails_dir` (Task 1); `api.models.Person`, `api.models.FaceEmbedding`.
- Produces: `api.serializers.person_response(person: Person) -> dict` (the
  shared `{id, name, thumbnailUrl}` response shape, also used by Task 6's
  `GET /people`). `POST /enroll` accepting form fields `name` (str), `image`
  (file), `force` (bool, default false). Helper functions `_decode_and_embed`,
  `_save_thumbnail`, `_create_person`, `_add_embedding` (module-private in
  `enroll.py`, reused by Task 5 within the same file).

- [ ] **Step 1: Write the failing tests `tests/api/test_enroll.py`**

```python
def _enroll(client, name, image_bytes, force=None):
    data = {"name": name}
    if force is not None:
        data["force"] = str(force).lower()
    return client.post(
        "/enroll",
        data=data,
        files={"image": ("face.jpg", image_bytes, "image/jpeg")},
    )


def test_enroll_creates_person(client, face_known_1_bytes):
    resp = _enroll(client, "Alice", face_known_1_bytes)
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Alice"
    assert body["thumbnailUrl"] == f"/people/{body['id']}/thumbnail"

    thumb = client.get(body["thumbnailUrl"])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"


def test_enroll_no_face_returns_422(client, no_face_bytes):
    resp = _enroll(client, "Ghost", no_face_bytes)
    assert resp.status_code == 422


def test_enroll_duplicate_face_different_name_conflicts(
    client, face_known_1_bytes, face_known_2_bytes
):
    first = _enroll(client, "Alice", face_known_1_bytes)
    assert first.status_code == 201

    second = _enroll(client, "Bob", face_known_2_bytes)
    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["existingPerson"]["name"] == "Alice"
    assert detail["score"] >= 0.28


def test_enroll_duplicate_face_same_name_merges(
    client, face_known_1_bytes, face_known_2_bytes
):
    first = _enroll(client, "Alice", face_known_1_bytes)
    alice_id = first.json()["id"]

    second = _enroll(client, "Alice", face_known_2_bytes)
    assert second.status_code == 201
    assert second.json()["id"] == alice_id

    people = client.get("/people").json()
    assert len(people) == 1


def test_enroll_force_creates_new_person_anyway(
    client, face_known_1_bytes, face_known_2_bytes
):
    first = _enroll(client, "Alice", face_known_1_bytes)
    alice_id = first.json()["id"]

    second = _enroll(client, "Bob", face_known_2_bytes, force=True)
    assert second.status_code == 201
    assert second.json()["id"] != alice_id

    people = client.get("/people").json()
    assert len(people) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_enroll.py -v
```

Expected: FAIL — `/enroll` returns 404 (no route registered yet) and
`/people` also 404 (Task 6 not done yet, but these tests need it; see note
below).

> Note: `test_enroll_duplicate_face_same_name_merges` and
> `test_enroll_force_creates_new_person_anyway` call `GET /people`, which
> doesn't exist until Task 6. Implement Task 6's `GET /people` route stub
> (just the route, returning `select(Person)` results) **before** running
> this task's tests, or run this task's other 3 tests first and come back to
> these 2 after Task 6. This plan implements Task 6 next specifically to
> unblock this — if executing tasks out of order, do Task 6 before finishing
> Task 4's test run.

- [ ] **Step 3: Write `api/serializers.py`**

```python
from .models import Person


def person_response(person: Person) -> dict:
    return {
        "id": person.id,
        "name": person.name,
        "thumbnailUrl": f"/people/{person.id}/thumbnail",
    }
```

- [ ] **Step 4: Write `api/routers/enroll.py`**

```python
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
```

- [ ] **Step 5: Implement Task 6's `GET /people` now, so this task's tests can run**

Write `api/routers/people.py` (full version, also satisfies Task 6 — Task 6
below just adds the thumbnail route and its own tests on top of this):

```python
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..db import get_session
from ..models import Person
from ..serializers import person_response

router = APIRouter()


@router.get("/people")
def list_people(session: Session = Depends(get_session)):
    people = session.exec(select(Person)).all()
    return [person_response(p) for p in people]
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/api/test_enroll.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add api/serializers.py api/routers/enroll.py api/routers/people.py tests/api/test_enroll.py
git commit -m "Add POST /enroll with duplicate-face detection and merge/force handling"
```

---

### Task 5: `POST /people/{id}/embeddings`

**Files:**
- Modify: `api/routers/enroll.py`
- Test: `tests/api/test_embeddings.py`

**Interfaces:**
- Consumes: `_add_embedding`, `_decode_and_embed` from Task 4 (already in
  `api/routers/enroll.py`); `api.serializers.person_response` (Task 4).
- Produces: `POST /people/{person_id}/embeddings` accepting form field `image`
  (file). 201 `{id, name, thumbnailUrl}` on success, 404 if person doesn't
  exist, 422 if no face detected.

- [ ] **Step 1: Write the failing tests `tests/api/test_embeddings.py`**

```python
def _enroll(client, name, image_bytes):
    return client.post(
        "/enroll",
        data={"name": name},
        files={"image": ("face.jpg", image_bytes, "image/jpeg")},
    )


def _add_embedding(client, person_id, image_bytes):
    return client.post(
        f"/people/{person_id}/embeddings",
        files={"image": ("face.jpg", image_bytes, "image/jpeg")},
    )


def test_add_embedding_to_existing_person(client, face_known_1_bytes, face_known_2_bytes):
    alice_id = _enroll(client, "Alice", face_known_1_bytes).json()["id"]

    resp = _add_embedding(client, alice_id, face_known_2_bytes)
    assert resp.status_code == 201
    assert resp.json()["id"] == alice_id

    people = client.get("/people").json()
    assert len(people) == 1


def test_add_embedding_unknown_person_404(client, face_known_1_bytes):
    resp = _add_embedding(client, 9999, face_known_1_bytes)
    assert resp.status_code == 404


def test_add_embedding_no_face_422(client, face_known_1_bytes, no_face_bytes):
    alice_id = _enroll(client, "Alice", face_known_1_bytes).json()["id"]

    resp = _add_embedding(client, alice_id, no_face_bytes)
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_embeddings.py -v
```

Expected: FAIL with 404 Not Found on `/people/{id}/embeddings` (route doesn't
exist yet).

- [ ] **Step 3: Add the route to `api/routers/enroll.py`**

Append to the end of `api/routers/enroll.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_embeddings.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/enroll.py tests/api/test_embeddings.py
git commit -m "Add POST /people/{id}/embeddings to attach a photo to an existing person"
```

---

### Task 6: `GET /people` listing + `GET /people/{id}/thumbnail`

**Files:**
- Modify: `api/routers/people.py` (add thumbnail route; list route already
  exists from Task 4 Step 4)
- Test: `tests/api/test_people.py`

**Interfaces:**
- Consumes: `api.db.get_thumbnails_dir` (Task 1).
- Produces: `GET /people/{person_id}/thumbnail` returning the JPEG file, 404
  if the person or its thumbnail doesn't exist.

- [ ] **Step 1: Write the failing tests `tests/api/test_people.py`**

```python
def _enroll(client, name, image_bytes):
    return client.post(
        "/enroll",
        data={"name": name},
        files={"image": ("face.jpg", image_bytes, "image/jpeg")},
    )


def test_list_people_empty(client):
    resp = client.get("/people")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_people_after_enroll(client, face_known_1_bytes):
    _enroll(client, "Alice", face_known_1_bytes)

    resp = client.get("/people")
    assert resp.status_code == 200
    people = resp.json()
    assert len(people) == 1
    assert people[0]["name"] == "Alice"
    assert people[0]["thumbnailUrl"] == f"/people/{people[0]['id']}/thumbnail"


def test_thumbnail_serves_jpeg(client, face_known_1_bytes):
    person_id = _enroll(client, "Alice", face_known_1_bytes).json()["id"]

    resp = client.get(f"/people/{person_id}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_thumbnail_404_unknown_person(client):
    resp = client.get("/people/9999/thumbnail")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_people.py -v
```

Expected: the two list tests PASS already (route exists from Task 4 Step 4);
the two thumbnail tests FAIL with 404 (no route registered).

- [ ] **Step 3: Add the thumbnail route to `api/routers/people.py`**

Replace the file's contents with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_people.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full backend test suite**

```bash
pytest tests/api/ -v
```

Expected: all tests across every file PASS (18 tests total across Tasks 1–6).

- [ ] **Step 6: Commit**

```bash
git add api/routers/people.py tests/api/test_people.py
git commit -m "Add GET /people listing and thumbnail serving"
```

---

### Task 7: Gitignore + manual smoke test of the running backend

**Files:**
- Modify: `.gitignore`

**Interfaces:** None (manual verification task, no new code).

- [ ] **Step 1: Add the runtime data directory to `.gitignore`**

Append to `.gitignore`:

```
# FastAPI runtime data (SQLite DB + thumbnails) — not source.
api/data/
```

- [ ] **Step 2: Start the backend for real**

```bash
uvicorn api.main:app --reload --port 8000
```

Expected console output includes `Application startup complete.` with no
tracebacks.

- [ ] **Step 3: Manually verify enrollment end-to-end with curl**

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/enroll \
  -F "name=Alice" \
  -F "image=@tests/api/fixtures/face_known_1.jpg;type=image/jpeg"
```

Expected: JSON response like
`{"id":1,"name":"Alice","thumbnailUrl":"/people/1/thumbnail"}`.

```bash
curl -s http://localhost:8000/people
curl -s -o /tmp/thumb.jpg -w "%{http_code}\n" http://localhost:8000/people/1/thumbnail
open /tmp/thumb.jpg   # confirm it's a real face crop
```

Expected: `/people` lists Alice; the thumbnail downloads as `200` and opens
as a valid JPEG of the enrolled face.

- [ ] **Step 4: Stop the server and clean up the manual-test database**

```bash
rm -rf api/data
```

(Press Ctrl-C in the terminal running uvicorn first.)

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "Ignore FastAPI runtime data directory"
```

---

### Task 8: Next.js scaffold + typed API client

**Files:**
- Create: `web/` (via `create-next-app`)
- Create: `web/lib/api.ts`
- Create: `web/.env.local.example`
- Create: `web/.env.local` (local only, gitignored by Next.js's default `.gitignore`)

**Interfaces:**
- Produces: `Person` type, `EnrollConflictError`, `NoFaceDetectedError`,
  `listPeople()`, `enroll(name, image, force?)`, `addEmbedding(personId, image)`,
  `thumbnailUrl(person)`. These are consumed by Tasks 9 and 10.

- [ ] **Step 1: Scaffold the Next.js app**

```bash
npx create-next-app@latest web --typescript --eslint --tailwind --app --no-src-dir --import-alias "@/*" --use-npm
```

When prompted interactively, accept the defaults shown above (the flags
should make it non-interactive, but confirm App Router + Tailwind + TS are
selected if asked).

- [ ] **Step 2: Verify the scaffold**

```bash
ls web/app/page.tsx web/app/layout.tsx web/package.json
```

Expected: all three files exist.

- [ ] **Step 3: Create `web/.env.local.example`**

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 4: Create `web/.env.local`** (copy of the example, for local dev)

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 5: Write `web/lib/api.ts`**

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Person = {
  id: number;
  name: string;
  thumbnailUrl: string;
};

export type EnrollConflict = {
  existingPerson: Person;
  score: number;
};

export class EnrollConflictError extends Error {
  existingPerson: Person;
  score: number;
  constructor(conflict: EnrollConflict) {
    super("Face matches an existing person under a different name");
    this.existingPerson = conflict.existingPerson;
    this.score = conflict.score;
  }
}

export class NoFaceDetectedError extends Error {
  constructor() {
    super("No face detected in the image");
  }
}

export function thumbnailUrl(person: Person): string {
  return `${API_URL}${person.thumbnailUrl}`;
}

export async function listPeople(): Promise<Person[]> {
  const res = await fetch(`${API_URL}/people`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to list people: ${res.status}`);
  return res.json();
}

async function postImage(
  path: string,
  image: Blob,
  fields: Record<string, string> = {}
): Promise<Person> {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    form.append(key, value);
  }
  form.append("image", image, "capture.jpg");

  const res = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  if (res.status === 422) throw new NoFaceDetectedError();
  if (res.status === 409) {
    const body = await res.json();
    throw new EnrollConflictError(body.detail);
  }
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export function enroll(name: string, image: Blob, force = false): Promise<Person> {
  return postImage("/enroll", image, { name, force: String(force) });
}

export function addEmbedding(personId: number, image: Blob): Promise<Person> {
  return postImage(`/people/${personId}/embeddings`, image);
}
```

- [ ] **Step 6: Verify the project builds**

```bash
cd web && npm run lint && npm run build
```

Expected: both commands exit with code 0 (lint may print warnings but no
errors; build produces a `.next/` directory with no type errors — `api.ts`
is not yet imported by any page so it must type-check standalone).

- [ ] **Step 7: Commit**

```bash
cd web && git add -A . ../web ..
git -C .. add web .gitignore
git -C .. commit -m "Scaffold Next.js dashboard app with typed FastAPI client"
```

(Run `git add web .gitignore` and `git commit` from the repo root — adjust
the exact commands to your shell; the intent is to commit the entire new
`web/` directory plus any root `.gitignore` changes `create-next-app` made.)

---

### Task 9: Dashboard page — list enrolled people

**Files:**
- Create: `web/app/page.tsx`

**Interfaces:**
- Consumes: `listPeople`, `thumbnailUrl`, `Person` from `web/lib/api.ts` (Task 8).

- [ ] **Step 1: Write `web/app/page.tsx`**

```tsx
import Link from "next/link";
import { listPeople, thumbnailUrl } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const people = await listPeople();

  return (
    <main className="mx-auto max-w-4xl p-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Enrolled people</h1>
        <Link
          href="/enroll"
          className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
        >
          Enroll new person
        </Link>
      </div>

      {people.length === 0 ? (
        <p className="text-gray-500">No one enrolled yet.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
          {people.map((person) => (
            <div key={person.id} className="rounded border p-3 text-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={thumbnailUrl(person)}
                alt={person.name}
                className="mx-auto mb-2 h-32 w-32 rounded object-cover"
              />
              <p className="font-medium">{person.name}</p>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
```

- [ ] **Step 2: Manually verify the empty state**

```bash
# terminal 1, from repo root
rm -rf api/data && uvicorn api.main:app --reload --port 8000
# terminal 2
cd web && npm run dev
```

Open `http://localhost:3000` in a browser. Expected: "No one enrolled yet."
and an "Enroll new person" button (the `/enroll` link 404s for now — that's
Task 10).

- [ ] **Step 3: Manually verify the populated state**

```bash
curl -s -X POST http://localhost:8000/enroll \
  -F "name=Alice" \
  -F "image=@tests/api/fixtures/face_known_1.jpg;type=image/jpeg"
```

Reload `http://localhost:3000`. Expected: a card showing Alice's thumbnail
and name.

- [ ] **Step 4: Clean up the manual-test database**

Stop both dev servers (Ctrl-C), then:

```bash
rm -rf api/data
```

- [ ] **Step 5: Commit**

```bash
git add web/app/page.tsx
git commit -m "Add dashboard page listing enrolled people"
```

---

### Task 10: Enroll page — webcam capture + duplicate-conflict dialog

**Files:**
- Create: `web/app/enroll/page.tsx`

**Interfaces:**
- Consumes: `enroll`, `addEmbedding`, `EnrollConflictError`,
  `NoFaceDetectedError` from `web/lib/api.ts` (Task 8).

- [ ] **Step 1: Write `web/app/enroll/page.tsx`**

```tsx
"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  EnrollConflictError,
  NoFaceDetectedError,
  addEmbedding,
  enroll,
} from "@/lib/api";

export default function EnrollPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [name, setName] = useState("");
  const [captured, setCaptured] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<EnrollConflictError | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
  }

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob) {
        setCaptured(blob);
        setPreviewUrl(URL.createObjectURL(blob));
      }
    }, "image/jpeg");
  }

  function retake() {
    setCaptured(null);
    setPreviewUrl(null);
  }

  async function save(force = false) {
    if (!captured || !name) return;
    setSubmitting(true);
    setError(null);
    try {
      await enroll(name, captured, force);
      router.push("/");
    } catch (err) {
      if (err instanceof NoFaceDetectedError) {
        setError("No face detected, try again.");
      } else if (err instanceof EnrollConflictError) {
        setConflict(err);
      } else {
        setError("Something went wrong, please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function resolveConflictAddPhoto() {
    if (!captured || !conflict) return;
    setSubmitting(true);
    try {
      await addEmbedding(conflict.existingPerson.id, captured);
      router.push("/");
    } catch {
      setError("Something went wrong, please try again.");
    } finally {
      setSubmitting(false);
      setConflict(null);
    }
  }

  async function resolveConflictEnrollAnyway() {
    setConflict(null);
    await save(true);
  }

  return (
    <main className="mx-auto max-w-md p-8">
      <h1 className="mb-4 text-2xl font-semibold">Enroll a new person</h1>

      <input
        className="mb-4 w-full rounded border p-2"
        placeholder="Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />

      {!previewUrl ? (
        <div>
          <video ref={videoRef} className="w-full rounded bg-black" muted />
          <div className="mt-2 flex gap-2">
            <button
              className="rounded bg-gray-200 px-4 py-2"
              onClick={startCamera}
              type="button"
            >
              Start camera
            </button>
            <button
              className="rounded bg-blue-600 px-4 py-2 text-white"
              onClick={capture}
              type="button"
            >
              Capture
            </button>
          </div>
        </div>
      ) : (
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={previewUrl} alt="Captured face" className="w-full rounded" />
          <div className="mt-2 flex gap-2">
            <button className="rounded bg-gray-200 px-4 py-2" onClick={retake} type="button">
              Retake
            </button>
            <button
              className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
              onClick={() => save(false)}
              disabled={!name || submitting}
              type="button"
            >
              Save
            </button>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} className="hidden" />

      {error && <p className="mt-4 text-red-600">{error}</p>}

      {conflict && (
        <div className="mt-4 rounded border border-yellow-400 bg-yellow-50 p-4">
          <p className="mb-3">
            This looks like <strong>{conflict.existingPerson.name}</strong> (score{" "}
            {conflict.score.toFixed(2)}). Is this the same person?
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={() => setConflict(null)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={resolveConflictAddPhoto}
              type="button"
            >
              Add photo to {conflict.existingPerson.name} instead
            </button>
            <button
              className="rounded bg-gray-200 px-3 py-1"
              onClick={resolveConflictEnrollAnyway}
              type="button"
            >
              No, enroll as new person
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
```

- [ ] **Step 2: Verify the project still builds**

```bash
cd web && npm run lint && npm run build
```

Expected: both exit with code 0.

- [ ] **Step 3: Manual click-through — happy path**

```bash
# terminal 1, from repo root
rm -rf api/data && uvicorn api.main:app --reload --port 8000
# terminal 2
cd web && npm run dev
```

Open `http://localhost:3000/enroll`. Click "Start camera", grant webcam
permission, click "Capture", confirm a preview appears, type a name (e.g.
"Test Person"), click "Save". Expected: redirect to `/` with a new card
showing your face and name.

- [ ] **Step 4: Manual click-through — no-face case**

On `/enroll`, point the camera at a blank wall (or otherwise no face in
frame) before capturing, type a name, click "Save". Expected: inline message
"No face detected, try again." and you stay on the page.

- [ ] **Step 5: Manual click-through — conflict dialog, all three actions**

With "Test Person" already enrolled from Step 3, go to `/enroll`, capture
your face again, but type a **different** name (e.g. "Someone Else"), click
"Save". Expected: the yellow conflict box appears showing "This looks like
Test Person ...".

Test each button once (repeating Steps 3 capture beforehand as needed since
a successful action navigates away):
- "Cancel" → dialog closes, you stay on the page, no new card created.
- "Add photo to Test Person instead" → redirects to `/`, still only the one
  "Test Person" card (no new person created).
- "No, enroll as new person" → redirects to `/`, a **second** card now
  exists with the name you typed ("Someone Else"), distinct from "Test Person".

- [ ] **Step 6: Clean up the manual-test database**

Stop both dev servers (Ctrl-C), then:

```bash
rm -rf api/data
```

- [ ] **Step 7: Commit**

```bash
git add web/app/enroll/page.tsx
git commit -m "Add enroll page with webcam capture and duplicate-conflict dialog"
```

---

## Self-Review Notes

- **Spec coverage:** repo layout (Task 1, 8), DB schema incl. `best_sharpness`
  (Task 1), `/enroll` + duplicate detection incl. same-name merge and `force`
  (Task 4), `/people/{id}/embeddings` (Task 5), `/people` + thumbnail (Task 6),
  `.gitignore` for runtime data (Task 7), Next.js scaffold + API client
  (Task 8), dashboard UI (Task 9), enroll UI + conflict dialog (Task 10).
  Testing approach matches the spec: pytest integration tests for FastAPI,
  manual click-through for Next.js.
- **Type consistency checked:** `Person` response shape `{id, name,
  thumbnailUrl}` is produced by the single shared `person_response()`
  (Python, `api/serializers.py`), used by both `enroll.py` and `people.py` —
  no duplicated dict-literal across modules — and matches the `Person`
  TypeScript type field-for-field (`thumbnailUrl`, not `thumbnail_url`).
  `find_best_match`
  signature matches its usage in `enroll.py`. `detect_and_embed` return tuple
  order `(vector_bytes, crop, sharpness)` matches every call site.
- **No placeholders:** all code blocks are complete, runnable implementations
  verified against the real `ArcFaceONNXRecognizer`, `FaceProcessor`, and YOLO
  detector APIs (confirmed by hand-running the detect→align→embed pipeline
  against the committed test fixtures before writing this plan).
