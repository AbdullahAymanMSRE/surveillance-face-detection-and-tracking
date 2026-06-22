from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from api import db, gallery, supervisor
from api.main import app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_API_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FACE_API_ENABLE_REAPER", "0")
    monkeypatch.setenv("FACE_API_ENABLE_SUPERVISOR", "0")
    db.reset_engine()
    gallery.reset_gallery()
    supervisor.reset_supervisor()
    with TestClient(app) as c:
        yield c
    db.reset_engine()
    gallery.reset_gallery()
    supervisor.reset_supervisor()


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
