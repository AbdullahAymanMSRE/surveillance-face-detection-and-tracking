"""Schema self-heal: an app.db created before a column was added must gain it.

`SQLModel.metadata.create_all` only creates missing tables; it never alters an
existing one. `init_db` therefore also runs `_ensure_columns`, which adds known
later columns (currently `sighting.has_clip`) with a cheap SQLite ADD COLUMN so
an older database keeps working instead of 500ing on every sighting query.
"""

from sqlalchemy import inspect, text
from sqlmodel import create_engine

from api.db import _ensure_columns


def _legacy_sighting_columns(engine) -> set:
    return {c["name"] for c in inspect(engine).get_columns("sighting")}


def test_ensure_columns_adds_missing_has_clip(tmp_path):
    db = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{db}")
    # Simulate a pre-`has_clip` schema: a sighting table without the column.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE sighting (id INTEGER PRIMARY KEY, person_id INTEGER, "
            "camera_id INTEGER, started_at TEXT, last_seen TEXT, ended_at TEXT, "
            "best_sharpness REAL)"))
        conn.execute(text("INSERT INTO sighting (id, person_id) VALUES (1, 7)"))

    assert "has_clip" not in _legacy_sighting_columns(engine)

    _ensure_columns(engine)

    assert "has_clip" in _legacy_sighting_columns(engine)
    # Existing rows default to not-having-a-clip, and the column is queryable.
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT has_clip FROM sighting")).all()
    assert rows == [(0,)]


def test_ensure_columns_is_idempotent(tmp_path):
    db = tmp_path / "current.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE sighting (id INTEGER PRIMARY KEY, has_clip BOOLEAN "
            "NOT NULL DEFAULT 0)"))
    # Already has the column -> second call must not raise (no duplicate ADD).
    _ensure_columns(engine)
    _ensure_columns(engine)
    assert "has_clip" in _legacy_sighting_columns(engine)


def test_ensure_columns_noop_without_sighting_table(tmp_path):
    db = tmp_path / "empty.db"
    engine = create_engine(f"sqlite:///{db}")
    # No sighting table at all -> must return cleanly, not raise.
    _ensure_columns(engine)
    assert "sighting" not in inspect(engine).get_table_names()
