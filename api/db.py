import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import inspect, text
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


def get_clips_dir() -> Path:
    return get_data_dir() / "clips"


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


def _ensure_columns(engine) -> None:
    """Add columns introduced after a database was first created.

    ``SQLModel.metadata.create_all`` only creates missing *tables*; it never
    alters an existing one. A database created before a column was added would
    therefore lack it, and any query selecting it fails. SQLite supports a cheap
    ``ALTER TABLE ... ADD COLUMN``; apply the known additions idempotently so an
    older ``app.db`` self-heals on startup."""
    inspector = inspect(engine)
    if "sighting" not in inspector.get_table_names():
        return
    sighting_cols = {c["name"] for c in inspector.get_columns("sighting")}
    if "has_clip" not in sighting_cols:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE sighting ADD COLUMN has_clip BOOLEAN NOT NULL "
                "DEFAULT 0"))


def init_db() -> None:
    from . import models  # noqa: F401  (registers tables on SQLModel.metadata)
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _ensure_columns(engine)


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
