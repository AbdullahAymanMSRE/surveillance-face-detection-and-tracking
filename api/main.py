from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os

from . import db
from .gallery import get_gallery
from .reaper import start_reaper
from .routers import cameras, people, sightings
from .supervisor import get_supervisor

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
    # Connect to Qdrant and ensure the 'faces' collection exists. Qdrant is the
    # authoritative embedding store, so there is nothing to rebuild from SQLite.
    get_gallery()
    # Reaper closes orphaned open sightings; disabled under tests.
    if os.environ.get("FACE_API_ENABLE_REAPER", "1") == "1":
        start_reaper(
            interval_secs=float(os.environ.get("FACE_API_REAPER_INTERVAL", "5")),
            timeout_secs=float(os.environ.get("FACE_API_REAPER_TIMEOUT", "15")),
        )
    # Supervisor spawns a worker per enabled camera; disabled under tests.
    if os.environ.get("FACE_API_ENABLE_SUPERVISOR", "1") == "1":
        get_supervisor().start_monitor(
            interval=float(os.environ.get("FACE_API_SUPERVISOR_INTERVAL", "5")),
        )


@app.on_event("shutdown")
def on_shutdown() -> None:
    get_supervisor().shutdown()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(people.router)
app.include_router(sightings.router)
app.include_router(cameras.router)
