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
