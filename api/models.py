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
