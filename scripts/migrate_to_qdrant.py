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
