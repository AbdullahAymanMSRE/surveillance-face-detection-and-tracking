"""Identity consolidation: merge anonymous persons that are the same face.

The pipeline can split one person into several identities during the first
seconds it sees them — a profile or distant view that appears on a fresh track
before that person has accumulated matching pose exemplars enrolls as a new
person. Once coverage builds up those stranded duplicates are recognizably the
same face, but nothing retroactively joins them.

This pass closes that gap: it compares every pair of persons by their best
cross-exemplar cosine similarity and merges any pair above ``merge_threshold``
(transitively, so A~B~C collapse into one). The surviving person keeps the most
visits and the sharpest thumbnail; the others' sightings are reassigned to it,
their gallery embeddings are repointed to it in Qdrant, and the emptied persons
are deleted.
"""

from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from .db import get_thumbnails_dir
from .gallery import get_gallery
from .models import Person, Sighting

# Cosine at/above which two identities are judged the same face. Sits in the
# empirical gap between different people (cross-identity ArcFace cosine ~0.05-0.1)
# and the same person across pose (frontal-vs-profile ~0.40), above the live
# match threshold (0.28) so merging stays more conservative than live matching.
MERGE_THRESHOLD = 0.35


def _max_cross_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b.T).max())


def consolidate_identities(session: Session,
                           merge_threshold: float = MERGE_THRESHOLD) -> int:
    """Merge same-face persons. Returns how many persons were removed."""
    gallery = get_gallery()
    mats = gallery.all_vectors_by_person()
    pids = sorted(mats)
    if len(pids) < 2:
        return 0

    # Union-find over persons linked by a high cross-exemplar similarity.
    parent = {p: p for p in pids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pids[i], pids[j]
            if _max_cross_similarity(mats[a], mats[b]) >= merge_threshold:
                union(a, b)

    groups: dict = {}
    for p in pids:
        groups.setdefault(find(p), []).append(p)

    # Visit counts pick the survivor (the most-seen identity wins).
    visits: dict = {}
    for s in session.exec(select(Sighting)).all():
        visits[s.person_id] = visits.get(s.person_id, 0) + 1
    persons = {p.id: p for p in session.exec(select(Person)).all()}

    thumbs = get_thumbnails_dir()
    removed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = max(members, key=lambda p: (visits.get(p, 0), -p))
        keep_person = persons[keeper]
        for loser in (m for m in members if m != keeper):
            for s in session.exec(
                    select(Sighting).where(Sighting.person_id == loser)).all():
                s.person_id = keeper
                session.add(s)
            gallery.reassign_person(loser, keeper)
            lose_person = persons[loser]
            # Promote the sharper thumbnail if the loser had a better shot.
            if lose_person.best_sharpness > keep_person.best_sharpness:
                keep_person.best_sharpness = lose_person.best_sharpness
                src, dst = thumbs / f"{loser}.jpg", thumbs / f"{keeper}.jpg"
                if src.exists():
                    dst.write_bytes(src.read_bytes())
                session.add(keep_person)
            stale = thumbs / f"{loser}.jpg"
            if stale.exists():
                stale.unlink()
            session.delete(lose_person)
            removed += 1

    if removed:
        session.commit()
    return removed
