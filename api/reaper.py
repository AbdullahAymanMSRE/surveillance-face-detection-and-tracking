"""Background reaper: closes open sightings whose heartbeat went stale.

A worker normally closes a visit with ``POST /sightings/{id}/end``. If a worker
crashes or its stream dies mid-visit, that call never arrives and the sighting
would hang open forever (showing the person as "currently visible"). The reaper
periodically closes any open sighting whose ``last_seen`` is older than a
timeout, so liveness self-heals.
"""

import threading
import time
from datetime import datetime, timedelta

from sqlmodel import Session, select

from .consolidate import consolidate_identities
from .db import get_engine
from .models import Sighting


def close_stale_sightings(session: Session, timeout_secs: float) -> int:
    """Close every open sighting not heartbeated within ``timeout_secs``."""
    cutoff = datetime.utcnow() - timedelta(seconds=timeout_secs)
    stale = session.exec(
        select(Sighting).where(
            Sighting.ended_at == None,  # noqa: E711  (SQL IS NULL)
            Sighting.last_seen < cutoff,
        )
    ).all()
    now = datetime.utcnow()
    for sighting in stale:
        sighting.ended_at = now
        session.add(sighting)
    if stale:
        session.commit()
    return len(stale)


def start_reaper(interval_secs: float = 5.0, timeout_secs: float = 15.0,
                 consolidate_every: int = 4) -> threading.Thread:
    """Start the reaper loop on a daemon thread and return it.

    Every ``consolidate_every`` ticks it also runs identity consolidation
    (merging cold-start splits of the same face); set to 0 to disable.
    """
    def loop() -> None:
        tick = 0
        while True:
            time.sleep(interval_secs)
            tick += 1
            try:
                with Session(get_engine()) as session:
                    close_stale_sightings(session, timeout_secs)
                    if consolidate_every and tick % consolidate_every == 0:
                        consolidate_identities(session)
            except Exception:
                # The engine may be mid-reset (tests) or the DB briefly absent;
                # the next tick recovers.
                pass

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
