from typing import Optional

from .models import Camera, Person, Sighting


def display_name(person: Person) -> str:
    """Anonymous display name; the operator-provided label takes precedence."""
    return person.label or f"person_{person.id:03d}"


def person_response(person: Person) -> dict:
    return {
        "id": person.id,
        "label": person.label,
        "displayName": display_name(person),
        "thumbnailUrl": f"/people/{person.id}/thumbnail",
    }


def _iso(value) -> Optional[str]:
    return value.isoformat() if value is not None else None


def sighting_response(sighting: Sighting, camera: Optional[Camera]) -> dict:
    return {
        "id": sighting.id,
        "cameraId": sighting.camera_id,
        "cameraName": camera.name if camera else None,
        "location": camera.location if camera else None,
        "startedAt": _iso(sighting.started_at),
        "endedAt": _iso(sighting.ended_at),
        "lastSeen": _iso(sighting.last_seen),
        "active": sighting.ended_at is None,
    }
