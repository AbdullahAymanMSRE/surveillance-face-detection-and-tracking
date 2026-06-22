from .models import Person


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
