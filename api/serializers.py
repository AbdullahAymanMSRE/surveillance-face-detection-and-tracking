from .models import Person


def person_response(person: Person) -> dict:
    return {
        "id": person.id,
        "name": person.name,
        "thumbnailUrl": f"/people/{person.id}/thumbnail",
    }
