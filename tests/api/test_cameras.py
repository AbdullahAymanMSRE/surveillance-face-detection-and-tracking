def _create(client, **over):
    body = {"name": "Front Gate", "location": "Gate", "source": "0"}
    body.update(over)
    return client.post("/cameras", json=body)


def test_create_camera(client):
    resp = _create(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] is not None
    assert body["name"] == "Front Gate"
    assert body["status"] == "stopped"  # supervisor off under tests


def test_list_and_get_camera(client):
    cam_id = _create(client, name="A").json()["id"]
    _create(client, name="B")
    listed = client.get("/cameras").json()
    assert {c["name"] for c in listed} == {"A", "B"}

    one = client.get(f"/cameras/{cam_id}")
    assert one.status_code == 200
    assert one.json()["name"] == "A"


def test_get_missing_camera_404(client):
    assert client.get("/cameras/999").status_code == 404


def test_update_camera(client):
    cam_id = _create(client).json()["id"]
    resp = client.patch(f"/cameras/{cam_id}", json={"location": "Lobby"})
    assert resp.status_code == 200
    assert resp.json()["location"] == "Lobby"


def test_start_stop_toggle_enabled(client):
    cam_id = _create(client, enabled=False).json()["id"]
    started = client.post(f"/cameras/{cam_id}/start")
    assert started.status_code == 200
    assert started.json()["enabled"] is True

    stopped = client.post(f"/cameras/{cam_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["enabled"] is False


def test_delete_camera(client):
    cam_id = _create(client).json()["id"]
    assert client.delete(f"/cameras/{cam_id}").status_code == 204
    assert client.get(f"/cameras/{cam_id}").status_code == 404
