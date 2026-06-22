from api.supervisor import Supervisor


def test_inactive_supervisor_never_spawns():
    """Under tests the supervisor is inactive; control calls must be no-ops so
    no real worker subprocess is ever launched."""
    s = Supervisor()
    assert s.active is False
    s.start_camera(1)
    assert s.is_running(1) is False
    assert s.preview_port(1) is None
    s.stop_camera(1)  # safe no-op
    s.reconcile()     # safe no-op while inactive
