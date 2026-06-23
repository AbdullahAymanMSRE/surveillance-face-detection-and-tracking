import os

import numpy as np

from pipeline.clip import ClipRecorder


def _frame(w=1280, h=720):
    return (np.ones((h, w, 3), dtype=np.uint8) * 127)


def test_respects_fps_cadence():
    r = ClipRecorder(max_frames=50, fps=10, width=640)
    r.maybe_add(_frame(), now=100.0)
    r.maybe_add(_frame(), now=100.02)  # too soon (<0.1s) -> dropped
    r.maybe_add(_frame(), now=100.2)   # ok
    assert r.frame_count == 2


def test_caps_frame_count():
    r = ClipRecorder(max_frames=3, fps=10, width=640)
    t = 0.0
    for _ in range(20):
        r.maybe_add(_frame(), now=t)
        t += 0.2
    assert r.frame_count == 3


def test_downscales_to_width():
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    r.maybe_add(_frame(1280, 720), now=0.0)
    assert r._frames[0].shape[1] == 640  # width
    assert r._frames[0].shape[0] == 360  # height preserves aspect


def test_encode_writes_a_playable_file(tmp_path):
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    t = 0.0
    for _ in range(5):
        r.maybe_add(_frame(), now=t)
        t += 0.2
    out = str(tmp_path / "clip.mp4")
    assert r.encode(out) is True
    assert os.path.getsize(out) > 0


def test_encode_empty_returns_false(tmp_path):
    r = ClipRecorder()
    assert r.encode(str(tmp_path / "x.mp4")) is False


def test_box_label_draws_green_overlay():
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)  # plain gray, will downscale
    r.maybe_add(frame, now=0.0, box=(100, 100, 300, 400), label="person_001")
    out = r._frames[0]
    assert out.shape[1] == 640                 # downscaled
    assert not np.all(out == 127)              # overlay changed pixels
    assert bool((out[:, :, 1] == 255).any())   # a pure-green pixel exists (the box)


def test_box_does_not_mutate_input_frame():
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    # width == 640 so _downscale returns the same array; annotation must copy.
    frame = np.full((360, 640, 3), 127, dtype=np.uint8)
    r.maybe_add(frame, now=0.0, box=(10, 10, 100, 100), label="x")
    assert np.all(frame == 127)                # caller's frame untouched


def test_none_box_still_buffers_plain_frame():
    r = ClipRecorder(max_frames=5, fps=10, width=640)
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    r.maybe_add(frame, now=0.0, box=None, label=None)
    assert r.frame_count == 1
    assert np.all(r._frames[0] == 127)         # nothing drawn
