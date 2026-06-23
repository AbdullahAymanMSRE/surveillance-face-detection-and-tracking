# Annotated Clips (face box + label) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw a green rectangle around the subject's face (tracking it across the clip) with their display name burned into each recorded per-visit MP4.

**Architecture:** The API returns the person's `displayName` when a visit opens; the worker passes the subject's per-frame box plus that name to `ClipRecorder`, which draws the rectangle + label bar at output resolution before buffering each frame. Builds directly on the existing clip feature.

**Tech Stack:** FastAPI/SQLModel, OpenCV (`cv2.rectangle`, `cv2.putText`, `cv2.getTextSize`), the headless worker `pipeline_node.py`.

## Global Constraints

- Python deps via the existing `.venv` (python3.12). Run python as `.venv/bin/python`, pytest as `.venv/bin/python -m pytest`. Do NOT run `make install`.
- `numpy>=1.26,<2`; OpenCV is `cv2` (opencv-python).
- Box coordinates from `core.step` are `(x1, y1, x2, y2)` ints in **source-frame** coordinates; the recorder downscales to `width` and must scale the box by the same ratio.
- Style: rectangle color green `(0, 255, 0)`, label text white `(255, 255, 255)` on a filled green bar; always on (no toggle); **subject only**.
- Drawing must NOT mutate the worker's shared frame (it is also handed to the preview and may be the same object across recorders when no downscale happens) — annotate on a copy.
- Tests run with no external services.
- Label is captured at visit-open (burned in at record time); later renames do not change existing clips — this is expected, not a bug.

---

## Task 1: API returns displayName on visit open

**Files:**
- Modify: `api/routers/sightings.py`
- Test: `tests/api/test_sightings.py`

**Interfaces:**
- Produces: `POST /sightings` (open) response gains `"displayName": str` — the operator label if set, else `person_NNN`.

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_sightings.py` (the file already has `_make_camera()`, `_emb(idx)`, `_open(client, camera_id, emb)` helpers; `_open` returns the response):

```python
def test_open_sighting_returns_display_name(client):
    cam_id = _make_camera()
    body = _open(client, cam_id, _emb(1)).json()
    # A freshly created anonymous person has no label -> person_NNN.
    assert body["displayName"] == f"person_{body['personId']:03d}"
```

- [ ] **Step 2: Run it to verify failure**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py::test_open_sighting_returns_display_name -q`
Expected: FAIL with `KeyError: 'displayName'`.

- [ ] **Step 3: Implement**

In `api/routers/sightings.py`, add `display_name` to the serializers import:

```python
from ..serializers import display_name, person_response, sighting_response
```

In `open_sighting`, change the return statement to include the display name (the `person` variable is in scope after the `match_create_lock` block, assigned in both the new-person and matched branches):

```python
    return {"personId": person_id, "sightingId": sighting_id,
            "score": score, "isNew": is_new,
            "displayName": display_name(person)}
```

- [ ] **Step 4: Run the test + the sightings suite**

Run: `.venv/bin/python -m pytest tests/api/test_sightings.py -q`
Expected: PASS (the new test plus all existing sightings tests).

- [ ] **Step 5: Commit**

```bash
git add api/routers/sightings.py tests/api/test_sightings.py
git commit -m "feat(sightings): return displayName on visit open"
```

---

## Task 2: ClipRecorder draws the box + label

**Files:**
- Modify: `pipeline/clip.py`
- Test: `tests/test_clip_recorder.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ClipRecorder.maybe_add(self, frame, now, box=None, label=None)` — `box` is `(x1,y1,x2,y2)` in source-frame coords, `label` is an optional str. When `box` is given it is scaled to the downscaled output and drawn (green rectangle + white-on-green label bar) on a copy; `box=None` buffers the plain frame as before. Existing 2-arg callers keep working.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_clip_recorder.py` (it already imports `ClipRecorder` and `numpy as np`):

```python
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
```

- [ ] **Step 2: Run them to verify failure**

Run: `.venv/bin/python -m pytest tests/test_clip_recorder.py -q`
Expected: FAIL — `maybe_add()` got an unexpected keyword argument `box`.

- [ ] **Step 3: Implement the drawing**

In `pipeline/clip.py`, replace `maybe_add` and add a `_draw` helper (keep `__init__`, `frame_count`, `_downscale`, and `encode` unchanged):

```python
    def maybe_add(self, frame: np.ndarray, now: float,
                  box=None, label=None) -> None:
        if len(self._frames) >= self.max_frames:
            return
        if (now - self._last_ts) < (1.0 / self.fps):
            return
        self._last_ts = now
        src_w = frame.shape[1]
        out = self._downscale(frame)
        if box is not None:
            if out is frame:          # no resize happened -> don't mutate caller's frame
                out = out.copy()
            self._draw(out, src_w, box, label)
        self._frames.append(out)

    def _draw(self, img: np.ndarray, src_width: int, box, label) -> None:
        scale = img.shape[1] / src_width
        x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not label:
            return
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bar_h = th + base + 6
        by1, by2 = y1 - bar_h, y1        # bar sits just above the box...
        if by1 < 0:                      # ...unless the box hugs the top edge
            by1, by2 = y1, y1 + bar_h
        cv2.rectangle(img, (x1, by1), (x1 + tw + 8, by2), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 4, by2 - base - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
```

- [ ] **Step 4: Run the recorder tests**

Run: `.venv/bin/python -m pytest tests/test_clip_recorder.py -q`
Expected: PASS — the 3 new tests plus all existing recorder tests (the existing `maybe_add(frame, now)` calls still work since `box`/`label` default to `None`).

- [ ] **Step 5: Commit**

```bash
git add pipeline/clip.py tests/test_clip_recorder.py
git commit -m "feat(clip): draw subject box + label onto recorded frames"
```

---

## Task 3: Worker supplies box + name per frame

**Files:**
- Modify: `pipeline_node.py`

**Interfaces:**
- Consumes: `POST /sightings` returns `displayName` (Task 1); `ClipRecorder.maybe_add(frame, now, box=, label=)` (Task 2).
- Produces: the worker passes the subject's current box and stored display name to each active recorder; `_post_open` returns `(sighting_id, display_name)`.

- [ ] **Step 1: Add `Tuple` to the typing import**

In `pipeline_node.py`, the import is `from typing import Dict, Optional`. Change it to:

```python
from typing import Dict, Optional, Tuple
```

- [ ] **Step 2: Add the names map next to `recorders`**

Where `recorders: Dict[int, ClipRecorder] = {}` is declared, add below it:

```python
    clip_names: Dict[int, str] = {}   # track_id -> display name for the clip overlay
```

- [ ] **Step 3: Have `_post_open` also return the display name**

Replace `_post_open`:

```python
    def _post_open(ev) -> Optional[Tuple[int, str]]:
        try:
            payload = _encode(ev)
            payload["data"]["camera_id"] = camera_id
            r = client.post(f"{api}/sightings", timeout=10, **payload)
            if r.status_code == 201:
                body = r.json()
                return body["sightingId"], body.get("displayName", "")
        except httpx.HTTPError as e:
            print(f"[node] open failed: {e}", flush=True)
        return None
```

- [ ] **Step 4: Store the name when a visit opens**

Replace the open branch in the events loop:

```python
                else:
                    opened = _post_open(ev)
                    if opened is not None:
                        sid, name = opened
                        open_sightings[ev.track_id] = sid
                        clip_names[ev.track_id] = name
                        recorders[ev.track_id] = ClipRecorder(
                            max_frames=int(CLIP_SECS * CLIP_FPS),
                            fps=CLIP_FPS, width=CLIP_WIDTH)
```

- [ ] **Step 5: Feed the subject's box + name to its recorder**

Replace the frame-feeding loop (the `for tid in track_ids: rec = recorders.get(tid)...` block) so it zips boxes with track ids and passes the box + label:

```python
            for box, tid in zip(boxes, track_ids):
                rec = recorders.get(tid)
                if rec is not None:
                    rec.maybe_add(frame, now, box=box, label=clip_names.get(tid))
```

- [ ] **Step 6: Drop the name on visit close (both paths)**

In the grace-timeout close loop, after `last_present.pop(tid, None)` add:

```python
                    clip_names.pop(tid, None)
```

In the `finally:` block, inside the `for tid, sid in list(open_sightings.items()):` loop, after `_post_end(sid)` add:

```python
            clip_names.pop(tid, None)
```

- [ ] **Step 7: Verify the worker imports and the suite is green**

Run: `.venv/bin/python -c "import pipeline_node"`
Expected: no output, exit 0.

Run: `.venv/bin/python -m pytest tests -q`
Expected: PASS (full suite; nothing regressed).

- [ ] **Step 8: Commit**

```bash
git add pipeline_node.py
git commit -m "feat(worker): pass subject box + display name to the clip recorder"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- API returns `displayName` on open → Task 1. ClipRecorder draws scaled box + white-on-green label bar at output resolution, copy-on-annotate, `box=None` graceful → Task 2. Worker captures `displayName`, stores `clip_names`, passes per-frame subject box + label, cleans up on close → Task 3.
- Tests: overlay pixels + no-mutation + `box=None` (Task 2); `displayName` field (Task 1); full-suite + import check (Task 3).
- Limitation (label burned at record time) is inherent and documented in Global Constraints.

**Placeholder scan:** No TBD/TODO; every code step shows full code. The `_open`/`_make_camera`/`_emb` helpers referenced in Task 1 already exist in `tests/api/test_sightings.py`.

**Type consistency:** `_post_open -> Optional[Tuple[int, str]]` matches the `opened = _post_open(ev); sid, name = opened` unpacking (Task 3). `maybe_add(frame, now, box=None, label=None)` signature is identical between Task 2's definition and Task 3's call. `clip_names: Dict[int, str]` is declared (Step 2), written (Step 4), read (Step 5), and popped (Step 6) consistently. `displayName` key is consistent between Task 1's response and Task 3's `body.get("displayName", "")`.
