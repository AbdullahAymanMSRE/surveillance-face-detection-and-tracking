# Design: Annotated clips (face box + label)

**Date:** 2026-06-23
**Status:** Approved (pre-implementation)
**Branch:** feature/qdrant-gallery-and-visit-clips (extends the per-visit clip feature)

## Goal

Draw a rectangle around the subject's face — tracking it across the clip — and
burn the person's display name onto each recorded per-visit clip. Builds directly
on the existing clip feature (`pipeline/clip.py` `ClipRecorder`, the worker wiring
in `pipeline_node.py`, and the `POST /sightings/{id}/clip` upload path).

## Decisions (user-approved)

- **Label text:** the display name — the operator label if set, else `person_NNN`
  — captured when the visit opens.
- **Box scope:** only the clip's subject (the tracked face whose visit it is);
  other faces in frame are left unmarked.
- **Style:** green rectangle, white label text on a filled green bar; always on
  (no toggle).

## Components

### 1. API — return the display name on visit open

`api/routers/sightings.py` `open_sighting` already returns
`{"personId", "sightingId", "score", "isNew"}`. Add **`displayName`** using the
existing `serializers.display_name(person)` helper (label or `person_NNN`). This
is additive; no existing caller breaks. (`person_response` already exposes
`displayName`; this just surfaces it on the open response so the worker can label
the box.)

### 2. Worker — supply box + name per frame

`pipeline_node.py`:

- `_post_open(ev)` currently returns only `sightingId`. Change it to also surface
  the `displayName` from the response so the caller can store it. Keep the open
  map keyed by `track_id`; add a parallel `clip_names: Dict[int, str]`.
- When a new visit opens, store `clip_names[ev.track_id] = displayName`.
- Each loop iteration the worker already computes `boxes, track_ids = core.step(frame)`.
  When feeding the subject's recorder, locate that track's current box —
  `boxes[i]` where `track_ids[i] == tid`, else `None` if the subject wasn't
  detected this frame — and pass it with the stored name:
  `recorder.maybe_add(frame, now, box=box, label=clip_names.get(tid))`.
- On visit close, drop `clip_names[tid]` alongside the recorder (mirror the
  existing `recorders.pop`).

Boxes from `core.step` are in full source-frame coordinates.

### 3. ClipRecorder — draw at output resolution

`pipeline/clip.py` `ClipRecorder.maybe_add` gains two optional params:
`maybe_add(self, frame, now, box=None, label=None)`.

- The existing cadence/cap gate and downscale are unchanged.
- After downscaling, compute the scale ratio (`self.width / original_width`) and,
  when `box` is given, scale the box to output coordinates and draw:
  - `cv2.rectangle` in green.
  - The `label` text (when given) in white on a filled green bar positioned just
    above the box, clamped so it stays within the frame (if the box top is near
    the top edge, place the bar just below the top instead).
- Drawing at the **downscaled output resolution** keeps text legible (vs. drawing
  on the full frame and shrinking it).
- `box is None` (or `label is None`) degrades gracefully: buffer the frame with
  whatever is available, no error.
- `encode()` is unchanged.

The callers that already invoke `maybe_add(frame, now)` (none outside the worker,
plus the unit tests) keep working because `box`/`label` default to `None`.

## Data flow

```
core.step(frame) -> boxes, track_ids   (worker has them already)
                         |
   subject tid ----------+--> box for tid (or None)
   clip_names[tid] ----------> label
                         v
   recorder.maybe_add(frame, now, box, label)
       -> downscale, scale box, cv2.rectangle + label bar -> buffer
       -> (visit end) encode() -> MP4 -> POST /sightings/{id}/clip
```

## Testing

- **ClipRecorder unit (`tests/test_clip_recorder.py`):**
  - Feeding a `box` + `label` produces an annotated frame whose pixels differ from
    the plain input where the rectangle/label are drawn (e.g. a green pixel appears
    on the box edge), proving the overlay was applied.
  - Cadence gate, frame cap, downscale, and `encode()` still hold with box/label
    present.
  - `box=None` path still buffers and encodes (no crash).
- **API (`tests/api/test_sightings.py`):** the `POST /sightings` open response
  includes `displayName` equal to `person_NNN` for a freshly created person.

## Limitation

The label is **burned into the video at record time** (captured at visit open).
Renaming the person afterward does not change already-recorded clips; only new
clips reflect the new name. This is inherent to rendering text into pixels and is
acceptable per the approved design.

## Out of scope

- Boxing non-subject faces.
- Configurable colors/styles or an on/off toggle.
- Re-rendering existing clips after a rename.
- Annotating the live preview stream (clips only).
