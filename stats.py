#!/usr/bin/env python3
"""Print a summary of the face database: visits, sightings, first/last seen.

Reads the vector DB's metadata (no model needed) and prints one row per identity.

Run:
    python stats.py                 # default: arcface DB
    python stats.py --recognizer dino
    python stats.py --db-dir path/to/face_db/arcface
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_ROOT = _REPO_ROOT / "face_recognition" / "face_db"


def _fmt_time(ts) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    p = argparse.ArgumentParser(description="Show face database stats")
    p.add_argument("--recognizer", default="arcface",
                   help="Which backend's DB to read (arcface/dino)")
    p.add_argument("--db-dir", default=None,
                   help="DB directory (overrides --recognizer)")
    p.add_argument("--sort", choices=["visits", "sightings", "recent", "name"],
                   default="visits", help="Sort order")
    args = p.parse_args()

    db_dir = Path(args.db_dir) if args.db_dir else DEFAULT_DB_ROOT / args.recognizer
    meta_path = db_dir / "metadata.json"
    if not meta_path.exists():
        print(f"No database found at {db_dir} (expected metadata.json).")
        return 1

    meta = json.loads(meta_path.read_text())
    if not meta:
        print(f"Database at {db_dir} is empty.")
        return 0

    rows = []
    for label, m in meta.items():
        # Count actual archived crop files as a cross-check on 'sightings'.
        crop_dir = db_dir / "sightings" / label
        archived = len(list(crop_dir.glob("*.jpg"))) if crop_dir.exists() else 0
        rows.append({
            "label": label,
            "visits": m.get("appearances", 0),
            "sightings": m.get("sightings", 0),
            "archived": archived,
            "first": m.get("first_seen"),
            "last": m.get("last_seen"),
        })

    keymap = {"visits": lambda r: -r["visits"],
              "sightings": lambda r: -r["sightings"],
              "recent": lambda r: -(r["last"] or 0),
              "name": lambda r: r["label"]}
    rows.sort(key=keymap[args.sort])

    print(f"\nFace database: {db_dir}")
    print(f"{'identity':<14}{'visits':>7}{'sightings':>11}{'crops':>7}"
          f"  {'first seen':<17}{'last seen':<17}")
    print("-" * 73)
    for r in rows:
        print(f"{r['label']:<14}{r['visits']:>7}{r['sightings']:>11}{r['archived']:>7}"
              f"  {_fmt_time(r['first']):<17}{_fmt_time(r['last']):<17}")
    print("-" * 73)
    print(f"{len(rows)} identities | "
          f"{sum(r['visits'] for r in rows)} total visits | "
          f"{sum(r['sightings'] for r in rows)} total sightings\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
