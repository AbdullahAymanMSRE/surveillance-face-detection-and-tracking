"""Self-populating vector database for face embeddings.

Unlike a fixed gallery, this database grows at run time:

  - A new embedding is compared (cosine similarity) against every stored
    embedding.
  - If the best match is >= threshold, it is recognized as that existing
    identity (and that identity's running-mean embedding is updated).
  - If nothing matches (including the very first face ever seen), the
    embedding is saved as a brand-new identity.

State persists to a directory:
    <db_dir>/
        embeddings.pt      # {"labels", "embeds", "counts"}
        metadata.json      # per-identity: first/last seen, sighting count
        thumbnails/        # one reference crop per identity (optional)
"""

import itertools
import json
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class FaceDatabase:
    def __init__(
        self,
        threshold: float = 0.155,
        db_dir: Optional[str | Path] = None,
        embed_dim: int = 256,
        update_running_mean: bool = True,
        save_thumbnails: bool = True,
        archive_all: bool = True,
    ):
        self.threshold = threshold
        self.embed_dim = embed_dim
        self.update_running_mean = update_running_mean
        self.save_thumbnails = save_thumbnails
        self.archive_all = archive_all  # save every sighting crop, not just one ref
        self.db_dir = Path(db_dir) if db_dir else None

        self.labels: List[str] = []
        self.embeds: Optional[torch.Tensor] = None  # [N, embed_dim] cpu, L2-normalized
        self.counts: List[int] = []
        self.meta: dict = {}  # label -> {first_seen, last_seen, sightings}
        self._next_id = 1
        self._seq = itertools.count()  # thread-safe unique counter for sighting files
        self._lock = threading.Lock()

        if self.db_dir and (self.db_dir / "embeddings.pt").exists():
            self.load()

    # ── core: match against stored, else enroll as new ───────────────────────

    def match_or_add(
        self, embedding: torch.Tensor, crop_bgr: Optional[np.ndarray] = None
    ) -> Tuple[str, float, bool]:
        """Return (label, score, is_new) for one L2-normalized embedding."""
        emb = F.normalize(embedding.detach().float().cpu(), dim=-1)
        has_crop = crop_bgr is not None and self.db_dir is not None
        sharp = (self._sharpness(crop_bgr)
                 if has_crop and self.save_thumbnails else None)
        with self._lock:
            label, score = self._match_locked(emb)
            is_new = label is None
            if is_new:
                label = self._add_locked(emb)
            else:
                self._touch_locked(label, emb)
            # Keep the sharpest crop ever seen (across runs) as the thumbnail.
            write_thumb = False
            if sharp is not None and sharp > self.meta[label].get("best_sharpness", -1.0):
                self.meta[label]["best_sharpness"] = sharp
                write_thumb = True
        # I/O kept outside the lock.
        if has_crop and self.archive_all:
            self._archive(label, crop_bgr)
        if write_thumb:
            self._write_thumbnail(label, crop_bgr)
        return label, score, is_new

    @staticmethod
    def _sharpness(crop_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _write_thumbnail(self, label: str, crop_bgr: np.ndarray) -> None:
        thumbs = self.db_dir / "thumbnails"
        thumbs.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(thumbs / f"{label}.jpg"), crop_bgr)

    def rebuild_thumbnails_from_sightings(self) -> int:
        """Set each identity's thumbnail to its sharpest archived sighting crop.

        One-off backfill so thumbnails reflect the best frame across *all* past
        runs, not just frames seen since this feature was added.
        """
        sightings_root = self.db_dir / "sightings"
        if not sightings_root.exists():
            return 0
        updated = 0
        for folder in sorted(p for p in sightings_root.iterdir() if p.is_dir()):
            label = folder.name
            best_img, best_sharp = None, -1.0
            for f in folder.glob("*.jpg"):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                s = self._sharpness(img)
                if s > best_sharp:
                    best_img, best_sharp = img, s
            if best_img is not None:
                self._write_thumbnail(label, best_img)
                self.meta.setdefault(label, {})["best_sharpness"] = best_sharp
                updated += 1
        return updated

    def _archive(self, label: str, crop_bgr: np.ndarray) -> None:
        """Save this sighting's crop under sightings/<label>/<datetime>_<seq>.jpg.

        A monotonic sequence guarantees unique filenames even when several crops
        land within the same OS clock tick (coarse on Windows).
        """
        folder = self.db_dir / "sightings" / label
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        seq = next(self._seq)
        cv2.imwrite(str(folder / f"{stamp}_{seq:09d}.jpg"), crop_bgr)

    def _match_locked(self, emb: torch.Tensor) -> Tuple[Optional[str], float]:
        if self.embeds is None or len(self.labels) == 0:
            return None, 0.0
        sims = self.embeds @ emb  # [N]
        best_score, best_idx = sims.max(dim=0)
        score = float(best_score)
        if score >= self.threshold:
            return self.labels[int(best_idx)], score
        return None, score

    def _add_locked(self, emb: torch.Tensor) -> str:
        label = f"person_{self._next_id:03d}"
        self._next_id += 1
        self.labels.append(label)
        self.counts.append(1)
        row = emb.unsqueeze(0)
        self.embeds = row if self.embeds is None else torch.cat([self.embeds, row])
        now = time.time()
        self.meta[label] = {"first_seen": now, "last_seen": now, "sightings": 1}
        return label

    def _touch_locked(self, label: str, emb: torch.Tensor) -> None:
        idx = self.labels.index(label)
        self.counts[idx] += 1
        if self.update_running_mean:
            n = self.counts[idx]
            blended = self.embeds[idx] * (n - 1) + emb
            self.embeds[idx] = F.normalize(blended, dim=0)
        m = self.meta.setdefault(
            label, {"first_seen": time.time(), "sightings": 0}
        )
        m["last_seen"] = time.time()
        m["sightings"] = m.get("sightings", 0) + 1

    # ── seed from a known gallery (optional) ─────────────────────────────────

    def seed_from_gallery(self, names: List[str], embeds: torch.Tensor) -> None:
        """Pre-populate with named identities (e.g. from enroll.py's gallery.pt)."""
        with self._lock:
            for name, emb in zip(names, embeds):
                row = F.normalize(emb.float().cpu(), dim=-1).unsqueeze(0)
                self.labels.append(name)
                self.counts.append(1)
                self.embeds = row if self.embeds is None else torch.cat([self.embeds, row])
                self.meta.setdefault(
                    name, {"first_seen": time.time(), "last_seen": time.time(),
                           "sightings": 1}
                )

    # ── persistence ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, db_dir: Optional[str | Path] = None) -> None:
        target = Path(db_dir) if db_dir else self.db_dir
        if target is None:
            return
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            torch.save(
                {"labels": self.labels,
                 "embeds": self.embeds.cpu() if self.embeds is not None else None,
                 "counts": self.counts,
                 "next_id": self._next_id},
                str(target / "embeddings.pt"),
            )
            (target / "metadata.json").write_text(json.dumps(self.meta, indent=2))

    def load(self, db_dir: Optional[str | Path] = None) -> int:
        target = Path(db_dir) if db_dir else self.db_dir
        data = torch.load(str(target / "embeddings.pt"), map_location="cpu",
                          weights_only=False)
        self.labels = list(data["labels"])
        self.embeds = data["embeds"]
        self.counts = list(data["counts"])
        self._next_id = data.get("next_id", len(self.labels) + 1)
        meta_path = target / "metadata.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return len(self.labels)
