"""Reusable face recognition wrapper around the trained DINO+ArcFace model.

Loads a DinoModel checkpoint and exposes:
  - embed(crop_bgr)        -> 256-d L2-normalized embedding for one OpenCV crop
  - embed_batch(crops_bgr) -> embeddings for a list of crops
  - load_gallery(dir)      -> build a {name -> mean embedding} gallery from images
  - identify(embedding)    -> (name, cosine_score) against the loaded gallery

Preprocessing matches the validation transform in dino_vggface2.ImageValData
(resize 224, scale to [0,1], ImageNet normalize) so embeddings are consistent
with how the model was evaluated.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

UNKNOWN_LABEL = "Unknown"


class DinoModel(nn.Module):
    """Pretrained DINO ViT backbone + 3-layer MLP projection head.

    Architecture mirrors dino_vggface2.DinoModel so the trained checkpoint loads
    here, but without importing the training module (which pulls in
    pytorch_lightning / deepspeed). Inference needs only torch + torchvision.
    """

    def __init__(self, backbone_name: str, backbone_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dino:main", backbone_name)
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection_head(self.backbone(x)), dim=-1)


class FaceRecognizer:
    def __init__(
        self,
        run_dir: str | Path,
        checkpoint: str = "arcface_finetuned.pt",
        threshold: float = 0.155,
        device: Optional[str] = None,
    ):
        self.run_dir = Path(run_dir)
        self.threshold = threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        summary = json.loads((self.run_dir / "run_summary.json").read_text())
        backbone = summary["backbone"]
        backbone_dim = summary["backbone_dim"]
        self.embed_dim = summary["embed_dim"]

        self.model = DinoModel(backbone, backbone_dim, self.embed_dim)
        # weights_only=False: this is our own trusted checkpoint (torch>=2.6 default flipped).
        state_dict = torch.load(self.run_dir / checkpoint, map_location="cpu",
                                weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        # Same as ImageValData.transform, applied to a uint8 CHW tensor.
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Gallery: parallel lists kept in sync.
        self.gallery_names: List[str] = []
        self.gallery_embeds: Optional[torch.Tensor] = None  # [N, embed_dim] on device

    # ── preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """OpenCV BGR HWC uint8 crop -> normalized CHW float tensor [3, 224, 224]."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # CHW uint8
        return self._transform(tensor)

    # ── embedding ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def embed_batch(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        """Return L2-normalized embeddings [B, embed_dim] on the model device."""
        if not crops_bgr:
            return torch.empty(0, self.embed_dim, device=self.device)
        batch = torch.stack([self._preprocess(c) for c in crops_bgr]).to(self.device)
        # fp16 autocast on CUDA only. On CPU it stays fp32: bf16/fp16 autocast is
        # emulated on CPUs without native support and is dramatically slower.
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = self.model(batch)
        else:
            emb = self.model(batch)
        return emb.float()

    def embed(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Return a single L2-normalized embedding [embed_dim] on the model device."""
        return self.embed_batch([crop_bgr])[0]

    # ── gallery ──────────────────────────────────────────────────────────────

    def load_gallery(self, gallery_dir: str | Path, batch_size: int = 32) -> int:
        """Build a one-embedding-per-person gallery from gallery_dir/<name>/*.jpg.

        Each person's embedding is the L2-normalized mean of their image
        embeddings. Returns the number of identities enrolled.
        """
        gallery_dir = Path(gallery_dir)
        if not gallery_dir.exists():
            raise FileNotFoundError(f"Gallery dir not found: {gallery_dir}")

        names: List[str] = []
        embeds: List[torch.Tensor] = []
        people = sorted(d for d in gallery_dir.iterdir() if d.is_dir())
        for person in people:
            imgs = sorted(
                p for p in person.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            crops = [cv2.imread(str(p)) for p in imgs]
            crops = [c for c in crops if c is not None]
            if not crops:
                continue

            person_embeds = []
            for i in range(0, len(crops), batch_size):
                person_embeds.append(self.embed_batch(crops[i:i + batch_size]))
            mean = torch.cat(person_embeds).mean(dim=0)
            mean = torch.nn.functional.normalize(mean, dim=-1)
            names.append(person.name)
            embeds.append(mean)

        self.gallery_names = names
        self.gallery_embeds = torch.stack(embeds).to(self.device) if embeds else None
        return len(names)

    def add_identity(self, name: str, embedding: torch.Tensor) -> None:
        """Append/replace a single identity's embedding (already on any device)."""
        embedding = torch.nn.functional.normalize(
            embedding.detach().float().to(self.device), dim=-1
        ).unsqueeze(0)
        if name in self.gallery_names:
            self.gallery_embeds[self.gallery_names.index(name)] = embedding[0]
            return
        self.gallery_names.append(name)
        self.gallery_embeds = (
            embedding if self.gallery_embeds is None
            else torch.cat([self.gallery_embeds, embedding])
        )

    def save_gallery(self, path: str | Path) -> None:
        if self.gallery_embeds is None:
            raise ValueError("Gallery is empty; nothing to save.")
        torch.save(
            {"names": self.gallery_names, "embeds": self.gallery_embeds.cpu()},
            str(path),
        )

    def load_gallery_cache(self, path: str | Path) -> int:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        self.gallery_names = list(data["names"])
        self.gallery_embeds = data["embeds"].to(self.device)
        return len(self.gallery_names)

    # ── identification ───────────────────────────────────────────────────────

    def identify(self, embedding: torch.Tensor) -> Tuple[str, float]:
        """Nearest gallery identity by cosine similarity (single embedding)."""
        names, scores = self.identify_batch(embedding.unsqueeze(0))
        return names[0], scores[0]

    def identify_batch(
        self, embeddings: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """Vectorized identification for a batch of embeddings [B, embed_dim]."""
        if self.gallery_embeds is None or embeddings.numel() == 0:
            n = embeddings.shape[0] if embeddings.ndim == 2 else 0
            return [UNKNOWN_LABEL] * n, [0.0] * n

        sims = embeddings.to(self.device) @ self.gallery_embeds.T  # [B, N]
        best_scores, best_idx = sims.max(dim=1)
        names, scores = [], []
        for score, idx in zip(best_scores.tolist(), best_idx.tolist()):
            scores.append(float(score))
            names.append(self.gallery_names[idx] if score >= self.threshold
                         else UNKNOWN_LABEL)
        return names, scores
