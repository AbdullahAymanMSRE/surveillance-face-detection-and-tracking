#!/usr/bin/env python3
"""
VGG Face2 Face Recognition — DINO ViT-Base fine-tuning with multi-crop + ArcFace
Ahmed Elmersawy — Purdue University / davisjam lab

Paper:   https://arxiv.org/pdf/2104.14294.pdf
GitHub:  https://github.com/facebookresearch/dino
Dataset: https://www.kaggle.com/datasets/hearfool/vggface2

Training pipeline:
  Stage 1 — DINO self-supervised fine-tuning (no labels, multi-crop)
  Stage 2 — ArcFace supervised fine-tuning (identity labels, optional via --arcface)

Expected dataset layout (Kaggle download):
    $DATA_ROOT/
        train/
            n000001/
                0001_01.jpg ...
        test/
            n009131/
                ...

Run:
    python dino_vggface2.py --data /scratch/gilbreth/$USER/vggface2 --out ./checkpoints
    python dino_vggface2.py --data /scratch/gilbreth/$USER/vggface2 --out ./checkpoints --arcface
"""

import argparse
import copy
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    sys.exit("[error] pytorch_lightning not installed.  Run: pip install pytorch-lightning")

try:
    from deepspeed.ops.adam import FusedAdam as _FusedAdam
    _HAS_DEEPSPEED = True
except ImportError:
    _HAS_DEEPSPEED = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Signal handling (Slurm SIGTERM) ──────────────────────────────────────────

_TERMINATE = False

def _handle_sigterm(signum, frame):
    global _TERMINATE
    log.warning("SIGTERM received — will checkpoint after this step.")
    _TERMINATE = True

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Constants (match notebook v5) ────────────────────────────────────────────

IMAGE_SIZE       = 224
LOCAL_SIZE       = 96
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

N_GLOBAL_CROPS   = 2
N_LOCAL_CROPS    = 4
GLOBAL_SCALE     = (0.4, 1.0)
LOCAL_SCALE      = (0.05, 0.4)

TEMPERATURE_S    = 0.1
TEMPERATURE_T    = 0.05
CENTER_MOMENTUM  = 0.9
TEACHER_MOMENTUM = 0.996

# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageData(Dataset):
    """Multi-crop: N_GLOBAL_CROPS global views + N_LOCAL_CROPS local views per image."""

    def __init__(self, files: List[str]):
        self.files       = files
        self.global_crop = transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=GLOBAL_SCALE)
        self.local_crop  = transforms.RandomResizedCrop((LOCAL_SIZE, LOCAL_SIZE), scale=LOCAL_SCALE)
        self.augment     = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])
        self.normalize = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

    def __len__(self):
        return len(self.files)

    def _view(self, img: torch.Tensor, crop_fn: Callable) -> torch.Tensor:
        return self.normalize(self.augment(crop_fn(img)))

    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        global_views = [self._view(img, self.global_crop) for _ in range(N_GLOBAL_CROPS)]
        local_views  = [self._view(img, self.local_crop)  for _ in range(N_LOCAL_CROPS)]
        return tuple(global_views + local_views)


class ImageValData(Dataset):
    """Validation: single resized view + identity label (parent folder name)."""

    def __init__(self, files: List[str]):
        self.files  = files
        self.labels = [Path(f).parent.name for f in files]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return self.transform(img), self.labels[i]


class IdentityLabeledData(Dataset):
    """ArcFace stage: single crop + integer identity label."""

    def __init__(self, files: List[str], label_to_idx: dict):
        self.files  = files
        self.labels = [label_to_idx[Path(f).parent.name] for f in files]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return self.transform(img), self.labels[i]


def collate_multicrop(batch):
    views = list(zip(*batch))
    return tuple(torch.stack(v) for v in views)


def collate_val(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


# ── Model ─────────────────────────────────────────────────────────────────────

class DinoModel(nn.Module):
    """Pretrained DINO ViT backbone + 3-layer MLP projection head."""

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


# ── Loss ──────────────────────────────────────────────────────────────────────

class HLoss:
    def __init__(self, temperature_t: float, temperature_s: float):
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s

    def __call__(self, t: torch.Tensor, s: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        t    = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
        logs = F.log_softmax(s / self.temperature_s, dim=1)
        return -(t * logs).sum(dim=1).mean()


# ── Lightning module — Stage 1: DINO ─────────────────────────────────────────

class DINOLightning(pl.LightningModule):
    def __init__(self, teacher: nn.Module, lr: float, loss_fn: Callable,
                 embed_dim: int, center_momentum: float, param_momentum: float,
                 warmup_epochs: int, total_epochs: int):
        super().__init__()
        self.teacher       = teacher
        self.student       = copy.deepcopy(teacher)
        self.lr            = lr
        self.loss_fn       = loss_fn
        self.c_mom         = center_momentum
        self.p_mom         = param_momentum
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.register_buffer("center", torch.zeros(1, embed_dim))

        for p in self.teacher.parameters():
            p.requires_grad = False

    def training_step(self, batch, _batch_idx):
        if _TERMINATE:
            self.trainer.should_stop = True
            return None

        views           = batch
        student_outputs = [self.student(v) for v in views]
        teacher_outputs = [self.teacher(v) for v in views[:N_GLOBAL_CROPS]]

        loss    = 0.0
        n_terms = 0
        for t_idx, t_out in enumerate(teacher_outputs):
            for s_idx, s_out in enumerate(student_outputs):
                if s_idx == t_idx:
                    continue
                loss    = loss + self.loss_fn(t_out, s_out, self.center)
                n_terms += 1
        loss = loss / n_terms

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        emp_center  = F.normalize(torch.cat(teacher_outputs).mean(dim=0, keepdim=True), dim=-1)
        self.center = F.normalize(self.c_mom * self.center + (1 - self.c_mom) * emp_center, dim=-1)

        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data = self.p_mom * t_p.data + (1 - self.p_mom) * s_p.data

        return loss

    def validation_step(self, batch, _batch_idx):
        images, labels = batch
        embeddings     = self.teacher(images)
        sim            = embeddings @ embeddings.T
        sim.fill_diagonal_(-1.0)
        nearest        = sim.argmax(dim=1)
        correct        = sum(labels[i] == labels[nearest[i].item()] for i in range(len(labels)))
        self.log("val_rank1_acc", torch.tensor(correct / len(labels), device=self.device),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = (
            _FusedAdam(self.student.parameters(), lr=self.lr)
            if _HAS_DEEPSPEED
            else torch.optim.Adam(self.student.parameters(), lr=self.lr)
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_epochs, eta_min=1e-6)
        sched  = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── Stage 2: ArcFace fine-tuning ─────────────────────────────────────────────

def run_arcface_stage(student: nn.Module, train_files: List[str], train_dir: Path,
                      args: argparse.Namespace, out_dir: Path, device: torch.device):
    try:
        from pytorch_metric_learning.losses import ArcFaceLoss
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "pytorch-metric-learning"])
        from pytorch_metric_learning.losses import ArcFaceLoss

    from tqdm.auto import tqdm

    identity_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    num_identities = len(identity_names)
    label_to_idx   = {name: idx for idx, name in enumerate(identity_names)}
    log.info(f"ArcFace stage: {num_identities} identities")

    loader = DataLoader(
        IdentityLabeledData(train_files, label_to_idx),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    for p in student.backbone.parameters():
        p.requires_grad = False
    student = student.to(device)

    arc_loss_fn = ArcFaceLoss(num_classes=num_identities,
                               embedding_size=args.embed_dim).to(device)
    optimizer   = torch.optim.Adam(
        list(student.projection_head.parameters()) + list(arc_loss_fn.parameters()),
        lr=args.arcface_lr,
    )

    student.train()
    for epoch in range(args.arcface_epochs):
        total = 0.0
        for imgs, labels in tqdm(loader, desc=f"ArcFace {epoch+1}/{args.arcface_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            loss = arc_loss_fn(student(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        log.info(f"ArcFace epoch {epoch+1} | loss {total/len(loader):.4f}")

    ckpt = out_dir / "arcface_finetuned.pt"
    torch.save(student.state_dict(), ckpt)
    log.info(f"ArcFace checkpoint saved: {ckpt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VGGFace2 DINO ViT-Base + ArcFace")
    p.add_argument("--data",           required=True,
                   help="VGGFace2 root — must contain train/ subfolder")
    p.add_argument("--out",            default="./checkpoints")
    p.add_argument("--backbone",       default="dino_vitb16",
                   choices=["dino_vits16", "dino_vitb16"])
    p.add_argument("--backbone-dim",   type=int, default=768,
                   help="768 for vitb16, 384 for vits16")
    p.add_argument("--embed-dim",      type=int, default=256)
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--warmup-epochs",  type=int, default=2)
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--lr",             type=float, default=3e-5)
    p.add_argument("--workers",        type=int, default=4)
    p.add_argument("--val-fraction",   type=float, default=0.15)
    p.add_argument("--save-every",     type=int, default=3)
    p.add_argument("--precision",      type=int, default=16, choices=[16, 32])
    p.add_argument("--arcface",        action="store_true",
                   help="Run ArcFace supervised stage after DINO")
    p.add_argument("--arcface-epochs", type=int, default=10)
    p.add_argument("--arcface-lr",     type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()

    train_dir = Path(args.data) / "train"
    if not train_dir.exists():
        sys.exit(
            f"[error] Train folder not found at: {train_dir}\n"
            f"  Download from: https://www.kaggle.com/datasets/hearfool/vggface2\n"
            f"  Expected: {args.data}/train/n000001/img.jpg ..."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(str(f) for f in train_dir.rglob("*.jpg"))
    if not all_files:
        sys.exit(f"[error] No .jpg files found under {train_dir}")
    log.info(f"Found {len(all_files):,} images in {train_dir}")

    train_files, val_files = train_test_split(all_files, test_size=args.val_fraction, random_state=42)
    log.info(f"Train: {len(train_files):,} | Val: {len(val_files):,}")

    train_loader = DataLoader(
        ImageData(train_files),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_multicrop,
    )
    val_loader = DataLoader(
        ImageValData(val_files),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_val,
    )

    log.info(f"Backbone: {args.backbone} | backbone_dim={args.backbone_dim} | embed_dim={args.embed_dim}")
    log.info(f"Multi-crop: {N_GLOBAL_CROPS} global + {N_LOCAL_CROPS} local views")

    ckpt_callback = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="dino_epoch={epoch:02d}",
        save_last=True,
        every_n_epochs=args.save_every,
    )

    teacher = DinoModel(args.backbone, args.backbone_dim, args.embed_dim)
    model   = DINOLightning(
        teacher,
        lr=args.lr,
        loss_fn=HLoss(TEMPERATURE_T, TEMPERATURE_S),
        embed_dim=args.embed_dim,
        center_momentum=CENTER_MOMENTUM,
        param_momentum=TEACHER_MOMENTUM,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    last_ckpt   = out_dir / "last.ckpt"
    resume_path = str(last_ckpt) if last_ckpt.exists() else None
    log.info(f"{'Resuming from: ' + resume_path if resume_path else 'Starting fresh.'}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=max(1, torch.cuda.device_count()),
        gradient_clip_val=1.0,
        precision=args.precision,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    if args.arcface:
        log.info("Starting ArcFace fine-tuning stage...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_arcface_stage(model.student, train_files, train_dir, args, out_dir, device)

    summary = {
        "backbone": args.backbone,
        "backbone_dim": args.backbone_dim,
        "embed_dim": args.embed_dim,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_global_crops": N_GLOBAL_CROPS,
        "n_local_crops": N_LOCAL_CROPS,
        "arcface": args.arcface,
        "arcface_epochs": args.arcface_epochs if args.arcface else 0,
        "num_train_images": len(train_files),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Training complete. Checkpoints in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
