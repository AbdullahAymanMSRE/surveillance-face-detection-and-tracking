#!/usr/bin/env python3
"""
Evaluate a trained VGGFace2 DINO+ArcFace checkpoint on the held-out val split.

Rebuilds the exact train/val split used by dino_vggface2.py (same glob order,
same random_state) so the val images here were never seen during training,
then reports:
  - global rank-1 accuracy (nearest neighbor by cosine sim, over the whole
    val set, not just within a random mini-batch like the training-time metric)
  - verification AUC + accuracy on balanced genuine/impostor pairs

Run:
    python eval_dino_vggface2.py \
        --data /scratch/gilbreth/$USER/vggface2 \
        --run-dir runs/dino_20260616_203019
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dino_vggface2 import DinoModel, ImageValData, collate_val


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DINO+ArcFace VGGFace2 checkpoint")
    p.add_argument("--data", required=True, help="VGGFace2 root (contains train/)")
    p.add_argument("--run-dir", required=True, help="Run dir with run_summary.json + checkpoint")
    p.add_argument("--checkpoint", default="arcface_finetuned.pt",
                   help="Checkpoint filename inside --run-dir")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Must match the value used at training time")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-pairs", type=int, default=20000,
                   help="Genuine + impostor pairs each for verification AUC")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_val_files(data_root: Path, val_fraction: float):
    train_dir = data_root / "train"
    all_files = sorted(str(f) for f in train_dir.rglob("*.jpg"))
    _, val_files = train_test_split(all_files, test_size=val_fraction, random_state=42)
    return val_files


@torch.no_grad()
def compute_embeddings(model, val_files, batch_size, workers, device):
    loader = DataLoader(
        ImageValData(val_files),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_val,
    )
    all_embeds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             dtype=torch.float16, enabled=device.type == "cuda"):
            emb = model(imgs)
        all_embeds.append(emb.float().cpu())
        all_labels.extend(labels)
    return torch.cat(all_embeds), all_labels


def rank1_accuracy(embeds: torch.Tensor, labels, device, chunk_size=1024):
    n = embeds.shape[0]
    embeds = embeds.to(device)
    labels_t = labels
    correct = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = embeds[start:end] @ embeds.T
        for i in range(start, end):
            sims[i - start, i] = -2.0
        nearest = sims.argmax(dim=1).cpu()
        for offset, idx in enumerate(nearest):
            if labels_t[start + offset] == labels_t[idx.item()]:
                correct += 1
    return correct / n


def verification_metrics(embeds: torch.Tensor, labels, num_pairs: int, seed: int):
    rng = random.Random(seed)
    by_label = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(lbl, []).append(i)
    multi_label_ids = [i for i, lbl in enumerate(labels) if len(by_label[lbl]) > 1]
    all_labels = list(by_label.keys())

    sims, targets = [], []

    genuine_made = 0
    attempts = 0
    while genuine_made < num_pairs and attempts < num_pairs * 20:
        attempts += 1
        i = rng.choice(multi_label_ids)
        candidates = [j for j in by_label[labels[i]] if j != i]
        j = rng.choice(candidates)
        sims.append(F.cosine_similarity(embeds[i].unsqueeze(0), embeds[j].unsqueeze(0)).item())
        targets.append(1)
        genuine_made += 1

    impostor_made = 0
    while impostor_made < num_pairs:
        i = rng.randrange(len(labels))
        j = rng.randrange(len(labels))
        if labels[i] == labels[j]:
            continue
        sims.append(F.cosine_similarity(embeds[i].unsqueeze(0), embeds[j].unsqueeze(0)).item())
        targets.append(0)
        impostor_made += 1

    auc = roc_auc_score(targets, sims)

    best_acc, best_thresh = 0.0, 0.0
    for thresh in [t / 200.0 for t in range(-200, 201)]:
        preds = [1 if s >= thresh else 0 for s in sims]
        acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    return {
        "auc": auc,
        "best_accuracy": best_acc,
        "best_threshold": best_thresh,
        "num_genuine_pairs": genuine_made,
        "num_impostor_pairs": impostor_made,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())
    backbone     = summary["backbone"]
    backbone_dim = summary["backbone_dim"]
    embed_dim    = summary["embed_dim"]
    print(f"[eval] backbone={backbone} backbone_dim={backbone_dim} embed_dim={embed_dim}")

    val_files = build_val_files(Path(args.data), args.val_fraction)
    print(f"[eval] Val images: {len(val_files):,}")

    model = DinoModel(backbone, backbone_dim, embed_dim)
    state_dict = torch.load(run_dir / args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    embeds, labels = compute_embeddings(model, val_files, args.batch_size, args.workers, device)
    print(f"[eval] Computed {embeds.shape[0]:,} embeddings (dim={embeds.shape[1]})")

    acc = rank1_accuracy(embeds, labels, device)
    print(f"[eval] Global rank-1 accuracy: {acc:.4f}")

    verif = verification_metrics(embeds, labels, args.num_pairs, args.seed)
    print(f"[eval] Verification AUC: {verif['auc']:.4f} | "
          f"best accuracy: {verif['best_accuracy']:.4f} @ threshold {verif['best_threshold']:.3f}")

    results = {
        "checkpoint": str(run_dir / args.checkpoint),
        "num_val_images": len(val_files),
        "num_identities": len(set(labels)),
        "rank1_accuracy": acc,
        "verification": verif,
    }
    out_path = run_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] Saved results to {out_path}")


if __name__ == "__main__":
    main()
