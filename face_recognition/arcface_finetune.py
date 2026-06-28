#!/usr/bin/env python3
"""
ArcFace ONNX production-model fine-tuning — VGGFace2
Ahmed Elmersawy — Purdue University / davisjam lab

Loads the deployed surveillance recognizer (w600k_mbf.onnx, a MobileFaceNet
trained with the ArcFace margin loss), converts it to a trainable PyTorch
module with onnx2torch (carrying the production weights), fine-tunes the
last few embedding layers on identity-labeled face crops with an ArcFace
margin head, then exports the adapted weights back to ONNX so the live
system can run on them.

This is the batch-runnable counterpart of arcface_finetune.ipynb — same
training logic, no notebook runtime needed for SLURM.

Run:
    python arcface_finetune.py \
        --onnx-in   w600k_mbf.onnx \
        --onnx-out  runs/arcface_onnx_<ts>/w600k_mbf_finetuned.onnx \
        --data      /scratch/gilbreth/$USER/vggface2/train \
        --out       runs/arcface_onnx_<ts> \
        --epochs    3 --unfreeze-last 20

    Best-effort run (full backbone, cosine LR decay, more epochs):
        --epochs 60 --warmup-epochs 3 --lr 3e-5 --unfreeze-last 9999
"""

import argparse
import copy
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

try:
    from onnx2torch import convert
except ImportError:
    sys.exit("[error] onnx2torch not installed.  Run: pip install onnx onnx2torch onnxruntime")


# ── ArcFace margin head ──────────────────────────────────────────────────────
class ArcMarginHead(nn.Module):
    def __init__(self, in_features=512, out_features=10572, s=64.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        x = F.normalize(embeddings)
        w = F.normalize(self.weight)
        cos = F.linear(x, w).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, num_classes=w.size(0)).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


# ── Evidence: embedding separation gap ───────────────────────────────────────
@torch.no_grad()
def _embed(model, imgs, device):
    return F.normalize(model(imgs.to(device))).cpu()


@torch.no_grad()
def same_diff_gap(model, ds, device, n_pairs=300, seed=0):
    by_cls = {}
    for idx, (_, c) in enumerate(ds.samples):
        by_cls.setdefault(c, []).append(idx)
    classes = [c for c, v in by_cls.items() if len(v) >= 2]
    rng = np.random.default_rng(seed)

    def emb_idx(i):
        img, _ = ds[i]
        return _embed(model, img.unsqueeze(0), device)[0]

    same, diff = [], []
    for _ in range(n_pairs):
        c = classes[rng.integers(len(classes))]
        i, j = rng.choice(by_cls[c], 2, replace=False)
        same.append(float(emb_idx(i) @ emb_idx(j)))
        c2 = classes[rng.integers(len(classes))]
        while c2 == c:
            c2 = classes[rng.integers(len(classes))]
        a = by_cls[c][rng.integers(len(by_cls[c]))]
        b = by_cls[c2][rng.integers(len(by_cls[c2]))]
        diff.append(float(emb_idx(a) @ emb_idx(b)))
    return float(np.mean(same)), float(np.mean(diff))


# ── Evidence: LFW-style verification benchmarks (optional, skipped if missing) ──
def load_bin(path):
    import cv2
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    imgs = []
    for b in bins:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs, np.asarray(issame, dtype=bool)


@torch.no_grad()
def _embed_imgs(model, imgs, bin_tf, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        t = torch.stack([bin_tf(im) for im in imgs[i:i + batch]]).to(device)
        e = F.normalize(model(t)) + F.normalize(model(torch.flip(t, dims=[3])))
        out.append(F.normalize(e).cpu().numpy())
    return np.concatenate(out)


def _best_threshold_acc(sims, actual, folds=10):
    thresholds = np.linspace(-1.0, 1.0, 400)
    chunks = np.array_split(np.arange(len(actual)), folds)
    accs = []
    for k in range(folds):
        test = chunks[k]
        train = np.concatenate([chunks[j] for j in range(folds) if j != k])
        best_t = thresholds[np.argmax(
            [((sims[train] > t) == actual[train]).mean() for t in thresholds])]
        accs.append(((sims[test] > best_t) == actual[test]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def verification_accuracy(model, imgs, issame, bin_tf, device):
    e = _embed_imgs(model, imgs, bin_tf, device)
    sims = np.sum(e[0::2] * e[1::2], axis=1)
    return _best_threshold_acc(sims, issame)


def run_lfw_style_benchmarks(backbone_orig, backbone, val_dir, device):
    if val_dir is None:
        return {}
    bin_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    results = {}
    for name in ["lfw", "cfp_fp", "agedb_30"]:
        p = Path(val_dir) / f"{name}.bin"
        if not p.exists():
            log.info(f"benchmark {name}: missing {p}, skipping")
            continue
        imgs, issame = load_bin(p)
        a0, _ = verification_accuracy(backbone_orig, imgs, issame, bin_tf, device)
        a1, _ = verification_accuracy(backbone, imgs, issame, bin_tf, device)
        log.info(f"benchmark {name}: pretrained={a0*100:.2f}%  finetuned={a1*100:.2f}%")
        results[name] = {"pretrained": a0, "finetuned": a1}
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-in", required=True, help="Production ONNX model to fine-tune")
    p.add_argument("--onnx-out", required=True, help="Path to write the fine-tuned ONNX model")
    p.add_argument("--data", required=True,
                   help="ImageFolder root of identity subfolders, e.g. .../vggface2/train")
    p.add_argument("--out", required=True, help="Run directory for logs/summary")
    p.add_argument("--val-bin-dir", default=None,
                   help="Optional dir with lfw.bin/cfp_fp.bin/agedb_30.bin for benchmark eval")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine decay")
    p.add_argument("--unfreeze-last", type=int, default=20,
                   help="Number of trailing parameter tensors to unfreeze "
                        "(pass a number >= total tensor count to unfreeze everything)")
    p.add_argument("--arcface-s", type=float, default=64.0)
    p.add_argument("--arcface-m", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = Path(args.onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    onnx_in = Path(args.onnx_in)
    if not onnx_in.exists():
        sys.exit(f"[error] production model not found at {onnx_in.resolve()}")

    log.info(f"loading production ONNX model from {onnx_in}")
    backbone = convert(str(onnx_in)).to(device)

    backbone_orig = copy.deepcopy(backbone).eval()
    for param in backbone_orig.parameters():
        param.requires_grad_(False)

    n_params = sum(param.numel() for param in backbone.parameters())
    log.info(f"loaded production backbone: {n_params/1e6:.2f}M parameters")

    with torch.no_grad():
        dummy = torch.randn(2, 3, 112, 112, device=device)
        embed_dim = backbone(dummy).shape[1]
    log.info(f"embedding dim: {embed_dim}")
    assert embed_dim == 512, f"expected 512-d embedding, got {embed_dim}"

    train_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = datasets.ImageFolder(args.data, transform=train_tf)
    eval_ds = datasets.ImageFolder(args.data, transform=eval_tf)
    num_classes = len(train_ds.classes)
    log.info(f"{len(train_ds)} images across {num_classes} identities")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    head = ArcMarginHead(embed_dim, num_classes, s=args.arcface_s, m=args.arcface_m).to(device)

    params = list(backbone.named_parameters())
    for _, param in params:
        param.requires_grad_(False)
    trainable = []
    for name, param in params[-args.unfreeze_last:]:
        param.requires_grad_(True)
        trainable.append(param)
        log.info(f"unfrozen: {name} {tuple(param.shape)}")

    before_snapshot = {name: param.detach().clone() for name, param in params[-args.unfreeze_last:]}

    optimizer = torch.optim.AdamW(trainable + list(head.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, total_iters=max(warmup_epochs, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    log.info(f"starting fine-tune... (warmup_epochs={warmup_epochs}, base_lr={args.lr})")
    backbone.train()
    loss_history = []
    for epoch in range(args.epochs):
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            emb = backbone(imgs)
            logits = head(emb, labels)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        avg = running / len(train_loader)
        loss_history.append(avg)
        cur_lr = optimizer.param_groups[0]["lr"]
        log.info(f"epoch {epoch+1}/{args.epochs} | mean loss = {avg:.4f} | lr = {cur_lr:.2e}")
    backbone.eval()

    log.info("evidence: weights actually changed")
    now = dict(backbone.named_parameters())
    weight_deltas = {}
    for name, before in before_snapshot.items():
        delta = (now[name].detach() - before).abs().mean().item()
        weight_deltas[name] = delta
        log.info(f"{name[:48]:50s} mean|delta|={delta:.6e}")

    log.info("evidence: embedding separation before vs after")
    s0, d0 = same_diff_gap(backbone_orig, eval_ds, device, seed=args.seed)
    s1, d1 = same_diff_gap(backbone, eval_ds, device, seed=args.seed)
    log.info(f"ORIGINAL : same={s0:.3f} diff={d0:.3f} gap={s0-d0:.3f}")
    log.info(f"FINETUNED: same={s1:.3f} diff={d1:.3f} gap={s1-d1:.3f}")

    benchmark_results = run_lfw_style_benchmarks(backbone_orig, backbone, args.val_bin_dir, device)

    log.info(f"exporting fine-tuned model to {onnx_out}")
    backbone.eval().cpu()
    dummy_cpu = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        backbone, dummy_cpu, str(onnx_out),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13,
    )

    # torch.onnx.export may split large models into a small graph file plus an
    # external ".data" weights file. Re-save as a single self-contained file
    # so it stays a drop-in replacement for the original production model
    # (which ships as one file) and isn't fragile to a missing sidecar.
    import onnx as _onnx
    _model = _onnx.load(str(onnx_out))
    _onnx.save_model(_model, str(onnx_out), save_as_external_data=False)
    stray_data = Path(str(onnx_out) + ".data")
    if stray_data.exists():
        stray_data.unlink()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    blob = np.random.randn(1, 3, 112, 112).astype(np.float32)
    vec = sess.run(None, {inp_name: blob})[0]
    log.info(f"onnxruntime sanity check output shape: {vec.shape}")
    assert vec.shape == (1, 512)

    summary = {
        "onnx_in": str(onnx_in),
        "onnx_out": str(onnx_out),
        "num_train_images": len(train_ds),
        "num_identities": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "unfreeze_last": args.unfreeze_last,
        "arcface_s": args.arcface_s,
        "arcface_m": args.arcface_m,
        "loss_history": loss_history,
        "separation_gap": {"original": s0 - d0, "finetuned": s1 - d1},
        "benchmarks": benchmark_results,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out_dir / 'run_summary.json'}")
    log.info("done.")


if __name__ == "__main__":
    main()
