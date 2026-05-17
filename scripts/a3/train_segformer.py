#!/usr/bin/env python3
"""Fine-tune SegFormer (B0/B2/B4) for binary spheroid segmentation on SpheroMix.

"Fair" training protocol used for Table 8 of the CMPB manuscript:
  - Train at 512×512, evaluate at 1024×1024 (logits upsampled bilinearly)
  - AdamW + OneCycleLR (peak LR 6e-5, pct_start 0.1)
  - 40 epochs, effective batch size = batch_size × accum_steps
  - cross-entropy loss (2-class softmax), mixed precision (bfloat16)
  - Pre-trained checkpoints from HuggingFace (`nvidia/segformer-b{0,2,4}-...`)
  - Best by validation IoU; final eval on test set with bootstrap 95% CI

Uses HuggingFace `transformers` + `torch`. The companion evaluator
`scripts/a3/eval_segformer.py` produces the JSON schema consumed by Tables
2 / 7 / 8 of the manuscript.

Example:
    export SPHEROMIX_PATH=/path/to/SpheroMix
    python scripts/a3/train_segformer.py \
        --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
        --out-dir checkpoints/segformer_b0_fair \
        --epochs 40 --batch-size 8 --accum-steps 1
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation

SEED = 42


def set_seed():
    import random
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SpheroidDS(Dataset):
    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    def __init__(self, split_root: Path, image_size: int = 512,
                 train: bool = False):
        self.img_dir = split_root / "images"
        self.mask_dir = split_root / "masks"
        self.image_size = image_size
        self.train = train
        self.items = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            and not p.name.startswith(".")
        )

    def __len__(self):
        return len(self.items)

    def _mask_for(self, img_path):
        m = self.mask_dir / img_path.name
        if m.exists():
            return m
        m = self.mask_dir / (img_path.stem + ".png")
        if m.exists():
            return m
        raise FileNotFoundError(img_path)

    def __getitem__(self, idx):
        ip = self.items[idx]
        mp = self._mask_for(ip)
        im = Image.open(ip).convert("RGB")
        mk = Image.open(mp).convert("L")

        im = im.resize((self.image_size, self.image_size), Image.BILINEAR)
        mk = mk.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.train:
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                mk = mk.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                mk = mk.transpose(Image.FLIP_TOP_BOTTOM)

        x = np.asarray(im, dtype=np.float32)
        x = (x - self.IMG_MEAN) / self.IMG_STD
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        y = (np.asarray(mk, dtype=np.uint8) > 127).astype(np.int64)
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).long(),
                ip.name)


def iou_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    inter = ((pred == 1) & (target == 1)).sum().item()
    union = ((pred == 1) | (target == 1)).sum().item()
    return inter / union if union > 0 else 0.0


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    p = (pred == 1).sum().item()
    t = (target == 1).sum().item()
    inter = ((pred == 1) & (target == 1)).sum().item()
    return 2.0 * inter / (p + t) if (p + t) > 0 else 0.0


def evaluate(model, loader, device, image_size_eval=1024):
    """Eval at full resolution by upsampling logits."""
    model.eval()
    rows = []
    total_ms = 0.0
    with torch.inference_mode():
        for x, y, names in tqdm(loader, desc="val"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t0 = time.time()
            out = model(pixel_values=x)
            logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False)
            if logits.shape[-1] != image_size_eval:
                logits = F.interpolate(logits, size=(image_size_eval, image_size_eval),
                                        mode="bilinear", align_corners=False)
                y_eval = F.interpolate(y.unsqueeze(1).float(),
                                        size=(image_size_eval, image_size_eval),
                                        mode="nearest").squeeze(1).long()
            else:
                y_eval = y
            ms = (time.time() - t0) * 1000.0
            total_ms += ms

            for i in range(logits.size(0)):
                lo = logits[i:i + 1]
                yy = y_eval[i:i + 1]
                rows.append({
                    "image": names[i],
                    "iou": iou_score(lo, yy),
                    "dice": dice_score(lo, yy),
                    "infer_ms": ms / logits.size(0),
                })
    return rows, total_ms


def main():
    ap = argparse.ArgumentParser(description="SegFormer fair-protocol trainer (Table 8)")
    ap.add_argument("--data-root", default=os.environ.get("SPHEROMIX_PATH"),
                    help="SpheroMix root containing train/, val/, test/ "
                         "(env: SPHEROMIX_PATH)")
    ap.add_argument("--out-dir", required=True,
                    help="Where to save best.pth, train_history.json, summary")
    ap.add_argument("--pretrained", default="nvidia/segformer-b0-finetuned-ade-512-512",
                    help="HF model ID. For B0/B2/B4 fair baselines used in Table 8 use "
                         "segformer-b{0,2,4}-finetuned-ade-512-512")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--eval-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="Gradient-accumulation steps; effective BS = batch_size × accum_steps")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training; eval from existing best.pth in --out-dir")
    args = ap.parse_args()

    if not args.data_root:
        ap.error("--data-root is required (or set SPHEROMIX_PATH env var)")

    set_seed()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[segformer] device={device} pretrained={args.pretrained}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    if not args.eval_only:
        train_ds = SpheroidDS(data_root / "train", image_size=args.train_size, train=True)
        val_ds = SpheroidDS(data_root / "val", image_size=args.train_size, train=False)
        print(f"train n={len(train_ds)} val n={len(val_ds)}", flush=True)

        train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
        val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps_per_epoch = max(1, len(train_ld) // args.accum_steps)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.1)

        scaler = torch.amp.GradScaler("cuda")
        best_iou = 0.0
        hist = []
        eff_bs = args.batch_size * args.accum_steps
        print(f"effective batch size = {eff_bs} (bs={args.batch_size} × accum={args.accum_steps})",
              flush=True)

        for epoch in range(args.epochs):
            model.train()
            bar = tqdm(train_ld, desc=f"ep{epoch}")
            run_loss = 0.0
            n = 0
            opt.zero_grad(set_to_none=True)
            for step, (x, y, _) in enumerate(bar):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(pixel_values=x)
                    logits = F.interpolate(out.logits, size=x.shape[-2:],
                                            mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits, y) / args.accum_steps
                scaler.scale(loss).backward()
                run_loss += loss.item() * args.accum_steps * x.size(0)
                n += x.size(0)
                if (step + 1) % args.accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                bar.set_postfix(loss=run_loss / max(n, 1))

            rows, _ = evaluate(model, val_ld, device, image_size_eval=args.train_size)
            mean_iou = float(np.mean([r["iou"] for r in rows]))
            mean_dice = float(np.mean([r["dice"] for r in rows]))
            hist.append({"epoch": epoch, "loss": run_loss / max(n, 1),
                         "val_iou": mean_iou, "val_dice": mean_dice})
            print(f"[ep{epoch}] loss={run_loss / max(n, 1):.4f} "
                  f"val_iou={mean_iou:.4f} val_dice={mean_dice:.4f}", flush=True)
            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save(model.state_dict(), out_dir / "best.pth")
                print(f"  -> new best @ ep{epoch} iou={best_iou:.4f}", flush=True)

        with open(out_dir / "train_history.json", "w") as f:
            json.dump({"hist": hist, "best_iou": best_iou}, f, indent=2)

    # Final eval at full resolution on test set
    ckpt = out_dir / "best.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("[warn] no best.pth found; evaluating current weights", flush=True)

    test_ds = SpheroidDS(data_root / "test", image_size=args.eval_size, train=False)
    test_ld = DataLoader(test_ds, batch_size=2, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    rows, total_ms = evaluate(model, test_ld, device, image_size_eval=args.eval_size)

    with open(out_dir / "test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "iou", "dice", "infer_ms"])
        w.writeheader()
        w.writerows(rows)

    mean_iou = float(np.mean([r["iou"] for r in rows]))
    mean_dice = float(np.mean([r["dice"] for r in rows]))
    rng = np.random.default_rng(SEED)
    arr = np.array([r["iou"] for r in rows])
    boots = rng.choice(arr, size=(10000, len(arr)), replace=True).mean(axis=1)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    summary = {
        "model": "segformer",
        "pretrained": args.pretrained,
        "eval_size": args.eval_size,
        "n": len(rows),
        "iou_mean": mean_iou,
        "iou_ci95": ci,
        "dice_mean": mean_dice,
        "wall_ms_per_image": total_ms / max(len(rows), 1),
        "seed": SEED,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
