#!/usr/bin/env python
"""
A3 evaluation — produces the standardized-protocol comparison table.

For each (model, weights) pair, computes:
  - IoU, Dice, Precision, Recall (mean and per-image)
  - 95% bootstrap CI (10k iterations) per metric
  - Stratified: HQ-only test, DTS-only test, unified n=1019

Outputs JSON per (model,variant) and a combined CSV.

Usage:
  python evaluate_a3.py --weights <path.pth> --model <key> --output <json>
  python evaluate_a3.py --batch <run_dir>     # auto-eval all best_model.pth in dir tree
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

# Reuse dataset class from the training script
from CNN_main_spheroid import CachedSpheroidDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CTORS = {
    "unet":            lambda: __import__('models.unet', fromlist=['UNet']).UNet(in_channels=3, out_channels=1, use_instance_norm=True),
    "hrnet":           lambda: __import__('models.hrnet', fromlist=['HRNetV2']).HRNetV2(n_class=1, pretrained=False, use_instance_norm=True),
    "pspnet":          lambda: __import__('models.pspnet', fromlist=['PSPNet']).PSPNet(n_class=1, backbone='resnet101', pretrained=False, use_instance_norm=True),
    "lightm_unet":     lambda: __import__('models.lightm_unet', fromlist=['LightMUNet']).LightMUNet(in_channels=3, out_channels=1, use_instance_norm=True),
    "resunet_cbam":    lambda: __import__('models.resunet_cbam', fromlist=['ResUNetCBAM']).ResUNetCBAM(in_channels=3, out_channels=1, use_instance_norm=True),
    "resunet_lc":      lambda: __import__('models.resunet_lc', fromlist=['ResUNetSmall']).ResUNetSmall(in_channels=3, out_channels=1, use_instance_norm=True),
    "resunet_ma":      lambda: __import__('models.resunet_ma', fromlist=['AdvancedResUNet']).AdvancedResUNet(in_channels=3, out_channels=1, use_instance_norm=True),
    "resunet_ma_mini": lambda: __import__('models.resunet_ma_mini', fromlist=['AdvancedResUNet']).AdvancedResUNet(in_channels=3, out_channels=1, use_instance_norm=True),
}

def per_image_metrics(pred_mask, gt_mask):
    """All metrics per single image. pred and gt are 2D bool/uint8."""
    p = pred_mask.astype(bool); g = gt_mask.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    pp = p.sum(); gp = g.sum()
    iou  = inter / union if union > 0 else 1.0
    dice = (2*inter) / (pp + gp) if (pp + gp) > 0 else 1.0
    prec = inter / pp if pp > 0 else (1.0 if gp == 0 else 0.0)
    rec  = inter / gp if gp > 0 else (1.0 if pp == 0 else 0.0)
    return iou, dice, prec, rec

def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = np.percentile(means, (100-ci)/2)
    hi = np.percentile(means, 100-(100-ci)/2)
    return float(values.mean()), float(lo), float(hi)

def is_hq_test(filename: str) -> bool:
    """SpheroMix test split contains BxPC-3 (HQ) + DTS images.
    HQ images come from BxPC-3 cell line; identifiable by 'bxpc-3' in name (case-insensitive).
    DTS images use a different naming convention.
    Verify this assumption when SpheroMix unzips.
    """
    return 'bxpc' in filename.lower() or 'bxpc-3' in filename.lower()

def evaluate(weights_path, model_key, dataset_path, device='cuda:0', batch_size=4, num_workers=4):
    print(f"[eval] weights={weights_path} model={model_key}")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.','',1): v for k,v in sd.items()}

    model = CTORS[model_key]().to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] strict load issues: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # Same eval transform as training validation
    val_tf = A.Compose([
        A.Resize(1024,1024),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    ds = CachedSpheroidDataset(dataset_path, split='test', transform=val_tf, use_cache=False)
    print(f"[eval] test images: {len(ds)}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    rows = []
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(dl):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                if isinstance(logits, tuple): logits = logits[0]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5)
            preds = preds.squeeze(1).cpu().numpy()
            gts   = (masks > 0.5).squeeze(1).cpu().numpy()
            for i in range(preds.shape[0]):
                global_idx = batch_idx*batch_size + i
                fname = ds.valid_files[global_idx][0].name if global_idx < len(ds.valid_files) else f"idx{global_idx}"
                iou,dice,prec,rec = per_image_metrics(preds[i], gts[i])
                rows.append({"file": fname, "iou": iou, "dice": dice, "precision": prec, "recall": rec, "is_hq": is_hq_test(fname)})

    # Stratify
    def summarize(rs, label):
        if not rs: return None
        for k in ("iou","dice","precision","recall"):
            mean, lo, hi = bootstrap_ci([r[k] for r in rs])
            print(f"  {label:<10} {k:<10} = {mean:.4f}  [{lo:.4f}, {hi:.4f}]   (n={len(rs)})")
        out = {"n": len(rs)}
        for k in ("iou","dice","precision","recall"):
            mean, lo, hi = bootstrap_ci([r[k] for r in rs])
            out[k] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        return out

    hq  = [r for r in rows if r["is_hq"]]
    dts = [r for r in rows if not r["is_hq"]]
    print(f"\n[eval] {len(rows)} images total, {len(hq)} HQ-like, {len(dts)} non-HQ")
    summary = {
        "weights": str(weights_path),
        "model": model_key,
        "n_total": len(rows),
        "unified": summarize(rows, "unified"),
        "hq_only":  summarize(hq, "HQ"),
        "dts_only": summarize(dts, "DTS"),
        "per_image": rows,
    }
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model",   required=True, choices=list(CTORS.keys()))
    p.add_argument("--dataset", default=os.environ.get("SPHEROMIX_PATH"),
                   help="Path to SpheroMix dataset root (env: SPHEROMIX_PATH). "
                        "For HQ-only or DTS-only eval, point at SpheroHQ / DTS root instead.")
    p.add_argument("--output",  required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if not args.dataset:
        p.error("--dataset is required (or set SPHEROMIX_PATH env var)")

    summary = evaluate(args.weights, args.model, args.dataset,
                       batch_size=args.batch_size, num_workers=args.num_workers)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {args.output}")
