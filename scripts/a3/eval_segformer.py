"""
Evaluate SegFormer (HF transformers) checkpoint on SpheroMix test or HTS-Seg.

Same metric format/output as evaluate_a3.py — drop-in for A1 summary.

Usage:
  python eval_segformer.py --weights .../best.pth \
      --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
      --dataset $SPHEROMIX_PATH --output out.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

# Heavy imports (CNN_main_spheroid, albumentations, transformers) are deferred
# to main() so that --help works without those packages installed.


def per_image_metrics(pred_mask, gt_mask):
    p = pred_mask.astype(bool); g = gt_mask.astype(bool)
    inter = (p & g).sum(); union = (p | g).sum()
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
    return float(values.mean()), float(np.percentile(means, (100-ci)/2)), float(np.percentile(means, 100-(100-ci)/2))


def is_hq_test(filename: str) -> bool:
    return 'bxpc' in filename.lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   required=True)
    p.add_argument("--pretrained", default="nvidia/segformer-b0-finetuned-ade-512-512")
    p.add_argument("--dataset",   default=os.environ.get("SPHEROMIX_PATH"),
                   help="Path to dataset root (env: SPHEROMIX_PATH)")
    p.add_argument("--output",    required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    a = p.parse_args()
    if not a.dataset:
        p.error("--dataset is required (or set SPHEROMIX_PATH env var)")

    from CNN_main_spheroid import CachedSpheroidDataset
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    from transformers import SegformerForSemanticSegmentation

    # Load model
    print(f"[eval] loading SegFormer pretrained={a.pretrained}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        a.pretrained, num_labels=2, ignore_mismatched_sizes=True
    ).to("cuda").eval()
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[eval] strict load: missing={len(missing)} unexpected={len(unexpected)}")

    # Same eval transform as A3 evaluator
    val_tf = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    ds = CachedSpheroidDataset(a.dataset, split='test', transform=val_tf, use_cache=False)
    print(f"[eval] test images: {len(ds)}")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)

    rows = []
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(dl):
            imgs = imgs.to("cuda", non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
            # 2-class softmax → argmax → fg=class 1
            pred = (logits.argmax(dim=1)).cpu().numpy()
            gts  = (masks > 0.5).squeeze(1).cpu().numpy()
            for i in range(pred.shape[0]):
                gi = batch_idx * a.batch_size + i
                fname = ds.valid_files[gi][0].name if gi < len(ds.valid_files) else f"idx{gi}"
                iou, dice, prec, rec = per_image_metrics(pred[i], gts[i])
                rows.append({"file": fname, "iou": iou, "dice": dice, "precision": prec, "recall": rec, "is_hq": is_hq_test(fname)})

    def summarize(rs):
        if not rs: return None
        out = {"n": len(rs)}
        for k in ("iou","dice","precision","recall"):
            mean, lo, hi = bootstrap_ci([r[k] for r in rs])
            out[k] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        return out

    hq = [r for r in rows if r["is_hq"]]
    dts = [r for r in rows if not r["is_hq"]]
    print(f"[eval] {len(rows)} total, {len(hq)} HQ, {len(dts)} non-HQ")
    summary = {
        "weights": str(a.weights),
        "pretrained": a.pretrained,
        "model": "segformer",
        "n_total": len(rows),
        "unified": summarize(rows),
        "hq_only": summarize(hq),
        "dts_only": summarize(dts),
        "per_image": rows,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(summary, f, indent=2)
    for split, s in [("unified", summary["unified"]), ("HQ", summary["hq_only"]), ("DTS", summary["dts_only"])]:
        if s and 'iou' in s:
            i = s['iou']
            print(f"  {split:<10} iou = {i['mean']:.4f}  [{i['ci_low']:.4f}, {i['ci_high']:.4f}]   (n={s['n']})")
    print(f"[eval] wrote {a.output}")


if __name__ == "__main__":
    main()
