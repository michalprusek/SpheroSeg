"""Inter-annotator agreement (IAA) on the 150-image DTS re-annotation pilot.

Reproduces the §5.5.1 / Table 8 IAA numbers reported in the CMPB manuscript.

Inputs (organised under --raters-root):
  raters-root/
    a1/annotations/instances_default.json    CVAT COCO export, annotator 1
    a2/annotations/instances_default.json    annotator 2
    a3/annotations/instances_default.json    annotator 3
    a4/annotations/instances_default.json    annotator 4
  dts-gt-dir/<basename>.png                  DTS-original binary masks (0/255)

Outputs (in --out-dir):
  iaa_pairwise_iou.csv     6 pair × 150 image per-image IoU/Dice/precision/recall
  iaa_summary.json         pairwise mean ± CI, Fleiss' kappa, STAPLE consensus,
                           per-rater-vs-DTS IoU, policy gap, object-count kappa,
                           Bland-Altman area panel
  iaa_heatmap.pdf          5×5 mean IoU heatmap (4 raters + DTS-original)
  iaa_bland_altman_area.pdf  consensus-vs-DTS area scatter with limits of agreement

Usage:
  python scripts/a3/iaa_analysis.py \
      --raters-root path/to/cvat_exports \
      --dts-gt-dir  path/to/dts_original_masks \
      --out-dir     path/to/iaa_output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as pcmask
from scipy import stats
import SimpleITK as sitk
import matplotlib.pyplot as plt

RATERS = ["A1", "A2", "A3", "A4"]
RNG_SEED = 42
N_BOOT = 10_000


# ---------- COCO -> per-image binary mask ----------

def coco_to_masks(coco_path: Path) -> dict[str, np.ndarray]:
    """Return {file_basename: bool mask HxW} for every image in a COCO file."""
    coco = json.load(open(coco_path))
    img_by_id = {im["id"]: im for im in coco["images"]}
    masks: dict[str, np.ndarray] = {}
    for im in coco["images"]:
        basename = Path(im["file_name"]).name
        masks[basename] = np.zeros((im["height"], im["width"]), dtype=bool)
    for ann in coco["annotations"]:
        im = img_by_id[ann["image_id"]]
        h, w = im["height"], im["width"]
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rles = pcmask.frPyObjects(seg, h, w)
            rle = pcmask.merge(rles)
        elif isinstance(seg, dict):
            rle = seg if "counts" in seg and isinstance(seg["counts"], bytes) else pcmask.frPyObjects(seg, h, w)
        else:
            continue
        m = pcmask.decode(rle).astype(bool)
        if m.ndim == 3:
            m = m.any(axis=2)
        basename = Path(im["file_name"]).name
        masks[basename] |= m
    return masks


# ---------- pixel metrics ----------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 1.0


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / pred.sum())


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float(np.logical_and(pred, gt).sum() / gt.sum())


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap (mean, ci_low, ci_high)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def fleiss_kappa_pixels(rater_masks: list[np.ndarray]) -> float:
    """Fleiss' kappa over all pixels with N raters × 2 categories (FG/BG)."""
    n_raters = len(rater_masks)
    fg = np.zeros_like(rater_masks[0], dtype=np.int32)
    for m in rater_masks:
        fg += m.astype(np.int32)
    bg = n_raters - fg
    P_i = (fg * (fg - 1) + bg * (bg - 1)) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_fg = fg.sum() / (fg.size * n_raters)
    p_bg = 1 - p_fg
    P_e = p_fg ** 2 + p_bg ** 2
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def staple_consensus(rater_masks: list[np.ndarray]) -> tuple[np.ndarray, list[float], list[float]]:
    """SimpleITK STAPLE on a list of bool HxW masks."""
    sitk_imgs = [sitk.GetImageFromArray(m.astype(np.uint8)) for m in rater_masks]
    f = sitk.STAPLEImageFilter()
    f.SetForegroundValue(1)
    out = f.Execute(sitk_imgs)
    prob = sitk.GetArrayFromImage(out)
    return prob >= 0.5, list(f.GetSensitivity()), list(f.GetSpecificity())


def n_components(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi
    return int(ndi.label(mask)[1])


def total_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def perimeter(mask: np.ndarray) -> int:
    from scipy import ndimage as ndi
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return int((mask & ~eroded).sum())


def main() -> None:
    p = argparse.ArgumentParser(description="DTS-150 IAA pilot analysis")
    p.add_argument("--raters-root", type=Path, required=True,
                   help="Directory containing a1/, a2/, a3/, a4/ "
                        "each with annotations/instances_default.json (CVAT COCO export)")
    p.add_argument("--dts-gt-dir",  type=Path, required=True,
                   help="Directory with DTS-original binary masks (one PNG per basename)")
    p.add_argument("--out-dir",     type=Path, required=True,
                   help="Where to write iaa_summary.json + figures")
    p.add_argument("--n-images",    type=int, default=150,
                   help="Expected number of images (sanity check)")
    args = p.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "iaa_pairwise_iou.csv"
    out_json = args.out_dir / "iaa_summary.json"
    fig_heatmap = args.out_dir / "iaa_heatmap.pdf"
    fig_ba = args.out_dir / "iaa_bland_altman_area.pdf"

    # Load rater masks
    rater_masks: dict[str, dict[str, np.ndarray]] = {}
    for i, name in enumerate(RATERS, start=1):
        coco_path = args.raters_root / f"a{i}" / "annotations" / "instances_default.json"
        rater_masks[name] = coco_to_masks(coco_path)
        print(f"[load] {name}: {len(rater_masks[name])} images", file=sys.stderr)

    bases = sorted(rater_masks[RATERS[0]].keys())
    for r in RATERS[1:]:
        assert sorted(rater_masks[r].keys()) == bases, f"basename mismatch for {r}"
    assert len(bases) == args.n_images, f"expected {args.n_images} images, got {len(bases)}"

    # Load DTS original GT masks
    dts_gt: dict[str, np.ndarray] = {}
    for b in bases:
        gt_path = args.dts_gt_dir / b
        if not gt_path.exists():
            raise FileNotFoundError(f"missing DTS GT: {gt_path}")
        dts_gt[b] = np.array(Image.open(gt_path).convert("L")) > 127

    # ----- pairwise IoU/Dice/precision/recall -----
    pairs = [(r1, r2) for i, r1 in enumerate(RATERS) for r2 in RATERS[i + 1:]]
    pair_records = []
    for b in bases:
        for r1, r2 in pairs:
            m1, m2 = rater_masks[r1][b], rater_masks[r2][b]
            pair_records.append({
                "image": b, "r1": r1, "r2": r2,
                "iou": iou(m1, m2), "dice": dice(m1, m2),
                "precision": precision(m1, m2), "recall": recall(m1, m2),
            })
    df_pair = pd.DataFrame(pair_records)
    df_pair.to_csv(out_csv, index=False)
    print(f"[write] {out_csv} ({len(df_pair)} rows)", file=sys.stderr)

    mat = np.eye(4)
    for (r1, r2), sub in df_pair.groupby(["r1", "r2"]):
        i, j = RATERS.index(r1), RATERS.index(r2)
        mat[i, j] = mat[j, i] = sub["iou"].mean()
    pair_iou_mean = mat[np.triu_indices(4, k=1)].mean()

    overall_vals = df_pair["iou"].values
    overall_mean, overall_lo, overall_hi = bootstrap_mean_ci(overall_vals, rng)

    # ----- Fleiss kappa -----
    kappas = []
    for b in bases:
        ms = [rater_masks[r][b] for r in RATERS]
        kappas.append(fleiss_kappa_pixels(ms))
    fleiss_mean, fleiss_lo, fleiss_hi = bootstrap_mean_ci(np.array(kappas), rng)

    # ----- STAPLE consensus -----
    consensus_masks: dict[str, np.ndarray] = {}
    sens_per_rater = {r: [] for r in RATERS}
    spec_per_rater = {r: [] for r in RATERS}
    cons_vs_rater_iou = {r: [] for r in RATERS}
    rater_vs_dts_iou = {r: [] for r in RATERS}
    cons_vs_dts_iou: list[float] = []
    cons_area: list[int] = []
    dts_area: list[int] = []
    for b in bases:
        ms = [rater_masks[r][b] for r in RATERS]
        cons, sens, spec = staple_consensus(ms)
        consensus_masks[b] = cons
        for r, s, p_ in zip(RATERS, sens, spec):
            sens_per_rater[r].append(s)
            spec_per_rater[r].append(p_)
            cons_vs_rater_iou[r].append(iou(cons, rater_masks[r][b]))
            rater_vs_dts_iou[r].append(iou(rater_masks[r][b], dts_gt[b]))
        cons_vs_dts_iou.append(iou(cons, dts_gt[b]))
        cons_area.append(total_area(cons))
        dts_area.append(total_area(dts_gt[b]))

    sens_summary = {r: bootstrap_mean_ci(np.array(sens_per_rater[r]), rng) for r in RATERS}
    spec_summary = {r: bootstrap_mean_ci(np.array(spec_per_rater[r]), rng) for r in RATERS}
    cons_vs_rater_summary = {r: bootstrap_mean_ci(np.array(cons_vs_rater_iou[r]), rng) for r in RATERS}
    cons_vs_dts_summary = bootstrap_mean_ci(np.array(cons_vs_dts_iou), rng)
    rater_vs_dts_summary = {r: bootstrap_mean_ci(np.array(rater_vs_dts_iou[r]), rng) for r in RATERS}
    rater_vs_dts_mean_only = {r: float(np.mean(rater_vs_dts_iou[r])) for r in RATERS}

    # ----- policy gap -----
    cons_vs_dts_arr = np.array(cons_vs_dts_iou)
    per_img_pair_iou = df_pair.groupby("image")["iou"].mean().reindex(bases).values
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(per_img_pair_iou, cons_vs_dts_arr)
    delta = per_img_pair_iou - cons_vs_dts_arr
    policy_gap = float((1 - cons_vs_dts_arr).mean())
    within_var = float((1 - per_img_pair_iou).mean())
    fold_ratio = policy_gap / within_var if within_var > 0 else float("inf")

    # ----- object-level count agreement (Cohen kappa pairwise) -----
    from sklearn.metrics import cohen_kappa_score
    counts = {r: np.array([n_components(rater_masks[r][b]) for b in bases]) for r in RATERS}
    counts["DTS"] = np.array([n_components(dts_gt[b]) for b in bases])
    pair_kappa = {}
    for i, r1 in enumerate(list(counts.keys())[:-1]):
        for r2 in list(counts.keys())[i + 1:]:
            pair_kappa[f"{r1}_vs_{r2}"] = float(cohen_kappa_score(counts[r1], counts[r2]))

    # ----- Bland-Altman -----
    cons_arr = np.array(cons_area, dtype=float)
    dts_arr = np.array(dts_area, dtype=float)
    diff_area = cons_arr - dts_arr
    mean_area = (cons_arr + dts_arr) / 2
    ba_bias = float(diff_area.mean())
    ba_loa_lo = float(ba_bias - 1.96 * diff_area.std())
    ba_loa_hi = float(ba_bias + 1.96 * diff_area.std())

    summary = {
        "n_images": len(bases),
        "raters": RATERS,
        "pairwise_iou": {
            "matrix": mat.tolist(),
            "mean": float(pair_iou_mean),
            "overall_mean_ci": [overall_mean, overall_lo, overall_hi],
        },
        "fleiss_kappa": {
            "per_image_mean": fleiss_mean,
            "ci": [fleiss_lo, fleiss_hi],
        },
        "staple": {
            "sensitivity_per_rater": {r: list(v) for r, v in sens_summary.items()},
            "specificity_per_rater": {r: list(v) for r, v in spec_summary.items()},
            "consensus_vs_rater_iou": {r: list(v) for r, v in cons_vs_rater_summary.items()},
            "consensus_vs_dts_iou": list(cons_vs_dts_summary),
        },
        "per_rater_vs_dts_iou": {
            "mean_per_rater": rater_vs_dts_mean_only,
            "bootstrap_ci_per_rater": {r: list(v) for r, v in rater_vs_dts_summary.items()},
        },
        "policy_gap": {
            "consensus_vs_dts_mean_iou": float(cons_vs_dts_arr.mean()),
            "within_policy_mean_pairwise_iou": float(per_img_pair_iou.mean()),
            "policy_gap_pp": (1 - cons_vs_dts_arr.mean()) * 100,
            "within_policy_var_pp": (1 - per_img_pair_iou.mean()) * 100,
            "fold_ratio_gap_over_variance": fold_ratio,
            "paired_wilcoxon_stat": float(wilcoxon_stat),
            "paired_wilcoxon_p": float(wilcoxon_p),
            "delta_median": float(np.median(delta)),
            "delta_iqr": [float(np.percentile(delta, 25)), float(np.percentile(delta, 75))],
        },
        "object_count_kappa": pair_kappa,
        "bland_altman_area": {
            "bias_px": ba_bias,
            "loa_lo_px": ba_loa_lo,
            "loa_hi_px": ba_loa_hi,
        },
        "config": {"seed": RNG_SEED, "n_boot": N_BOOT},
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[write] {out_json}", file=sys.stderr)

    # ----- 5×5 mean IoU heatmap (raters + DTS) -----
    all_keys = RATERS + ["DTS"]
    big = np.eye(5)
    big[:4, :4] = mat
    for i, r in enumerate(RATERS):
        v = float(np.mean(rater_vs_dts_iou[r]))
        big[i, 4] = big[4, i] = v
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(big, vmin=0.6, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(all_keys)
    ax.set_yticklabels(all_keys)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{big[i, j]:.3f}", ha="center", va="center",
                    color="white" if big[i, j] < 0.85 else "black", fontsize=8)
    ax.set_title("Pairwise mean IoU (4 annotators + DTS-original)")
    fig.colorbar(im, ax=ax, label="IoU")
    fig.tight_layout()
    fig.savefig(fig_heatmap)
    plt.close(fig)
    print(f"[write] {fig_heatmap}", file=sys.stderr)

    # ----- Bland-Altman -----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mean_area, diff_area, s=10, alpha=0.6, color="C0")
    ax.axhline(ba_bias, color="black", linewidth=1, label=f"bias = {ba_bias:.0f}")
    ax.axhline(ba_loa_lo, color="red", linestyle="--", linewidth=0.8,
               label=f"LoA = [{ba_loa_lo:.0f}, {ba_loa_hi:.0f}]")
    ax.axhline(ba_loa_hi, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="grey", linewidth=0.4)
    ax.set_xlabel("Mean of consensus and DTS-original area (px)")
    ax.set_ylabel("Consensus − DTS-original (px)")
    ax.set_title("Bland–Altman: consensus vs DTS-original spheroid area")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_ba)
    plt.close(fig)
    print(f"[write] {fig_ba}", file=sys.stderr)

    print("\n=== IAA summary (n=%d, 4 raters, DTS-original as core-only reference) ===" % len(bases))
    print(f"Pairwise mean IoU (within-policy): {overall_mean:.4f} [{overall_lo:.4f}, {overall_hi:.4f}]")
    print(f"Fleiss kappa (per-image mean):     {fleiss_mean:.4f} [{fleiss_lo:.4f}, {fleiss_hi:.4f}]")
    print(f"Consensus vs DTS-original IoU:     {cons_vs_dts_summary[0]:.4f} "
          f"[{cons_vs_dts_summary[1]:.4f}, {cons_vs_dts_summary[2]:.4f}]")
    print(f"Policy gap / within-policy var:    {fold_ratio:.2f}×")
    print(f"Paired Wilcoxon (within vs vs-DTS): p={wilcoxon_p:.2e}")


if __name__ == "__main__":
    main()
