"""
Generate Sauvola pre-annotations for DTS test images using prusek-spheroid pipeline.

Replicates the SpheroHQ annotation pipeline (Sauvola + manual correction in CVAT)
on the DTS test subset, so 4 annotators can independently re-annotate via the same
process. This isolates whether the SpheroHQ↔DTS IoU gap is policy-driven.

Usage:
    python generate_sauvola_dts.py [--params PARAMS_JSON] [--out OUT_DIR] [--limit N]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
# `prusek_spheroid` is an optional dep imported lazily inside run_sauvola()
# so that --help and test imports work without it installed.

# prusek-spheroid GradientDescentGUI seed defaults — same starting point as
# what was used for SpheroHQ pre-annotation in the GUI workflow.
DEFAULT_PARAMS = {
    "window_size": 51,    # odd, smaller than 800 default to actually resolve cell boundaries on 1024x1024
    "std_k": 0.5,
    "min_area": 0.005,    # 0.5% of image area = ~5243 px on 1024x1024
    "sigma": 1.0,
    "dilation_size": 2,
}

def is_dts(filename: str) -> bool:
    """DTS images are numerically named (1.png, 100.png); BxPC-3 are bxpc-3_*.png"""
    return not filename.startswith("bxpc-3_")


def run_sauvola(img_path: Path, params: dict) -> np.ndarray:
    """Load image, run prusek-spheroid Sauvola pipeline, return binary mask uint8 (0/255)."""
    from prusek_spheroid.methods import BaseImageProcessing as Methods

    img = Image.open(img_path)
    if img.mode != "L":
        img = img.convert("L")
    img_gray = np.array(img, dtype=np.uint8)

    # methods.Methods.sauvola is a @staticmethod returning (result_mask 0/1, None)
    result_mask, _ = Methods.sauvola(params, img_gray, inner_contours=False)
    return (result_mask * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default=None,
                    help="Optional JSON file with Sauvola params; defaults to prusek-spheroid GUI seed")
    ap.add_argument("--input-dir", type=Path, default=None,
                    help="Directory of DTS images (default: $SPHEROMIX_PATH/test/images)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for masks + overlays + manifest")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N DTS images (for testing)")
    a = ap.parse_args()

    if a.input_dir is None:
        spheromix = os.environ.get("SPHEROMIX_PATH")
        if not spheromix:
            ap.error("--input-dir required (or set SPHEROMIX_PATH and we use $SPHEROMIX_PATH/test/images)")
        a.input_dir = Path(spheromix) / "test" / "images"
    if not a.input_dir.exists():
        ap.error(f"input dir does not exist: {a.input_dir}")

    if a.params:
        with open(a.params) as f:
            params = json.load(f)
    else:
        params = DEFAULT_PARAMS

    out_dir = Path(a.out)
    out_masks = out_dir / "masks"
    out_overlays = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)

    # save params used for reproducibility
    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    dts_images = sorted([f for f in os.listdir(a.input_dir) if is_dts(f)])
    if a.limit:
        dts_images = dts_images[:a.limit]
    print(f"DTS images to process: {len(dts_images)}")
    print(f"Output: {out_dir}")
    print(f"Params: {params}")

    manifest = []
    for i, fn in enumerate(dts_images):
        img_path = a.input_dir / fn
        try:
            mask = run_sauvola(img_path, params)
            n_fg = int((mask > 0).sum())
            frac_fg = n_fg / mask.size

            # save mask
            out_mask_path = out_masks / fn
            Image.fromarray(mask).save(out_mask_path)

            # save overlay (grayscale image + red-tinted mask) for QC
            img_orig = np.array(Image.open(img_path).convert("RGB"))
            overlay = img_orig.copy()
            mask_bool = mask > 0
            overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 80, 0, 255)
            overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1].astype(int) - 30, 0, 255)
            overlay[mask_bool, 2] = np.clip(overlay[mask_bool, 2].astype(int) - 30, 0, 255)
            Image.fromarray(overlay).save(out_overlays / fn)

            manifest.append({"filename": fn, "n_fg_pixels": n_fg, "fraction_fg": round(frac_fg, 4)})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(dts_images)}  last={fn}  fg={frac_fg:.3f}")
        except Exception as e:
            print(f"FAIL on {fn}: {e}", file=sys.stderr)
            manifest.append({"filename": fn, "error": str(e)})

    # save manifest
    import csv
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        keys = sorted({k for m in manifest for k in m})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in manifest:
            w.writerow(m)

    # summary
    fracs = [m["fraction_fg"] for m in manifest if "fraction_fg" in m]
    if fracs:
        print(f"\nDone. {len(fracs)}/{len(dts_images)} processed successfully.")
        print(f"Foreground fraction: mean={np.mean(fracs):.3f}, median={np.median(fracs):.3f}, "
              f"min={np.min(fracs):.3f}, max={np.max(fracs):.3f}")


if __name__ == "__main__":
    main()
