#!/usr/bin/env python
"""
Convert HTS-Seg original_images/ into a 1024x1024 SpheroMix-style test split
that the existing CachedSpheroidDataset and evaluate_a3.py can consume directly.

Source layout (from Zenodo 14161434):
  HTS_Seg/original_images/
    images/{control,treatment}/day N/Image__YYYY-MM-DD__HH-MM-SS.bmp   (2160x3840 RGB uint8)
    labels/{control,treatment}/day N/Image__YYYY-MM-DD__HH-MM-SS.tif   (2160x3840 binary uint8 0/1)

Output layout:
  <out-dir>/test/
    images/htsseg_<class>_dayN_<basename>.png      (1024x1024 RGB)
    masks/<same>.png                               (1024x1024 binary 0/255)

Notes:
- 1 BMP is unlabeled (97 images vs 96 labels) — that one is skipped, with a warning.
- Resize uses Lanczos for the image, NEAREST for the binary label (preserves edges).
- Filenames are sanitized (spaces→underscores, no ":") so the dataset class accepts them.

Usage:
    python scripts/a3/prepare_htsseg_eval.py \
        --src /path/to/HTS_Seg/original_images \
        --out /path/to/HTS_Seg_eval
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TARGET = 1024


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace(":", "-")


def main():
    ap = argparse.ArgumentParser(description="Convert HTS-Seg native images to 1024x1024 SpheroMix layout")
    ap.add_argument("--src", type=Path, required=True,
                    help="HTS_Seg/original_images root with images/ and labels/ subdirs")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root; will create <out>/test/{images,masks}/")
    args = ap.parse_args()

    dst = args.out / "test"
    dst_img = dst / "images"
    dst_msk = dst / "masks"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_msk.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_mask = 0
    total_imgs = 0

    for img_path in sorted(args.src.glob("images/*/*/*.bmp")):
        total_imgs += 1
        rel = img_path.relative_to(args.src / "images")
        cls = rel.parts[0]
        day = rel.parts[1]
        base = img_path.stem
        mask_path = args.src / "labels" / cls / day / (base + ".tif")
        if not mask_path.exists():
            print(f"[skip] no mask: {rel}")
            skipped_no_mask += 1
            continue

        out_name = f"htsseg_{cls}_{sanitize(day)}_{base}.png"
        out_img = dst_img / out_name
        out_msk = dst_msk / out_name

        im = Image.open(img_path).convert("RGB")
        im_r = im.resize((TARGET, TARGET), Image.LANCZOS)
        im_r.save(out_img, optimize=True)

        mk = np.array(Image.open(mask_path))
        mk_pil = Image.fromarray(mk.astype(np.uint8), mode="L")
        mk_r = np.array(mk_pil.resize((TARGET, TARGET), Image.NEAREST))
        mk_r = np.where(mk_r > 0, 255, 0).astype(np.uint8)
        Image.fromarray(mk_r, mode="L").save(out_msk, optimize=True)

        written += 1
        if written % 20 == 0:
            print(f"[hts] {written:3d}/{total_imgs} processed; sample {out_name}")

    print(f"\n[hts] total source images: {total_imgs}")
    print(f"[hts] written pairs:        {written}")
    print(f"[hts] skipped (no mask):    {skipped_no_mask}")
    print(f"[hts] target dir:           {dst}")


if __name__ == "__main__":
    main()
