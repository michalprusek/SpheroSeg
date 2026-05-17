#!/usr/bin/env python
"""
Convert HTS-Seg original_images/ into a 1024x1024 SpheroMix-style test split
that the existing CachedSpheroidDataset and evaluate_a3.py can consume directly.

Source layout (from Zenodo 14161434):
  HTS_Seg/original_images/
    images/{control,treatment}/day N/Image__YYYY-MM-DD__HH-MM-SS.bmp   (2160x3840 RGB uint8)
    labels/{control,treatment}/day N/Image__YYYY-MM-DD__HH-MM-SS.tif   (2160x3840 binary uint8 0/1)

Output layout:
  HTS_Seg_eval/test/
    images/htsseg_<class>_dayN_<basename>.png      (1024x1024 RGB)
    masks/<same>.png                               (1024x1024 binary 0/255)

Notes:
- 1 BMP is unlabeled (97 images vs 96 labels) — that one is skipped, with a warning.
- Resize uses Lanczos for the image, NEAREST for the binary label (preserves edges).
- Filenames are sanitized (spaces→underscores, no ":") so the dataset class accepts them.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SRC = Path("/disk1/prusek/SpheroSeg/data/HTS_Seg/original_images")
DST = Path("/disk1/prusek/SpheroSeg/data/HTS_Seg_eval/test")
DST_IMG = DST / "images"
DST_MSK = DST / "masks"
TARGET = 1024

DST_IMG.mkdir(parents=True, exist_ok=True)
DST_MSK.mkdir(parents=True, exist_ok=True)

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace(":", "-")

written = 0
skipped_no_mask = 0
total_imgs = 0

for img_path in sorted(SRC.glob("images/*/*/*.bmp")):
    total_imgs += 1
    rel = img_path.relative_to(SRC / "images")     # e.g. control/day 1/Image__....bmp
    cls   = rel.parts[0]                            # control / treatment
    day   = rel.parts[1]                            # day 1 ...
    base  = img_path.stem                           # Image__2023-05-01__11-15-49
    mask_path = SRC / "labels" / cls / day / (base + ".tif")
    if not mask_path.exists():
        print(f"[skip] no mask: {rel}")
        skipped_no_mask += 1
        continue
    out_name = f"htsseg_{cls}_{sanitize(day)}_{base}.png"
    out_img  = DST_IMG / out_name
    out_msk  = DST_MSK / out_name

    # Image: BMP RGB → resize to 1024 with Lanczos
    im = Image.open(img_path).convert("RGB")
    im_r = im.resize((TARGET, TARGET), Image.LANCZOS)
    im_r.save(out_img, optimize=True)

    # Mask: 0/1 → resize NEAREST → ×255
    mk = np.array(Image.open(mask_path))
    mk_pil = Image.fromarray(mk.astype(np.uint8), mode="L")
    mk_r = np.array(mk_pil.resize((TARGET, TARGET), Image.NEAREST))
    mk_r = np.where(mk_r > 0, 255, 0).astype(np.uint8)
    Image.fromarray(mk_r, mode="L").save(out_msk, optimize=True)

    written += 1
    if written % 20 == 0:
        print(f"[hts] {written:3d}/{total_imgs} processed; sample {out_name}")

print(f"\n[hts] total source images: {total_imgs}")
print(f"[hts] written pairs: {written}")
print(f"[hts] skipped (no matching mask): {skipped_no_mask}")
print(f"[hts] target dir: {DST}")
