#!/usr/bin/env python
"""
Prepare HTS-Seg tile variants for direct comparability with the HTS-Seg paper
(Tahmasbi 2026, Jaccard 91.17 with DeepLab).

Both tile sets are kept at NATIVE 512×512 — same resolution as the paper's
benchmark — and converted into the SpheroMix test-split layout.

Outputs (under --out-root):
  HTS_Seg_eval_tiles/test/    {images,masks}/<int>_<class>.png   (~2971 pairs)
  HTS_Seg_eval_cleaned/test/  {images,masks}/<int>_<class>.png   (~2180 pairs)

Source layout (under --src-root):
  HTS_Seg/{tiled_images,cleaned_tiled_images}/
    images/{control,treatment}/<N>.bmp     (512x512 RGB)
    labels/{control,treatment}/<N>.tif     (512x512 binary uint8 0/1)

Usage:
    python scripts/a3/prepare_htsseg_tiles.py \
        --src-root /path/to/HTS_Seg \
        --out-root /path/to/output
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert(src_root: Path, src_dir_name: str, dst_root: Path, label_name: str) -> tuple[int, int, int]:
    src_dir = src_root / src_dir_name
    dst_img = dst_root / "test" / "images"
    dst_msk = dst_root / "test" / "masks"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_msk.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    bad = 0

    for img_path in sorted(src_dir.glob("images/*/*.bmp")):
        cls = img_path.parent.name
        n = img_path.stem
        msk_path = src_dir / "labels" / cls / f"{n}.tif"
        if not msk_path.exists():
            skipped += 1
            continue

        out_name = f"{n.zfill(4)}_{cls}.png"
        out_img = dst_img / out_name
        out_msk = dst_msk / out_name

        try:
            im = Image.open(img_path).convert("RGB")
            mk = np.array(Image.open(msk_path))
            if im.size != (512, 512) or mk.shape != (512, 512):
                bad += 1
                continue

            im.save(out_img, optimize=True)
            mk_bin = np.where(mk > 0, 255, 0).astype(np.uint8)
            Image.fromarray(mk_bin, mode="L").save(out_msk, optimize=True)
            written += 1
        except Exception as e:
            print(f"[err] {img_path}: {e!r}")
            bad += 1

        if written % 500 == 0 and written > 0:
            print(f"  [{label_name}] {written} pairs written so far ({cls}/{n})")

    return written, skipped, bad


def main():
    ap = argparse.ArgumentParser(description="Convert HTS-Seg native-512 tiles into SpheroMix layout")
    ap.add_argument("--src-root", type=Path, required=True,
                    help="HTS_Seg/ root containing tiled_images/ and cleaned_tiled_images/")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Where to write HTS_Seg_eval_tiles/ and HTS_Seg_eval_cleaned/")
    args = ap.parse_args()

    print("=== HTS_Seg tiled_images (raw, ~2971) → 512 native ===")
    w, s, b = convert(args.src_root, "tiled_images",
                      args.out_root / "HTS_Seg_eval_tiles", "tiles")
    print(f"  written={w}, skipped(no_mask)={s}, bad={b}")

    print("\n=== HTS_Seg cleaned_tiled_images (cleaned, ~2180) → 512 native ===")
    w2, s2, b2 = convert(args.src_root, "cleaned_tiled_images",
                         args.out_root / "HTS_Seg_eval_cleaned", "cleaned")
    print(f"  written={w2}, skipped(no_mask)={s2}, bad={b2}")


if __name__ == "__main__":
    main()
