"""
Build a CVAT-uploadable package from Sauvola DTS pre-annotations.

Output (under --out-dir):
  images/         — N random DTS images (seed=42)
  annotations.xml — CVAT for Images 1.1 with polygons (one per spheroid)
  labels.txt      — label list

Upload instructions: in CVAT, create a task with images/ contents,
then 'Upload annotations' → 'CVAT 1.1' → annotations.xml.

Usage:
    python scripts/a3/build_cvat_package.py \
        --dts-test-dir   $SPHEROMIX_PATH/test/images \
        --sauvola-dir    path/to/sauvola_dts_preannot \
        --out-dir        path/to/cvat_dts_sauvola_150 \
        [--n 150] [--seed 42]
"""
import argparse, os, random, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
from xml.dom import minidom

LABEL = "spheroid"
# polygon simplification — fewer points = lighter CVAT, but worse fidelity
APPROX_EPS_FRAC = 0.001  # 0.1% of contour perimeter; CVAT handles ~50-200 pt polygons fine


def is_dts(fn):
    return not fn.startswith("bxpc-3_")


def mask_to_polygons(mask_path, min_area_px=200):
    """Binary mask PNG → list of (Nx2) polygon point arrays. External contours only."""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask_bin = (mask > 127).astype(np.uint8)
    contours, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv.contourArea(c) < min_area_px:
            continue


        pts = c.reshape(-1, 2)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def build_xml(image_records):
    """Build CVAT for Images 1.1 XML."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = "DTS_Sauvola_preannot_150"
    ET.SubElement(task, "size").text = str(len(image_records))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = LABEL
    ET.SubElement(label, "color").text = "#ff0000"
    ET.SubElement(label, "attributes")

    for i, rec in enumerate(image_records):
        img_el = ET.SubElement(root, "image",
                               id=str(i),
                               name=rec["filename"],
                               width=str(rec["width"]),
                               height=str(rec["height"]))
        for poly in rec["polygons"]:
            pts_str = ";".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly)
            ET.SubElement(img_el, "polygon",
                          label=LABEL,
                          source="manual",
                          occluded="0",
                          points=pts_str,
                          z_order="0")
    return root


def prettify(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build CVAT package from Sauvola DTS pre-annotations")
    ap.add_argument("--dts-test-dir", type=Path, default=None,
                    help="Directory of DTS test images (default: $SPHEROMIX_PATH/test/images)")
    ap.add_argument("--sauvola-dir",  type=Path, required=True,
                    help="Output of generate_sauvola_dts.py (must contain masks/ subdir)")
    ap.add_argument("--out-dir",      type=Path, required=True,
                    help="Where to write the CVAT package")
    ap.add_argument("--n",            type=int, default=150,
                    help="Number of images to sample (default: 150)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    if args.dts_test_dir is None:
        sm = os.environ.get("SPHEROMIX_PATH")
        if not sm:
            ap.error("--dts-test-dir required (or set SPHEROMIX_PATH)")
        args.dts_test_dir = Path(sm) / "test" / "images"

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    (args.out_dir / "images").mkdir(parents=True)

    # 1) random sample
    all_dts = sorted([f for f in os.listdir(args.dts_test_dir) if is_dts(f)])
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(all_dts, args.n))
    print(f"sampled {len(sample)} of {len(all_dts)} DTS images (seed={args.seed})")

    # 2) copy images + build polygons
    records = []
    for fn in sample:
        src_img = args.dts_test_dir / fn
        dst_img = args.out_dir / "images" / fn
        shutil.copy2(src_img, dst_img)

        mask_path = args.sauvola_dir / "masks" / fn
        if not mask_path.exists():
            print(f"  WARN: no mask for {fn}", file=sys.stderr)
            polys = []
        else:
            polys = mask_to_polygons(mask_path)

        with Image.open(src_img) as im:
            w, h = im.size

        records.append({"filename": fn, "width": w, "height": h, "polygons": polys})

    # 3) write annotations.xml
    xml_root = build_xml(records)
    xml_bytes = prettify(xml_root)
    (args.out_dir / "annotations.xml").write_bytes(xml_bytes)
    (args.out_dir / "labels.txt").write_text(f"{LABEL}\n")

    n_polys = sum(len(r["polygons"]) for r in records)
    avg_pts = sum(sum(len(p) for p in r["polygons"]) for r in records) / max(1, n_polys)
    print(f"\ntotal images: {len(records)}")
    print(f"total polygons: {n_polys}  (avg {n_polys/len(records):.1f} per image)")
    print(f"avg points per polygon: {avg_pts:.1f}")
    print(f"output: {args.out_dir}")
    print(f"  images/        ({len(records)} files)")
    print(f"  annotations.xml ({len(xml_bytes)/1024:.1f} KB)")
    print(f"  labels.txt")


if __name__ == "__main__":
    main()
