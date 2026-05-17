"""
Build a CVAT-uploadable package from Sauvola DTS pre-annotations.

Output:
  cvat_dts_sauvola_150/
    images/         — 150 random DTS images (seed=42)
    annotations.xml — CVAT for Images 1.1 with polygons (one per spheroid)
    labels.txt      — label list

Upload instructions: in CVAT, create a task with images/ contents,
then 'Upload annotations' → 'CVAT 1.1' → annotations.xml.
"""
import os, random, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
from xml.dom import minidom

DTS_TEST_DIR = Path("/disk1/prusek/SpheroSeg/data/SpheroMix/test/images")
SAUVOLA_DIR = Path("/disk1/prusek/SpheroSeg/eval_a3/sauvola_dts_preannot")
OUT_DIR = Path("/disk1/prusek/SpheroSeg/eval_a3/cvat_dts_sauvola_150")
SEED = 42
N = 150
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
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images").mkdir(parents=True)

    # 1) random sample
    all_dts = sorted([f for f in os.listdir(DTS_TEST_DIR) if is_dts(f)])
    rng = random.Random(SEED)
    sample = sorted(rng.sample(all_dts, N))
    print(f"sampled {len(sample)} of {len(all_dts)} DTS images (seed={SEED})")

    # 2) copy images + build polygons
    records = []
    for fn in sample:
        src_img = DTS_TEST_DIR / fn
        dst_img = OUT_DIR / "images" / fn
        shutil.copy2(src_img, dst_img)

        mask_path = SAUVOLA_DIR / "masks" / fn
        if not mask_path.exists():
            print(f"  WARN: no mask for {fn}", file=sys.stderr)
            polys = []
        else:
            polys = mask_to_polygons(mask_path)

        # get image dims
        with Image.open(src_img) as im:
            w, h = im.size

        records.append({"filename": fn, "width": w, "height": h, "polygons": polys})

    # 3) write annotations.xml
    xml_root = build_xml(records)
    xml_bytes = prettify(xml_root)
    (OUT_DIR / "annotations.xml").write_bytes(xml_bytes)

    # 4) labels.txt
    (OUT_DIR / "labels.txt").write_text(f"{LABEL}\n")

    # 5) sample manifest
    n_polys = sum(len(r["polygons"]) for r in records)
    avg_pts = sum(sum(len(p) for p in r["polygons"]) for r in records) / max(1, n_polys)
    print(f"\ntotal images: {len(records)}")
    print(f"total polygons: {n_polys}  (avg {n_polys/len(records):.1f} per image)")
    print(f"avg points per polygon: {avg_pts:.1f}")
    print(f"output: {OUT_DIR}")
    print(f"  images/        ({len(records)} files)")
    print(f"  annotations.xml ({len(xml_bytes)/1024:.1f} KB)")
    print(f"  labels.txt")


if __name__ == "__main__":
    main()
