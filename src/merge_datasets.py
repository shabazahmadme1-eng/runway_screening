#!/usr/bin/env python3
"""Step 2 — Dataset merging.

Downloads the Kaggle RDD2022 road-damage dataset, converts its PASCAL VOC
XML annotations to YOLO format under the runway class schema, and merges
the result with the local drone frames into a single master YOLO training
directory.

Class schema (0-2 from the original spec, 3-11 from the client annotation
Q&A — see docs/annotation_guide.md):
    0  crack                 <- RDD2022 D00/D10/D20 + drone
    1  spalling              <- RDD2022 D40 (pothole proxy) + drone
    2  fod                   <- drone only
    3  faded_paint_marking   <- RDD2022 D43/D44 (blurred markings) + drone
    4  band_joint            <- drone only
    5  gap_vegetation        <- drone only
    6  aged_surface          <- drone only
    7  repair_patch          <- RDD2022 "Repair" + drone
    8  weathered_surface     <- drone only
    9  surface_discoloration <- drone only
    10 paint_marking         <- drone only
    11 faded_surface_marking <- drone only

Drone frames without labels are staged under <out>/staging/drone_frames for
the pseudo-labelling step; once labels are verified, re-run with
--drone-labels to fold them into the train/val split.
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

CLASS_NAMES = [
    "crack",
    "spalling",
    "fod",
    "faded_paint_marking",
    "band_joint",
    "gap_vegetation",
    "aged_surface",
    "repair_patch",
    "weathered_surface",
    "surface_discoloration",
    "paint_marking",
    "faded_surface_marking",
]

# RDD2022 damage code -> runway class id. Unlisted codes (D50 manhole,
# "Block crack", ...) are dropped.
RDD_CLASS_MAP = {
    "D00": 0,
    "D10": 0,
    "D20": 0,
    "D40": 1,
    "D43": 3,
    "D44": 3,
    "Repair": 7,
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kaggle-dataset", default="codderboy/rdd2022",
                   help="Kaggle dataset slug to download (default: %(default)s)")
    p.add_argument("--rdd-dir", type=Path, default=Path("data/rdd2022"),
                   help="where to download/extract RDD2022; if it already "
                        "contains data the download is skipped")
    p.add_argument("--drone-frames", type=Path, required=True,
                   help="directory of frames extracted from the drone video")
    p.add_argument("--drone-labels", type=Path, default=None,
                   help="directory of verified YOLO labels for the drone "
                        "frames (after the refinement step)")
    p.add_argument("--out", type=Path, default=Path("datasets/master"),
                   help="master YOLO dataset directory (default: %(default)s)")
    p.add_argument("--countries", nargs="*", default=None,
                   help="restrict RDD2022 to these country folders, e.g. "
                        "Japan Norway United_States (default: all)")
    p.add_argument("--max-rdd-images", type=int, default=None,
                   help="cap on RDD2022 images kept (random sample)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--keep-negatives", action="store_true",
                   help="keep RDD images whose boxes were all dropped by the "
                        "class mapping (as background negatives)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_rdd(slug: str, dest: Path) -> None:
    """Pull the dataset via the Kaggle API unless it is already present."""
    if dest.exists() and any(dest.rglob("*.xml")):
        print(f"[skip] RDD2022 already present in {dest}")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:  # kaggle raises at import time when creds missing
        sys.exit(f"Kaggle credentials not found ({e}). Place kaggle.json in "
                 "~/.kaggle/ or export KAGGLE_USERNAME / KAGGLE_KEY.")
    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] kaggle dataset {slug} -> {dest}")
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)


def voc_to_yolo(xml_path: Path):
    """Parse one VOC XML file -> list of 'cls cx cy w h' lines (normalised).

    Returns None when the file is unparseable or lacks image dimensions.
    """
    try:
        root = ET.parse(xml_path).getroot()
        w = float(root.findtext("size/width"))
        h = float(root.findtext("size/height"))
    except (ET.ParseError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    lines = []
    for obj in root.iter("object"):
        cls = RDD_CLASS_MAP.get((obj.findtext("name") or "").strip())
        if cls is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        # clamp to image bounds; RDD2022 has a few out-of-range boxes
        xmin, xmax = max(0.0, xmin), min(w, xmax)
        ymin, ymax = max(0.0, ymin), min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def collect_rdd_pairs(rdd_dir: Path, countries):
    """Find (image, xml) pairs in the extracted archive, any layout.

    The official layout is RDD2022/<Country>/train/{images,annotations/xmls},
    but Kaggle mirrors vary, so we index every XML by stem and match images
    against it. Test splits ship without XMLs and fall out naturally.
    """
    xml_index = {p.stem: p for p in rdd_dir.rglob("*.xml")}
    pairs = []
    for img in rdd_dir.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        xml = xml_index.get(img.stem)
        if xml is None:
            continue
        if countries and not any(c.lower() in str(img).lower() for c in countries):
            continue
        pairs.append((img, xml))
    return pairs


def safe_copy(src: Path, dst: Path):
    if not dst.exists():
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    random.seed(args.seed)

    out = args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    staging = out / "staging" / "drone_frames"

    # ------------------------------------------------------------------ RDD
    download_rdd(args.kaggle_dataset, args.rdd_dir)
    pairs = collect_rdd_pairs(args.rdd_dir, args.countries)
    if not pairs:
        sys.exit(f"No annotated RDD2022 images found under {args.rdd_dir} — "
                 "check the download/extraction.")
    print(f"[rdd] {len(pairs)} annotated images found")

    random.shuffle(pairs)
    if args.max_rdd_images:
        pairs = pairs[: args.max_rdd_images]

    kept, dropped, box_count = 0, 0, 0
    samples = []  # (img_path, label_lines, basename)
    for img, xml in tqdm(pairs, desc="converting VOC->YOLO"):
        lines = voc_to_yolo(xml)
        if lines is None or (not lines and not args.keep_negatives):
            dropped += 1
            continue
        samples.append((img, lines, f"rdd_{img.stem}{img.suffix.lower()}"))
        kept += 1
        box_count += len(lines)
    print(f"[rdd] kept {kept} images ({box_count} boxes), dropped {dropped} "
          "(unmapped classes / bad XML)")

    # ---------------------------------------------------------------- drone
    drone_imgs = sorted(p for p in args.drone_frames.rglob("*")
                        if p.suffix.lower() in IMG_EXTS)
    if not drone_imgs:
        print(f"[warn] no drone frames found in {args.drone_frames}")

    n_drone_labelled = 0
    for img in drone_imgs:
        base = f"drone_{img.stem}{img.suffix.lower()}"
        label = (args.drone_labels / f"{img.stem}.txt") if args.drone_labels else None
        if label and label.exists():
            samples.append((img, label.read_text().splitlines(), base))
            n_drone_labelled += 1
            (staging / base).unlink(missing_ok=True)  # no longer awaiting labels
        else:
            staging.mkdir(parents=True, exist_ok=True)
            safe_copy(img, staging / base)
    n_staged = len(drone_imgs) - n_drone_labelled
    print(f"[drone] {n_drone_labelled} labelled frames merged, "
          f"{n_staged} unlabelled frames staged in {staging}")

    # ----------------------------------------------------------- split+copy
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_fraction)
    for i, (img, lines, base) in enumerate(tqdm(samples, desc="writing master")):
        split = "val" if i < n_val else "train"
        safe_copy(img, out / "images" / split / base)
        (out / "labels" / split / Path(base).with_suffix(".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n")
    print(f"[done] {len(samples) - n_val} train / {n_val} val images -> {out}")
    print(f"[done] data config written to {yaml_path}")


if __name__ == "__main__":
    main()
