#!/usr/bin/env python3
"""Step 4 helper — package frames + pseudo-labels for an annotation tool.

Builds an upload-ready dataset so the pseudo-labels can be opened in CVAT or
Roboflow for the manual refinement pass (delete false positives, add the
drone-only classes the nano cannot detect: band_joint, gap_vegetation,
aged_surface, etc.).

Two layouts:
  --format cvat      CVAT "YOLO 1.1" import zip: obj.names / obj.data /
                     train.txt / obj_train_data/{*.jpg,*.txt}
  --format roboflow  flat folder: images/ + labels/ + data.yaml

ALL frames are included, even the 16 with no pseudo-boxes — they still need
a human pass for the drone-only classes. Frames with no label get an empty
.txt so the annotation tool treats them as reviewed-but-empty, not missing.

The output dir/zip is large (full-res frames) and is git-ignored; build it
on demand and upload it, don't commit it.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

CLASS_NAMES = [
    "crack", "spalling", "fod", "faded_paint_marking", "band_joint",
    "gap_vegetation", "aged_surface", "repair_patch", "weathered_surface",
    "surface_discoloration", "paint_marking", "faded_surface_marking",
]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def label_for(frame: Path, labels: Path) -> str:
    lbl = labels / frame.with_suffix(".txt").name
    return lbl.read_text() if lbl.exists() else ""


def build_cvat(frames, labels, out: Path):
    data = out / "obj_train_data"
    data.mkdir(parents=True, exist_ok=True)
    rel = []
    for f in frames:
        shutil.copy2(f, data / f.name)
        (data / f.with_suffix(".txt").name).write_text(label_for(f, labels))
        rel.append(f"obj_train_data/{f.name}")
    (out / "obj.names").write_text("\n".join(CLASS_NAMES) + "\n")
    (out / "obj.data").write_text(
        f"classes = {len(CLASS_NAMES)}\n"
        "train = data/train.txt\n"
        "names = data/obj.names\n"
        "backup = backup/\n")
    (out / "train.txt").write_text("\n".join(rel) + "\n")


def build_roboflow(frames, labels, out: Path):
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        shutil.copy2(f, img_dir / f.name)
        (lbl_dir / f.with_suffix(".txt").name).write_text(label_for(f, labels))
    (out / "data.yaml").write_text(
        "train: images\nval: images\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--labels", type=Path, default=Path("pseudo_labels"))
    p.add_argument("--format", choices=["cvat", "roboflow"], default="cvat")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: export/<format>)")
    p.add_argument("--zip", action="store_true", help="also produce <out>.zip")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    out = args.out or Path("export") / args.format
    if out.exists():
        shutil.rmtree(out)

    (build_cvat if args.format == "cvat" else build_roboflow)(frames, args.labels, out)
    labelled = sum(1 for f in frames if label_for(f, args.labels).strip())
    print(f"[{args.format}] {len(frames)} frames ({labelled} with boxes, "
          f"{len(frames) - labelled} empty) -> {out}")

    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in out.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        print(f"[zip] {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
