#!/usr/bin/env python3
"""Auto-refine drone pseudo-labels without a human pass.

The bootstrap pass (bootstrap_labels.py) runs at conf 0.10 on purpose —
high recall, lots of false positives, meant to be cleaned by hand in CVAT.
When the manual pass is skipped, that noise would poison training. This
script re-runs the RDD2022-tuned nano at a precision-tuned threshold and
applies NMS to collapse the fragmented/stacked crack boxes RDD2022 models
produce, yielding "silver" labels good enough to train a v1 on.

Only the four RDD-backed classes can appear (crack, spalling,
faded_paint_marking, repair_patch); the eight drone-only classes have no
teacher and stay empty until a human labels them.
"""

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

CLASS_NAMES = {
    0: "crack", 1: "spalling", 2: "fod", 3: "faded_paint_marking",
    4: "band_joint", 5: "gap_vegetation", 6: "aged_surface",
    7: "repair_patch", 8: "weathered_surface", 9: "surface_discoloration",
    10: "paint_marking", 11: "faded_surface_marking",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="weights/yolov8n_rdd.pt")
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("pseudo_labels"))
    p.add_argument("--conf", type=float, default=0.40,
                   help="precision-tuned threshold (vs 0.10 for the manual "
                        "bootstrap); higher = fewer false positives")
    p.add_argument("--iou", type=float, default=0.50,
                   help="NMS IoU; merges fragmented stacked crack boxes")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--keep-empty", action="store_true",
                   help="also write empty label files for frames with no "
                        "detection (treated as background negatives by the "
                        "merge); unsafe without human review — off by default")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cls_counts = Counter()
    frames_with = 0
    for f in tqdm(frames, desc=f"refining @conf>={args.conf}"):
        r = model.predict(f, conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, verbose=False)[0]
        lines = []
        for box in r.boxes:
            cls = int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            cls_counts[cls] += 1
        out_txt = args.out / f.with_suffix(".txt").name
        if lines:
            frames_with += 1
            out_txt.write_text("\n".join(lines) + "\n")
        elif args.keep_empty:
            # only trustworthy as a background negative if a human confirmed
            # it; off by default so unverified misses don't become negatives
            out_txt.write_text("")

    total = sum(cls_counts.values())
    print(f"\nwrote {total} boxes across {frames_with}/{len(frames)} frames "
          f"-> {args.out}")
    for c in sorted(cls_counts):
        print(f"  {c} {CLASS_NAMES[c]}: {cls_counts[c]}")


if __name__ == "__main__":
    main()
