#!/usr/bin/env python3
"""Measure non-crack surface features per frame from RGB (classical CV).

Adds the reliably-detectable, non-crack items to the report:
- vegetation coverage  (green; HSV — plants in joints/verge), and
- paint-marking coverage (white + yellow; presence, not condition).

Both are colour-separable from grey asphalt, so they are robust where
crack-width measurement is not. Outputs surface_features.csv (+ a coverage
profile PNG) with per-frame coverage % and hot zones.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from marking_detect import detect_markings

IMG_EXTS = {".jpg", ".jpeg", ".png"}
GREEN = ((40, 30, 25), (95, 255, 255))            # vegetation
WHITE = ((0, 0, 225), (180, 28, 255))             # white markings
YELLOW = ((18, 60, 120), (38, 255, 255))          # yellow markings


def coverage(hsv, lo, hi):
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_defects_full"))
    args = ap.parse_args()
    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    rows = []
    tv = tm = ts = 0.0
    for f in frames:
        im = cv2.imread(str(f))
        if im is None:
            continue
        H, W = im.shape[:2]
        area = float(H * W)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        veg = coverage(hsv, *GREEN)
        mark = detect_markings(im)        # line-based: paint, not grass
        vp = 100 * (veg > 0).sum() / area
        mp = 100 * (mark > 0).sum() / area
        rows.append([f.name, round(vp, 4), round(mp, 4)])
        tv += (veg > 0).sum(); tm += (mark > 0).sum(); ts += area

    with open(args.out / "surface_features.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "vegetation_pct", "marking_pct"])
        w.writerows(rows)

    # profile chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = range(len(rows))
    veg = [r[1] for r in rows]
    mark = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(xs, veg, color="#27ae60", alpha=0.85, label="vegetation %")
    ax.plot(xs, mark, color="#2980b9", lw=1.1, label="paint markings %")
    ax.set_xlabel("frame index  (≈ distance along runway pass)")
    ax.set_ylabel("surface coverage (% of frame)")
    ax.set_title("Non-crack surface features: vegetation & paint markings")
    ax.legend(loc="upper right", fontsize=9); ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(args.out / "surface_features_profile.png", dpi=130)
    plt.close(fig)

    ov = 100 * tv / ts if ts else 0
    om = 100 * tm / ts if ts else 0
    veg_frames = sum(1 for r in rows if r[1] > 0.05)
    mark_frames = sum(1 for r in rows if r[2] > 0.05)
    print("=== Non-crack surface features ===")
    print(f"frames: {len(rows)}")
    print(f"vegetation: overall {ov:.3f}% coverage | present in {veg_frames}/{len(rows)} frames")
    print(f"markings:   overall {om:.3f}% coverage | present in {mark_frames}/{len(rows)} frames")
    print("hottest vegetation frames:")
    for r in sorted(rows, key=lambda r: -r[1])[:6]:
        print(f"  {r[0]:24} veg {r[1]:.3f}%  mark {r[2]:.3f}%")
    print(f"-> {args.out}/surface_features.csv, surface_features_profile.png")


if __name__ == "__main__":
    main()
