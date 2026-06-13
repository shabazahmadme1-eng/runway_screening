#!/usr/bin/env python3
"""Classical-CV auto-labelling for the drone-only classes that have no
training data.

No ML, no manual labels — just color/geometry rules. Quality varies a lot by
class (see the per-class notes); this is a "test the limits" bootstrap, not a
substitute for human review. Output is YOLO-format labels using the 12-class
schema so it can be merged straight into training.

Tractable here:
  5  gap_vegetation        green, high saturation (actual plants in joints/verge)
  9  surface_discoloration green, low saturation (greenish stain on pavement)
  10 paint_marking         small scattered bright-white specks on open pavement
  4  band_joint            long, straight, dark lines (construction joints)
  2  fod                   non-grey/non-green/non-white compact anomalies

Not attempted (subjective or needs examples): aged_surface,
weathered_surface, faded_surface_marking.
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _norm(x, y, bw, bh, W, H):
    return f"{(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}"


def _local_contrast(gray, x, y, bw, bh, pad):
    """Mean brightness of the blob minus the surrounding ring."""
    H, W = gray.shape
    inner = gray[y:y + bh, x:x + bw].mean()
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
    ring = gray[y0:y1, x0:x1]
    n = ring.size - bw * bh
    if n <= 0:
        return 0.0
    ring_mean = (ring.sum() - inner * bw * bh) / n
    return inner - ring_mean


def detect(im):
    """Return list of (cls, x, y, w, h) pixel boxes for one BGR image.

    Tuned for precision over recall: naive thresholds flood on grooved
    runway texture, so every detector gates on a local-contrast check that
    distinguishes a real feature from background aggregate. band_joint is
    intentionally NOT attempted — runway grooves are parallel dark lines
    indistinguishable from construction joints by geometry alone.
    """
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    out = []

    # --- green: vegetation vs surface discoloration (the reliable one) --
    # hue floor 40 (not 30) so yellow runway lines aren't caught as green
    green = cv2.inRange(hsv, (40, 30, 25), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # large close kernel so a grass verge becomes a few regions, not dozens
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(green)
    for x, y, bw, bh, a in stats[1:]:
        if a < 2500:
            continue
        sat = hsv[y:y + bh, x:x + bw, 1][green[y:y + bh, x:x + bw] > 0].mean()
        cls = 5 if sat >= 80 else 9          # plants vs faint stain
        out.append((cls, x, y, bw, bh))

    # --- bright white specks: scattered paint markings -----------------
    # genuine paint is far brighter & whiter than aggregate, and stands out
    # strongly from its immediate surroundings
    white = cv2.inRange(hsv, (0, 0, 225), (180, 28, 255))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    for x, y, bw, bh, a in stats[1:]:
        if not (50 <= a <= 1500):
            continue
        if max(bw, bh) > 4 * min(bw, bh) or max(bw, bh) > 0.10 * max(W, H):
            continue                          # elongated -> a painted line
        if _local_contrast(gray, x, y, bw, bh, 12) < 55:
            continue                          # low contrast -> aggregate
        out.append((10, x, y, bw, bh))

    # --- FOD: compact anomalies that aren't grey/green/white -----------
    asphalt = cv2.inRange(hsv, (0, 0, 40), (180, 55, 185))   # grey range
    anomaly = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_or(asphalt, green), white))
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(anomaly)
    for x, y, bw, bh, a in stats[1:]:
        if not (250 <= a <= 4000):
            continue
        if max(bw, bh) > 4 * min(bw, bh):
            continue                          # skip thin streaks
        if x <= 2 or y <= 2 or x + bw >= W - 2 or y + bh >= H - 2:
            continue                          # ignore the grass-edge border
        if abs(_local_contrast(gray, x, y, bw, bh, 15)) < 35:
            continue                          # low contrast -> texture
        out.append((2, x, y, bw, bh))

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("cv_labels"))
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="restrict to these class ids (default: all detected)")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    frames_with = 0
    for f in frames:
        im = cv2.imread(str(f))
        H, W = im.shape[:2]
        boxes = detect(im)
        if args.classes:
            boxes = [b for b in boxes if b[0] in args.classes]
        lines = [f"{c} {_norm(x, y, bw, bh, W, H)}" for c, x, y, bw, bh in boxes]
        for c, *_ in boxes:
            counts[c] += 1
        if lines:
            frames_with += 1
            (args.out / f.with_suffix(".txt").name).write_text("\n".join(lines) + "\n")

    names = {2: "fod", 4: "band_joint", 5: "gap_vegetation",
             9: "surface_discoloration", 10: "paint_marking"}
    print(f"wrote {sum(counts.values())} boxes across {frames_with}/{len(frames)} "
          f"frames -> {args.out}")
    for c in sorted(counts):
        print(f"  {c} {names.get(c, c)}: {counts[c]}")


if __name__ == "__main__":
    main()
