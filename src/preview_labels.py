#!/usr/bin/env python3
"""Render YOLO label boxes onto their frames for quick visual review.

Writes downscaled annotated JPEGs so pseudo-labels can be sanity-checked
before (and after) the manual refinement pass, without opening an
annotation tool.
"""

import argparse
from pathlib import Path

import cv2

CLASS_NAMES = {0: "crack", 1: "spalling", 2: "fod"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (255, 0, 0)}  # BGR
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def draw(frame_path: Path, label_path: Path, out_path: Path, max_size: int):
    im = cv2.imread(str(frame_path))
    h, w = im.shape[:2]
    n = 0
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        cls = int(cls)
        cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(im, (x1, y1), (x2, y2), color, max(2, w // 1000))
        cv2.putText(im, CLASS_NAMES.get(cls, str(cls)), (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 2500, color, max(2, w // 1500))
        n += 1
    if max(h, w) > max_size:
        s = max_size / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--labels", type=Path, default=Path("pseudo_labels"))
    p.add_argument("--out", type=Path, default=Path("previews"))
    p.add_argument("--max-size", type=int, default=1600,
                   help="longest preview edge in px")
    p.add_argument("--all", action="store_true",
                   help="also render frames with zero boxes")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    rendered = boxes = 0
    for f in frames:
        lbl = args.labels / f.with_suffix(".txt").name
        if not lbl.exists():
            continue
        if not args.all and not lbl.read_text().strip():
            continue
        boxes += draw(f, lbl, args.out / f.name, args.max_size)
        rendered += 1
    print(f"rendered {rendered} previews ({boxes} boxes) -> {args.out}")


if __name__ == "__main__":
    main()
