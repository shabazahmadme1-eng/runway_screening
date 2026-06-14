#!/usr/bin/env python3
"""Render detections onto frames with their pixel coordinates burned in.

Reads the report's detections.csv (no re-inference needed) and draws each
box plus a label of the form  "<class> <conf>  (cx,cy) WxH"  so the
annotated frame is self-describing for an inspection report.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

COLORS = {  # BGR
    "crack": (0, 0, 255), "spalling": (0, 165, 255),
    "faded_paint_marking": (0, 255, 255), "repair_patch": (255, 0, 255),
    "gap_vegetation": (0, 255, 0), "fod": (255, 0, 0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--detections", type=Path,
                   default=Path("reports/runway_defects/detections.csv"))
    p.add_argument("--out", type=Path, default=Path("annotated"))
    p.add_argument("--max-size", type=int, default=1600,
                   help="longest output edge in px (0 = full resolution)")
    args = p.parse_args()

    by_frame = defaultdict(list)
    with open(args.detections) as fh:
        for row in csv.DictReader(fh):
            by_frame[row["frame"]].append(row)
    if not by_frame:
        raise SystemExit(f"no detections in {args.detections}")
    args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    for fname, dets in by_frame.items():
        fpath = args.frames / fname
        if not fpath.exists():
            continue
        im = cv2.imread(str(fpath))
        H, W = im.shape[:2]
        thick = max(2, W // 1000)
        for d in dets:
            x1, y1 = int(float(d["x1"])), int(float(d["y1"]))
            x2, y2 = int(float(d["x2"])), int(float(d["y2"]))
            cls = d["class"]
            color = COLORS.get(cls, (255, 255, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"{cls} {float(d['conf']):.2f}  ({cx},{cy}) {x2 - x1}x{y2 - y1}"
            cv2.rectangle(im, (x1, y1), (x2, y2), color, thick)
            cv2.circle(im, (cx, cy), thick + 2, color, -1)
            y_text = y1 - 10 if y1 > 30 else y2 + 28
            cv2.putText(im, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                        W / 2600, color, thick)
        if args.max_size and max(H, W) > args.max_size:
            s = args.max_size / max(H, W)
            im = cv2.resize(im, (int(W * s), int(H * s)))
        cv2.imwrite(str(args.out / fname), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
    print(f"annotated {n} frames with coordinates -> {args.out}")


if __name__ == "__main__":
    main()
