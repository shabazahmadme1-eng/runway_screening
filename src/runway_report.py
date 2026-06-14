#!/usr/bin/env python3
"""Run the production model across every drone frame and emit a defect report.

Produces a per-detection CSV, a per-frame summary CSV, and a printed
inventory of how much of each defect class the runway shows. Optionally
saves annotated frames for the worst offenders.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="weights/yolov8m_v1.pt")
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("reports/runway_defects"))
    p.add_argument("--conf", type=float, default=0.20,
                   help="lower than default to favour recall for screening")
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    names = model.names

    det_rows = []
    per_frame = []
    class_totals = Counter()
    frames_per_class = defaultdict(set)

    for f in frames:
        r = model.predict(f, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        fc = Counter()
        for b in r.boxes:
            cls = names[int(b.cls)]
            conf = float(b.conf)
            x1, y1, x2, y2 = (round(v, 1) for v in b.xyxy[0].tolist())
            det_rows.append([f.name, cls, round(conf, 3), x1, y1, x2, y2])
            fc[cls] += 1
            class_totals[cls] += 1
            frames_per_class[cls].add(f.name)
        per_frame.append([f.name, sum(fc.values()), dict(fc)])

    with open(args.out / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(det_rows)
    with open(args.out / "per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_detections", "breakdown"])
        w.writerows(per_frame)

    print(f"\n=== Runway defect inventory ({len(frames)} frames @ conf>={args.conf}) ===")
    print(f"{'class':22} {'detections':>10} {'frames_affected':>16}")
    for cls in sorted(class_totals, key=lambda c: -class_totals[c]):
        print(f"{cls:22} {class_totals[cls]:>10} "
              f"{len(frames_per_class[cls]):>13}/{len(frames)}")
    clean = sum(1 for _, n, _ in per_frame if n == 0)
    print(f"\nframes with no detected defect: {clean}/{len(frames)}")
    print(f"reports -> {args.out}/detections.csv, per_frame.csv")


if __name__ == "__main__":
    main()
