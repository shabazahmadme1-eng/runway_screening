#!/usr/bin/env python3
"""Refine crack detections into pixel masks with SAM2 -> true crack density.

Our bbox-based crack density over-states area because cracks are thin lines
inside fat boxes. This feeds each crack box from the detector into SAM2 as a
box prompt, unions the returned masks per frame, and recomputes density from
actual crack-covered pixels — the measurement runway inspection actually
wants (crack area / surface area).

Outputs:
- crack_density_sam.csv : per-frame bbox vs mask area and density
- overlays/<frame>      : mask overlay for the top hot-zone frames
- printed bbox-proxy vs SAM-mask density comparison
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections", type=Path,
                    default=Path("reports/runway_defects_full/detections.csv"))
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/runway_defects_full"))
    ap.add_argument("--model", default="sam2_t.pt")
    ap.add_argument("--crack-class", default="crack")
    ap.add_argument("--overlays", type=int, default=6,
                    help="save mask overlays for the top-N densest frames")
    args = ap.parse_args()

    boxes = defaultdict(list)
    for r in csv.DictReader(open(args.detections)):
        if r["class"] == args.crack_class:
            boxes[r["frame"]].append([float(r["x1"]), float(r["y1"]),
                                      float(r["x2"]), float(r["y2"])])
    if not boxes:
        raise SystemExit("no crack detections found")
    frames = sorted(boxes)
    sam = SAM(args.model)

    rows = []
    tot_bbox = tot_mask = tot_surface = 0.0
    for i, fname in enumerate(frames, 1):
        fpath = args.frames / fname
        im = cv2.imread(str(fpath))
        if im is None:
            continue
        H, W = im.shape[:2]
        area = float(H * W)
        bb = boxes[fname]
        bbox_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bb)
        res = sam(str(fpath), bboxes=bb, verbose=False)[0]
        if res.masks is None:
            mask_px = 0
            union = np.zeros((H, W), bool)
        else:
            m = res.masks.data.cpu().numpy()           # [N,H,W]
            union = m.sum(0) > 0
            if union.shape != (H, W):
                union = cv2.resize(union.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_px = int(union.sum())
        rows.append([fname, len(bb), round(bbox_area, 1), mask_px, round(area, 1),
                     round(100 * bbox_area / area, 4),
                     round(100 * mask_px / area, 4)])
        tot_bbox += bbox_area
        tot_mask += mask_px
        tot_surface += area
        if i % 20 == 0 or i == len(frames):
            print(f"  [{i}/{len(frames)}] {fname} "
                  f"bbox {100*bbox_area/area:.2f}% -> mask {100*mask_px/area:.3f}%",
                  flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r[6])
    with open(args.out / "crack_density_sam.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_cracks", "bbox_area_px", "mask_area_px",
                    "frame_area_px", "bbox_density_pct", "mask_density_pct"])
        w.writerows(rows)

    # overlays for the densest frames
    odir = args.out / "overlays"
    odir.mkdir(parents=True, exist_ok=True)
    for r in rows_sorted[: args.overlays]:
        fname = r[0]
        im = cv2.imread(str(args.frames / fname))
        res = sam(str(args.frames / fname), bboxes=boxes[fname], verbose=False)[0]
        if res.masks is not None:
            m = (res.masks.data.cpu().numpy().sum(0) > 0)
            if m.shape != im.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (im.shape[1], im.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            red = np.zeros_like(im); red[m] = (0, 0, 255)
            im = cv2.addWeighted(im, 1.0, red, 0.6, 0)
            for x1, y1, x2, y2 in boxes[fname]:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 255), 2)
        cv2.imwrite(str(odir / fname), im, [cv2.IMWRITE_JPEG_QUALITY, 85])

    ob = 100 * tot_bbox / tot_surface if tot_surface else 0
    om = 100 * tot_mask / tot_surface if tot_surface else 0
    print("\n=== crack density: bbox proxy vs SAM2 mask (true area) ===")
    print(f"overall bbox-proxy density: {ob:.3f}%")
    print(f"overall SAM2-mask density:  {om:.3f}%   "
          f"({om/ob*100:.0f}% of the proxy — boxes overstate by "
          f"{ob/om:.1f}x)" if om else "")
    print(f"frames processed: {len(rows)}")
    print(f"-> {args.out}/crack_density_sam.csv, overlays in {odir}")


if __name__ == "__main__":
    main()
