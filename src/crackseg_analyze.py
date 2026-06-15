#!/usr/bin/env python3
"""Re-derive crack metrics using the pretrained crack-segmentation model.

Replaces the v1+SAM2 crack masks with a crack-specialised segmenter
(OpenSistemas/YOLOv8-crack-seg, AGPL-3.0) which localises cracks as clean thin
traces. Recomputes crack density, length, type and hot zones for the report.
Width is intentionally NOT reported (mask resolution too coarse for true mm
width — see docs/report_enhancement_roadmap.md).

Writes (same schema the report/linear-map consume):
- crack_density_sam.csv   (per-frame crack density from the seg masks)
- crack_metrics.csv (+_summary.json)  (per-crack type + length)
- crack_density_bbox_vs_mask.png      (v1 bbox proxy vs seg true area)
- overlays/<frame>                     (mask overlays for hot frames)
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from skimage.morphology import skeletonize
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def classify(skel, runway_axis="vertical"):
    n_skel = int(skel.sum())
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = convolve(skel.astype(np.uint8), k, mode="constant")
    n_branch = int(((nb >= 3) & skel).sum())
    ys, xs = np.nonzero(skel)
    ang, elong = 90.0, 1.0
    if len(xs) >= 5:
        pts = np.column_stack([ys - ys.mean(), xs - xs.mean()]).astype(float)
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        v = evec[:, -1]
        ang = math.degrees(math.atan2(abs(v[1]), abs(v[0])))
        if ev[0] > 1e-6:
            elong = float((ev[-1] / max(ev[0], 1e-6)) ** 0.5)
    if n_branch >= 5 and elong < 3.5:
        return "alligator"
    a = ang if runway_axis == "vertical" else 90 - ang
    return "longitudinal" if a < 35 else "transverse" if a > 55 else "diagonal"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="weights/crackseg/yolov8m-crackseg.pt")
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--gsd-mm", type=float, default=1.25)
    args = ap.parse_args()

    model = YOLO(args.model)
    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)

    dens_rows, crack_rows = [], []
    type_counts = Counter()
    cid = 0
    tot_mask = tot_surf = tot_len = 0.0
    for i, f in enumerate(frames, 1):
        im = cv2.imread(str(f))
        if im is None:
            continue
        H, W = im.shape[:2]
        area = float(H * W)
        tot_surf += area
        r = model.predict(im, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        union = np.zeros((H, W), bool)
        n = 0
        if r.masks is not None:
            for mk in r.masks.data.cpu().numpy():
                mk = cv2.resize((mk > 0).astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                if mk.sum() < 30:
                    continue
                union |= mk
                ys, xs = np.nonzero(mk)
                sk = skeletonize(mk)
                ln = int(sk.sum()) * args.gsd_mm / 1000.0
                ctype = classify(sk)
                cid += 1
                n += 1
                tot_len += ln
                type_counts[ctype] += 1
                crack_rows.append(dict(
                    crack_id=cid, frame=f.name, plausible_crack=True,
                    type=ctype, severity="n/a", fod_risk=False,
                    length_m=round(ln, 3), mean_width_mm="", max_width_mm="",
                    cx=int(xs.mean()), cy=int(ys.mean()), area_px=int(mk.sum())))
        mask_px = int(union.sum())
        tot_mask += mask_px
        dens_rows.append([f.name, n, 0, mask_px, round(area, 1), 0,
                          round(100 * mask_px / area, 4)])
        if i % 25 == 0 or i == len(frames):
            print(f"  [{i}/{len(frames)}] {cid} cracks", flush=True)

    # write density csv (schema matches report + spatial_map)
    with open(args.out / "crack_density_sam.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_cracks", "bbox_area_px", "mask_area_px",
                    "frame_area_px", "bbox_density_pct", "mask_density_pct"])
        w.writerows(dens_rows)
    fields = ["crack_id", "frame", "plausible_crack", "type", "severity",
              "fod_risk", "length_m", "mean_width_mm", "max_width_mm",
              "cx", "cy", "area_px"]
    with open(args.out / "crack_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(crack_rows)
    overall = 100 * tot_mask / tot_surf if tot_surf else 0
    (args.out / "crack_metrics_summary.json").write_text(json.dumps(dict(
        n_cracks=len(crack_rows), source="OpenSistemas/YOLOv8-crack-seg (AGPL-3.0)",
        total_crack_length_m=round(tot_len, 1), type_counts=dict(type_counts),
        overall_density_pct=round(overall, 3)), indent=2))

    # comparison chart: v1 bbox proxy vs crack-seg true area
    bbox = {r["frame"]: float(r["crack_density_pct"])
            for r in csv.DictReader(open(args.out / "crack_density.csv"))} \
        if (args.out / "crack_density.csv").exists() else {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = range(len(dens_rows))
    mk = [d[6] for d in dens_rows]
    bb = [bbox.get(d[0], 0) for d in dens_rows]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(xs, bb, color="#e6b89c", alpha=0.9, label="v1 bbox proxy")
    ax.fill_between(xs, mk, color="#c0392b", alpha=0.95, label="crack-seg true area")
    ax.set_xlabel("frame index (≈ distance along pass)")
    ax.set_ylabel("crack density (% of surface)")
    ax.set_title("Crack density: v1 bbox proxy vs crack-seg true-area mask")
    ax.legend(loc="upper right", fontsize=9); ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(args.out / "crack_density_bbox_vs_mask.png", dpi=130)
    plt.close(fig)

    # overlays for top hot frames
    odir = args.out / "overlays"; odir.mkdir(exist_ok=True)
    hot = sorted(dens_rows, key=lambda d: -d[6])[:6]
    for d in hot:
        f = d[0]; im = cv2.imread(str(args.frames / f))
        r = model.predict(im, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if r.masks is not None:
            u = np.zeros(im.shape[:2], bool)
            for m in r.masks.data.cpu().numpy():
                u |= cv2.resize((m > 0).astype(np.uint8), (im.shape[1], im.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            red = np.zeros_like(im); red[u] = (0, 0, 255)
            im = cv2.addWeighted(im, 1.0, red, 0.6, 0)
        cv2.imwrite(str(odir / f), im, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"\n=== crack-seg analysis ===")
    print(f"frames {len(dens_rows)} | cracks {len(crack_rows)} | "
          f"overall density {overall:.3f}% | total length {tot_len:.1f} m")
    print(f"type mix: {dict(type_counts)}")
    print(f"-> updated crack_density_sam.csv, crack_metrics.csv(+summary), "
          f"comparison png, overlays")


if __name__ == "__main__":
    main()
