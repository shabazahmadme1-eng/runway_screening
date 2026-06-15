#!/usr/bin/env python3
"""Spatial referencing for the survey: per-frame chainage + a linear distress map.

We have no GPS, but the frames are a continuous pass, so distance ALONG the
runway is recoverable by registering consecutive frames (phase correlation):
pixel shift x GSD = metres advanced per frame -> cumulative station (chainage).
Detections then get a real location ("STA 12.4 m") instead of a frame number.

Outputs:
- frame_chainage.csv      (frame, advance_m, station_m)
- runway_linear_map.png   (distress vs chainage in metres, hot zones marked)
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def advance_px(a, b, scale=0.25):
    """Estimate frame-to-frame shift magnitude (px, full-res) via phase corr."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, None, fx=scale, fy=scale).astype(np.float32)
    gb = cv2.resize(gb, None, fx=scale, fy=scale).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga * win, gb * win)
    return (abs(dx) + 0j + abs(dy) * 1j), (dx / scale, dy / scale), resp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--gsd-mm", type=float, default=1.25)
    args = ap.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if len(frames) < 2:
        raise SystemExit("need >=2 frames")

    # per-frame advance via phase correlation
    advances = [0.0]
    prev = cv2.imread(str(frames[0]))
    for f in frames[1:]:
        cur = cv2.imread(str(f))
        _, (dx, dy), resp = advance_px(prev, cur)
        advances.append(float(np.hypot(dx, dy)))
        prev = cur
    adv = np.array(advances)
    # robustify: replace zeros/outliers with the median advance
    med = np.median(adv[1:][adv[1:] > 0]) if (adv[1:] > 0).any() else 0.0
    adv[(adv <= 0.2 * med) | (adv > 4 * med)] = med
    adv[0] = 0.0
    advance_m = adv * args.gsd_mm / 1000.0
    station_m = np.cumsum(advance_m)

    rows = [[f.name, round(a, 3), round(s, 2)]
            for f, a, s in zip(frames, advance_m, station_m)]
    with open(args.out / "frame_chainage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "advance_m", "station_m"])
        w.writerows(rows)
    station = {f.name: s for f, s in zip(frames, station_m)}

    # align crack density + vegetation onto chainage
    def load(name, col):
        p = args.out / name
        if not p.exists():
            return {}
        return {r["frame"]: float(r[col]) for r in csv.DictReader(open(p))}
    crack = load("crack_density_sam.csv", "mask_density_pct")
    veg = load("surface_features.csv", "vegetation_pct")

    xs = [station[f.name] for f in frames]
    cd = [crack.get(f.name, 0.0) for f in frames]
    vg = [veg.get(f.name, 0.0) for f in frames]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(xs, cd, color="#c0392b", alpha=0.9, label="crack density %")
    ax.fill_between(xs, vg, color="#27ae60", alpha=0.7, label="vegetation %")
    if cd:
        thr = np.mean(cd) + np.std(cd)
        for x, y in zip(xs, cd):
            if y >= thr and y > 0:
                ax.axvspan(x - 0.05, x + 0.05, color="#f1c40f", alpha=0.20, lw=0)
        # label the worst hot zones by station
        order = sorted(zip(xs, cd), key=lambda t: -t[1])[:5]
        for x, y in order:
            ax.annotate(f"STA {x:.1f} m", (x, y), fontsize=7, color="#7a3b12",
                        ha="center", va="bottom")
    ax.set_xlabel("chainage — distance along runway pass (m)")
    ax.set_ylabel("surface coverage (%)")
    ax.set_title("Runway linear distress map — crack & vegetation vs chainage")
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(args.out / "runway_linear_map.png", dpi=130)
    plt.close(fig)

    print("=== Spatial referencing (chainage) ===")
    print(f"frames: {len(frames)} | surveyed length: {station_m[-1]:.1f} m")
    print(f"median frame advance: {med*args.gsd_mm/1000:.3f} m/frame")
    print(f"-> {args.out}/frame_chainage.csv, runway_linear_map.png")


if __name__ == "__main__":
    main()
