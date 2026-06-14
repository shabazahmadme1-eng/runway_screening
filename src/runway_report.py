#!/usr/bin/env python3
"""Run the production model across every drone frame and emit a defect report.

Outputs, aligned with how runway inspection is actually reported (cf.
Sequetrics case studies — crack *density* and hot-zone maps, not just counts):

- detections.csv      — one row per detection (class, conf, box)
- per_frame.csv       — per-frame detection counts + breakdown
- crack_density.csv   — per-frame crack density (the headline KPI)
- crack_density_profile.png — crack density along the runway (frame order ≈
                         distance along the pass), with hot zones shaded
- printed inventory   — per-class totals + overall crack density

Crack density here is the fraction of surface covered by crack *bounding
boxes* (length × width). Because cracks are thin lines inside their boxes,
this over-states true crack area, so treat it as a consistent relative index
for ranking and trend-tracking, not an absolute area measurement. The
frame-order profile is a 1-D proxy for the georeferenced density map you'd
build from a stitched orthomosaic; concentrations typically fall in
high-stress zones (touchdown, taxiway intersections).

Pass --gt <dir> with verified YOLO labels to also compute per-class
precision/recall (recall being the metric inspection products advertise).
Without verified ground truth no honest recall can be reported.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_gt(gt_dir: Path, stem: str, W: float, H: float):
    """Read a YOLO .txt of ground-truth boxes -> list of (cls, x1,y1,x2,y2)."""
    p = gt_dir / f"{stem}.txt"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        f = ln.split()
        if len(f) < 5:
            continue
        cls = int(float(f[0]))
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        out.append((cls, (cx - bw / 2) * W, (cy - bh / 2) * H,
                    (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/yolov8m_v1.pt")
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("reports/runway_defects"))
    p.add_argument("--conf", type=float, default=0.20,
                   help="lower than default to favour recall for screening")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--crack-class", default="crack",
                   help="class name treated as 'crack' for density")
    p.add_argument("--gt", type=Path, default=None,
                   help="dir of verified YOLO labels -> also report "
                        "precision/recall (needs real ground truth)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for matching detections to --gt")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.iterdir() if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    names = model.names
    crack_ids = {i for i, n in names.items() if n == args.crack_class}

    det_rows = []
    per_frame = []
    density_rows = []  # frame, crack_count, crack_area_px, frame_area_px, density_pct, length_px
    class_totals = Counter()
    frames_per_class = defaultdict(set)
    total_crack_area = 0.0
    total_surface_area = 0.0
    # precision/recall accumulators (only if --gt)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for f in frames:
        r = model.predict(f, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        H, W = r.orig_shape
        frame_area = float(H * W)
        total_surface_area += frame_area
        fc = Counter()
        crack_area = 0.0
        crack_len = 0.0
        crack_count = 0
        preds = []  # (cls_id, box) for gt matching
        for b in r.boxes:
            cid = int(b.cls)
            cls = names[cid]
            conf = float(b.conf)
            x1, y1, x2, y2 = (round(v, 1) for v in b.xyxy[0].tolist())
            det_rows.append([f.name, cls, round(conf, 3), x1, y1, x2, y2])
            fc[cls] += 1
            class_totals[cls] += 1
            frames_per_class[cls].add(f.name)
            preds.append((cid, (x1, y1, x2, y2)))
            if cid in crack_ids:
                crack_area += (x2 - x1) * (y2 - y1)
                crack_len += max(x2 - x1, y2 - y1)
                crack_count += 1
        per_frame.append([f.name, sum(fc.values()), dict(fc)])
        density_pct = 100.0 * crack_area / frame_area if frame_area else 0.0
        density_rows.append([f.name, crack_count, round(crack_area, 1),
                             round(frame_area, 1), round(density_pct, 4),
                             round(crack_len, 1)])
        total_crack_area += crack_area

        if args.gt is not None:
            gts = load_gt(args.gt, f.stem, W, H)
            used = set()
            for cid, box in preds:
                best, bj = 0.0, -1
                for j, (gc, gb) in enumerate(gts):
                    if j in used or gc != cid:
                        continue
                    v = iou(box, gb)
                    if v > best:
                        best, bj = v, j
                if best >= args.iou:
                    tp[cid] += 1
                    used.add(bj)
                else:
                    fp[cid] += 1
            for j, (gc, _) in enumerate(gts):
                if j not in used:
                    fn[gc] += 1

    with open(args.out / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(det_rows)
    with open(args.out / "per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "n_detections", "breakdown"])
        w.writerows(per_frame)
    with open(args.out / "crack_density.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "crack_count", "crack_bbox_area_px",
                    "frame_area_px", "crack_density_pct", "crack_length_px"])
        w.writerows(density_rows)

    profile_path = write_density_profile(density_rows, args.out)

    # ---------------------------------------------------------------- report
    print(f"\n=== Runway defect inventory ({len(frames)} frames @ conf>={args.conf}) ===")
    print(f"{'class':22} {'detections':>10} {'frames_affected':>16}")
    for cls in sorted(class_totals, key=lambda c: -class_totals[c]):
        print(f"{cls:22} {class_totals[cls]:>10} "
              f"{len(frames_per_class[cls]):>13}/{len(frames)}")
    clean = sum(1 for _, n, _ in per_frame if n == 0)
    print(f"\nframes with no detected defect: {clean}/{len(frames)}")

    overall = 100.0 * total_crack_area / total_surface_area if total_surface_area else 0.0
    affected = [d for d in density_rows if d[1] > 0]
    mean_aff = sum(d[4] for d in affected) / len(affected) if affected else 0.0
    print("\n=== Crack density (bbox-area proxy) ===")
    print(f"overall crack density (whole survey):   {overall:.3f}%")
    print(f"mean density over crack-bearing frames: {mean_aff:.3f}%")
    print(f"crack-bearing frames:                   {len(affected)}/{len(frames)}")
    print("hottest zones (top 8 frames by density):")
    for d in sorted(density_rows, key=lambda r: -r[4])[:8]:
        print(f"  {d[0]:24} {d[4]:>7.3f}%   ({d[1]} cracks)")

    if args.gt is not None:
        print(f"\n=== Precision / recall vs {args.gt} (IoU>={args.iou}) ===")
        ids = sorted(set(tp) | set(fp) | set(fn))
        if not ids:
            print("  (no ground-truth labels matched any frame)")
        for cid in ids:
            t, f_, n = tp[cid], fp[cid], fn[cid]
            prec = t / (t + f_) if (t + f_) else 0.0
            rec = t / (t + n) if (t + n) else 0.0
            print(f"  {names[cid]:22} P={prec:5.2f}  R={rec:5.2f}  "
                  f"(tp={t} fp={f_} fn={n})")
    else:
        print("\n[recall] pass --gt <verified-labels> to report precision/recall; "
              "no honest recall is possible without verified ground truth.")

    print(f"\nreports -> {args.out}/ "
          f"(detections.csv, per_frame.csv, crack_density.csv,\n"
          f"          {profile_path.name})")


def write_density_profile(density_rows, out: Path) -> Path:
    """Plot crack density vs frame order (≈ position along the runway pass)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(density_rows)))
    ys = [d[4] for d in density_rows]
    mean = sum(ys) / len(ys) if ys else 0.0
    # "hot" = density more than 1 std above the mean
    if len(ys) > 1:
        var = sum((y - mean) ** 2 for y in ys) / len(ys)
        thresh = mean + var ** 0.5
    else:
        thresh = mean

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(xs, ys, color="#c0392b", alpha=0.85, linewidth=0)
    ax.axhline(mean, color="#2c3e50", lw=1, ls="--",
               label=f"mean {mean:.3f}%")
    ax.axhline(thresh, color="#e67e22", lw=1, ls=":",
               label=f"hot-zone threshold {thresh:.3f}%")
    for x, y in zip(xs, ys):
        if y >= thresh and y > 0:
            ax.axvspan(x - 0.5, x + 0.5, color="#f1c40f", alpha=0.25, linewidth=0)
    ax.set_xlabel("frame index  (≈ distance along runway pass)")
    ax.set_ylabel("crack density  (% of surface, bbox proxy)")
    ax.set_title("Runway crack-density profile — hot zones shaded")
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0)
    fig.tight_layout()
    path = out / "crack_density_profile.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
