#!/usr/bin/env python3
"""Render a detailed multi-page PDF runway inspection report (matplotlib only).

Consumes the CSVs produced by runway_report.py and builds an A4 PDF:
  1. Executive summary + KPIs
  2. Crack-density hot-zone profile
  3. Defect inventory + per-frame severity
  4. Per-class detail (cracks / vegetation gaps / paint markings) with
     size statistics and distributions
  5. Annotated hot-zone gallery
  6. Full per-detection appendix (every box, with centre + size)

No external services or binaries — PdfPages is part of matplotlib.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)
ACCENT = "#e67e22"
INK = "#1f2a33"
MUTED = "#6b7785"

PAINT_CLASSES = {"faded_paint_marking", "paint_marking", "faded_surface_marking"}
CLASS_COLOR = {"crack": "#c0392b", "gap_vegetation": "#27ae60"}


def page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=A4)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.06)
    fig.text(0.08, 0.955, title, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.92, 0.957, "Runway Surface Condition Report", fontsize=8,
             color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.945, 0.945], color=ACCENT,
                              lw=2, transform=fig.transFigure))
    if subtitle:
        fig.text(0.08, 0.93, subtitle, fontsize=9, color=MUTED)
    return fig


def finish(pdf, fig, pageno):
    fig.text(0.92, 0.03, f"page {pageno}", fontsize=8, color=MUTED, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def boxstats(rows):
    """rows: detection dicts. -> list of (area_px, frac_pct, w, h) per box."""
    out = []
    for r in rows:
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        w, h = x2 - x1, y2 - y1
        out.append((w * h, w, h))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_inspection_report.pdf"))
    ap.add_argument("--site", default="Runway survey — UAV imagery, ~1.25 mm GSD")
    ap.add_argument("--conf", default="0.10")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    import datetime as dt
    dets = list(csv.DictReader(open(args.reports / "detections.csv")))
    dens = list(csv.DictReader(open(args.reports / "crack_density.csv")))
    profile = args.reports / "crack_density_profile.png"

    by_class = defaultdict(list)
    by_frame = defaultdict(list)
    for r in dets:
        by_class[r["class"]].append(r)
        by_frame[r["frame"]].append(r)
    class_counts = Counter({c: len(v) for c, v in by_class.items()})
    frames_with = {c: len({r["frame"] for r in v}) for c, v in by_class.items()}

    n_frames = len(dens)
    densities = [(d["frame"], float(d["crack_density_pct"]), int(d["crack_count"]))
                 for d in dens]
    total_area = sum(float(d["crack_bbox_area_px"]) for d in dens)
    total_surf = sum(float(d["frame_area_px"]) for d in dens)
    overall = 100 * total_area / total_surf if total_surf else 0.0
    aff = [d for _, d, _ in densities if d > 0]
    mean_aff = sum(aff) / len(aff) if aff else 0.0
    n_paint = sum(class_counts.get(c, 0) for c in PAINT_CLASSES)

    def tier(d):
        return ("clean" if d <= 0 else "low" if d < 1 else
                "moderate" if d < 4 else "high")
    tiers = Counter(tier(d) for _, d, _ in densities)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pg = 0
    with PdfPages(args.out) as pdf:
        # ---- page 1: executive summary -------------------------------------
        pg += 1
        fig = page(pdf, "Runway Surface Condition Report",
                   f"{args.site}   ·   generated {dt.date.today().isoformat()}"
                   f"   ·   detection confidence ≥ {args.conf}")
        kpis = [
            (f"{n_frames}", "frames analysed"),
            (f"{overall:.2f}%", "overall crack density"),
            (f"{class_counts.get('crack',0)}", "crack detections"),
            (f"{frames_with.get('crack',0)}/{n_frames}", "crack-bearing frames"),
            (f"{class_counts.get('gap_vegetation',0)}", "vegetation-gap detections"),
            (f"{n_paint}", "paint-marking detections"),
        ]
        for i, (num, lbl) in enumerate(kpis):
            x = 0.08 + (i % 3) * 0.29
            y = 0.80 - (i // 3) * 0.13
            fig.text(x, y, num, fontsize=22, fontweight="bold", color=ACCENT)
            fig.text(x, y - 0.035, lbl, fontsize=9, color=MUTED)
        body = (
            "This report summarises an automated surface screening of the runway from "
            "drone imagery, performed by a custom YOLOv8 detection model. Findings are "
            "expressed as crack density (the share of surface covered by crack detections) "
            "and grouped into hot zones to prioritise inspection and maintenance, "
            "consistent with routine aerodrome pavement-condition monitoring under "
            "ICAO Annex 14 and CAA CAP 168.\n\n"
            f"Across {n_frames} frames the model detected {class_counts.get('crack',0)} cracks "
            f"on {frames_with.get('crack',0)} frames and {class_counts.get('gap_vegetation',0)} "
            "vegetation-gap features. The overall crack density is "
            f"{overall:.2f}% of imaged surface, rising to {mean_aff:.2f}% averaged over "
            "crack-bearing frames. Hot zones and per-class detail follow.\n\n"
            "Limitations: crack density is a bounding-box-area proxy and over-states true "
            "crack area, so read it as a consistent relative index for ranking and trend "
            "tracking. Precision/recall are not quantified as no human-verified ground "
            "truth was available for this survey. Classes lacking training data for this "
            "domain (e.g. rubber deposits, band joints) are not assessed.")
        fig.text(0.08, 0.52, body, fontsize=10, color=INK, va="top", wrap=True)
        finish(pdf, fig, pg)

        # ---- page 2: density profile ---------------------------------------
        pg += 1
        fig = page(pdf, "Crack-density profile along the runway",
                   "Capture order ≈ distance along the pass; shaded bands are hot zones.")
        if profile.exists():
            ax = fig.add_axes([0.08, 0.45, 0.84, 0.42])
            ax.imshow(plt.imread(str(profile)))
            ax.axis("off")
        hot = sorted(densities, key=lambda t: -t[1])[:10]
        lines = ["Top hot zones (frame  ·  density  ·  cracks):", ""]
        for f, d, c in hot:
            if d > 0:
                lines.append(f"   {f:<26}{d:>7.2f}%     {c} cracks")
        fig.text(0.08, 0.40, "\n".join(lines), fontsize=9.5, color=INK,
                 va="top", family="monospace")
        finish(pdf, fig, pg)

        # ---- page 3: inventory + severity ----------------------------------
        pg += 1
        fig = page(pdf, "Defect inventory & severity")
        ax1 = fig.add_axes([0.08, 0.55, 0.40, 0.30])
        cls = sorted(class_counts, key=lambda c: -class_counts[c])
        ax1.barh(cls[::-1], [class_counts[c] for c in cls[::-1]],
                 color=[CLASS_COLOR.get(c, ACCENT) for c in cls[::-1]])
        ax1.set_title("Detections by class", fontsize=11)
        ax1.tick_params(labelsize=8)
        ax2 = fig.add_axes([0.56, 0.55, 0.36, 0.30])
        torder = ["clean", "low", "moderate", "high"]
        tcol = {"clean": "#2ecc71", "low": "#f1c40f",
                "moderate": "#e67e22", "high": "#c0392b"}
        ax2.bar(torder, [tiers.get(t, 0) for t in torder],
                color=[tcol[t] for t in torder])
        ax2.set_title("Frames by severity tier", fontsize=11)
        ax2.tick_params(labelsize=8)
        tbl = ["Class detail:", ""]
        for c in cls:
            tbl.append(f"   {c:<24}{class_counts[c]:>6} dets   "
                       f"{frames_with[c]:>4}/{n_frames} frames")
        tbl += ["", "Severity tiers by per-frame crack density:",
                "   high ≥4%    moderate 1–4%    low <1%    clean none"]
        fig.text(0.08, 0.46, "\n".join(tbl), fontsize=9.5, color=INK,
                 va="top", family="monospace")
        finish(pdf, fig, pg)

        # ---- page 4: per-class detail --------------------------------------
        pg += 1
        fig = page(pdf, "Per-class detail")
        y = 0.88
        for cname, pretty in [("crack", "Cracks"),
                              ("gap_vegetation", "Vegetation gaps")]:
            rows = by_class.get(cname, [])
            stats = boxstats(rows)
            areas = sorted(s[0] for s in stats)
            ax = fig.add_axes([0.56, y - 0.20, 0.36, 0.17])
            if areas:
                ax.hist(areas, bins=20, color=CLASS_COLOR.get(cname, ACCENT))
            ax.set_title(f"{pretty}: box-area distribution (px²)", fontsize=9)
            ax.tick_params(labelsize=7)
            if areas:
                med = areas[len(areas) // 2]
                mean = sum(areas) / len(areas)
                mn, mx = areas[0], areas[-1]
                txt = (f"{pretty}\n"
                       f"  detections: {len(rows)}\n"
                       f"  frames affected: {frames_with.get(cname,0)}/{n_frames}\n"
                       f"  box area px² — min {mn:.0f}, median {med:.0f},\n"
                       f"     mean {mean:.0f}, max {mx:.0f}\n"
                       f"  mean box: {sum(s[1] for s in stats)/len(stats):.0f}"
                       f" × {sum(s[2] for s in stats)/len(stats):.0f} px")
            else:
                txt = f"{pretty}\n  no detections"
            fig.text(0.08, y, txt, fontsize=10, color=INK, va="top",
                     family="monospace")
            y -= 0.27
        # paint note
        paint_txt = ["Paint markings", ""]
        if n_paint:
            for c in PAINT_CLASSES:
                if class_counts.get(c):
                    paint_txt.append(f"  {c}: {class_counts[c]} detections "
                                     f"({frames_with.get(c,0)} frames)")
        else:
            paint_txt.append(
                "  No paint-marking defects were detected by the model on this\n"
                "  survey. Intact centreline/edge markings are present in the\n"
                "  imagery but the model's paint classes target faded/damaged\n"
                "  markings and did not fire; reliable paint-condition scoring\n"
                "  would need dedicated training data for this runway.")
        fig.text(0.08, y, "\n".join(paint_txt), fontsize=10, color=INK,
                 va="top", family="monospace")
        finish(pdf, fig, pg)

        # ---- pages: hot-zone gallery (2 frames per page) -------------------
        hot_frames = [f for f, d, _ in sorted(densities, key=lambda t: -t[1])
                      if d > 0][:args.top]
        for i in range(0, len(hot_frames), 2):
            pg += 1
            fig = page(pdf, "Hot zones — annotated frames",
                       "red = crack   ·   green = vegetation gap")
            for slot, fname in enumerate(hot_frames[i:i + 2]):
                im = cv2.imread(str(args.frames / fname))
                if im is None:
                    continue
                for r in by_frame.get(fname, []):
                    c = (0, 0, 255) if r["class"] == "crack" else \
                        (0, 180, 0) if r["class"] == "gap_vegetation" else (255, 150, 0)
                    cv2.rectangle(im, (int(float(r["x1"])), int(float(r["y1"]))),
                                  (int(float(r["x2"])), int(float(r["y2"]))),
                                  c, max(2, im.shape[1] // 500))
                ax = fig.add_axes([0.08, 0.50 - slot * 0.42, 0.84, 0.38])
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax.axis("off")
                d = next(dd for ff, dd, _ in densities if ff == fname)
                nc = sum(1 for r in by_frame[fname] if r["class"] == "crack")
                ax.set_title(f"{fname} — {d:.1f}% density · {nc} cracks",
                             fontsize=9, color=INK)
            finish(pdf, fig, pg)

        # ---- appendix: every detection ------------------------------------
        rows = sorted(dets, key=lambda r: (r["frame"], r["class"]))
        hdr = f"{'frame':<22}{'class':<16}{'conf':>5}  {'cx,cy':>11}  {'w x h':>11}"
        per_col = 46
        per_page = per_col * 2
        for start in range(0, len(rows), per_page):
            pg += 1
            fig = page(pdf, "Appendix — all detections",
                       f"rows {start+1}–{min(start+per_page, len(rows))} of {len(rows)}")
            for col in range(2):
                cs = start + col * per_col
                chunk = rows[cs:cs + per_col]
                if not chunk:
                    break
                lines = [hdr, "-" * len(hdr)]
                for r in chunk:
                    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    w, h = int(x2 - x1), int(y2 - y1)
                    lines.append(f"{r['frame']:<22}{r['class']:<16}"
                                 f"{float(r['conf']):>5.2f}  {f'{cx},{cy}':>11}  "
                                 f"{f'{w}x{h}':>11}")
                fig.text(0.06 + col * 0.47, 0.90, "\n".join(lines), fontsize=6.0,
                         color=INK, va="top", family="monospace")
            finish(pdf, fig, pg)

    print(f"[done] {args.out}  ({pg} pages, {len(dets)} detections)")


if __name__ == "__main__":
    main()
