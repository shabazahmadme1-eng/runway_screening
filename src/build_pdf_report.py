#!/usr/bin/env python3
"""Multi-feature runway inspection report (PDF, matplotlib only).

Defensible build per docs/report_enhancement_roadmap.md: leads with what the
data supports — SAM2 true crack density, hot zones, total crack length, crack
type mix, vegetation coverage, and a chainage-referenced linear distress map —
framed against ASTM D5340 with an honest coverage matrix and limitations.
It deliberately does NOT publish a width-based severity or numeric PCI
(unreliable on this blurred/grooved imagery).

Inputs under --reports: detections.csv, crack_density_sam.csv,
crack_metrics.csv(+summary.json), surface_features.csv, frame_chainage.csv,
runway_linear_map.png, crack_density_bbox_vs_mask.png, overlays/.
"""

import argparse
import csv
import datetime as dt
import json
from collections import Counter
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


def page(title, subtitle=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.955, title, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.92, 0.957, "Runway Surface Condition Report", fontsize=8,
             color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.945, 0.945], color=ACCENT,
                              lw=2, transform=fig.transFigure))
    if subtitle:
        fig.text(0.08, 0.93, subtitle, fontsize=9, color=MUTED)
    return fig


def finish(pdf, fig, n):
    fig.text(0.92, 0.03, f"page {n}", fontsize=8, color=MUTED, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def rd(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def img_on(fig, path, rect):
    if Path(path).exists():
        ax = fig.add_axes(rect)
        ax.imshow(plt.imread(str(path)))
        ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_inspection_report.pdf"))
    ap.add_argument("--site", default="Runway survey — UAV imagery, ~1.25 mm GSD")
    args = ap.parse_args()
    R = args.reports

    dets = rd(R / "detections.csv")
    sam = rd(R / "crack_density_sam.csv")
    metrics = [c for c in rd(R / "crack_metrics.csv") if c.get("plausible_crack") == "True"]
    veg = rd(R / "surface_features.csv")
    chain = rd(R / "frame_chainage.csv")
    summ = json.loads((R / "crack_metrics_summary.json").read_text()) \
        if (R / "crack_metrics_summary.json").exists() else {}

    cc = Counter(r["class"] for r in dets)
    station = {r["frame"]: float(r["station_m"]) for r in chain}
    surveyed = max((float(r["station_m"]) for r in chain), default=0)
    if sam:
        mk = sum(float(r["mask_area_px"]) for r in sam)
        sf = sum(float(r["frame_area_px"]) for r in sam)
        true_density = 100 * mk / sf if sf else 0
        crack_by_frame = {r["frame"]: float(r["mask_density_pct"]) for r in sam}
    else:
        true_density, crack_by_frame = 0, {}
    veg_by_frame = {r["frame"]: float(r["vegetation_pct"]) for r in veg}
    veg_overall = (sum(veg_by_frame.values()) / len(veg_by_frame)) if veg_by_frame else 0
    veg_frames = sum(1 for v in veg_by_frame.values() if v > 0.05)
    total_len = summ.get("total_crack_length_m", 0)
    n_cracks = summ.get("n_cracks", len(metrics))
    type_counts = summ.get("type_counts", {})
    crack_frames = len({r["frame"] for r in dets if r["class"] == "crack"})
    n_frames = len(station) or len({r["frame"] for r in dets})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pg = 0
    with PdfPages(args.out) as pdf:
        # 1. executive summary
        pg += 1
        fig = page("Runway Surface Condition Report",
                   f"{args.site}   ·   generated {dt.date.today().isoformat()}")
        kpis = [
            (f"{surveyed:.0f} m", "surveyed length"),
            (f"{true_density:.2f}%", "crack density (SAM2 true area)"),
            (f"{total_len:.0f} m", "total crack length"),
            (f"{n_cracks}", "cracks measured"),
            (f"{crack_frames}/{n_frames}", "crack-bearing frames"),
            (f"{veg_overall:.2f}%", "vegetation coverage"),
        ]
        for i, (num, lbl) in enumerate(kpis):
            x = 0.08 + (i % 3) * 0.29
            y = 0.82 - (i // 3) * 0.13
            fig.text(x, y, num, fontsize=20, fontweight="bold", color=ACCENT)
            fig.text(x, y - 0.034, lbl, fontsize=8.3, color=MUTED)
        body = (
            "Automated surface screening of the runway from a continuous drone "
            "pass (custom YOLOv8 detection; crack masks refined with SAM2). "
            "Findings are referenced by chainage (distance along the pass) and "
            "summarised as crack density and vegetation coverage with hot zones "
            "to prioritise maintenance, consistent with routine pavement "
            "monitoring under ICAO Annex 14 / CAA CAP 168 and the ASTM D5340 "
            "distress framework.\n\n"
            f"Over {surveyed:.0f} m surveyed the model maps {n_cracks} cracks "
            f"({total_len:.0f} m total length) across {crack_frames} frames, a "
            f"true crack density of {true_density:.2f}% of imaged surface, and "
            f"vegetation in {veg_frames} frames.\n\n"
            "Scope & limitations: crack density is measured from SAM2 masks "
            "(relative index for prioritisation/trend). Per-crack WIDTH, ASTM "
            "severity grading and a numeric PCI are NOT reported — motion blur "
            "and grooved-runway texture make width unreliable on this survey "
            "(see Limitations). Precision/recall are not quantified (no "
            "verified ground truth). FOD, rubber deposits and marking condition "
            "are out of scope here (see the ASTM coverage matrix).")
        fig.text(0.08, 0.55, body, fontsize=9.5, color=INK, va="top", wrap=True)
        finish(pdf, fig, pg)

        # 2. spatial linear distress map
        pg += 1
        fig = page("Spatial distress map (by chainage)",
                   "Distance along the runway pass, recovered by frame registration.")
        img_on(fig, R / "runway_linear_map.png", [0.06, 0.50, 0.88, 0.36])
        if crack_by_frame and station:
            hot = sorted(((station.get(f, 0), v) for f, v in crack_by_frame.items()),
                         key=lambda t: -t[1])[:8]
            lines = ["Priority crack hot zones (station · density):", ""]
            lines += [f"   STA {s:6.1f} m     {v:5.2f}%" for s, v in hot if v > 0]
            fig.text(0.08, 0.44, "\n".join(lines), fontsize=9.5, color=INK,
                     va="top", family="monospace")
        finish(pdf, fig, pg)

        # 3. crack density profile (bbox vs SAM)
        pg += 1
        fig = page("Crack density — measurement",
                   "Bounding-box proxy vs SAM2 true-area mask.")
        img_on(fig, R / "crack_density_bbox_vs_mask.png", [0.06, 0.48, 0.88, 0.38])
        fig.text(0.08, 0.42,
                 "Bounding boxes overstate crack area ~3x; SAM2 masks give the\n"
                 "true crack-covered area. Both preserve the same hot-zone\n"
                 "pattern — only the magnitude is corrected.",
                 fontsize=10, color=INK, va="top", family="monospace")
        finish(pdf, fig, pg)

        # 4. inventory + vegetation + crack type
        pg += 1
        fig = page("Feature inventory")
        ax = fig.add_axes([0.08, 0.55, 0.5, 0.30])
        cls = sorted(cc, key=lambda c: -cc[c])
        ax.barh(cls[::-1], [cc[c] for c in cls[::-1]], color=ACCENT)
        ax.set_title("Detections by class", fontsize=11); ax.tick_params(labelsize=8)
        lines = ["Cracks", f"  detections: {cc.get('crack',0)}",
                 f"  measured cracks: {n_cracks}", f"  total length: {total_len:.0f} m",
                 f"  type mix: {type_counts}", "",
                 "Vegetation", f"  coverage: {veg_overall:.2f}% overall",
                 f"  present in {veg_frames}/{n_frames} frames"]
        fig.text(0.62, 0.84, "\n".join(lines), fontsize=9.5, color=INK,
                 va="top", family="monospace")
        fig.text(0.08, 0.46,
                 "Crack type is from skeleton orientation (geometry is robust);\n"
                 "vegetation from a green colour threshold (separable from asphalt).",
                 fontsize=8.5, color=MUTED, va="top")
        finish(pdf, fig, pg)

        # 5. ASTM D5340 coverage matrix
        pg += 1
        fig = page("ASTM D5340 coverage matrix",
                   "What this survey assesses, and what needs more.")
        matrix = [
            ("Cracking (long./trans.)", "ASSESSED", "density, length, location, type"),
            ("Vegetation encroachment", "ASSESSED", "coverage % + hot zones (operational)"),
            ("Patching", "PARTIAL", "detector class present; not validated here"),
            ("Crack width / severity / PCI", "NOT REPORTED", "blur + groove texture (see Limitations)"),
            ("FOD", "OUT OF SCOPE", "needs a trained FOD model (FOD-A)"),
            ("Rubber deposits", "OUT OF SCOPE", "needs TDZ location + friction data"),
            ("Raveling / weathering", "OUT OF SCOPE", "texture; unreliable on blurred RGB"),
            ("Marking condition", "OUT OF SCOPE", "colour threshold catches shoulder/verge"),
            ("Ponding / drainage", "N/A", "dry survey"),
        ]
        y = 0.86
        fig.text(0.08, y, f"{'item':<30}{'status':<15}note", fontsize=9,
                 family="monospace", color=INK, fontweight="bold")
        y -= 0.028
        col = {"ASSESSED": "#27ae60", "PARTIAL": "#e67e22", "NOT REPORTED": "#c0392b",
               "OUT OF SCOPE": "#6b7785", "N/A": "#6b7785"}
        for item, status, note in matrix:
            fig.text(0.08, y, f"{item:<30}", fontsize=8.5, family="monospace", color=INK)
            fig.text(0.08 + 0.30 * 0.0 + 0.355, y, f"{status:<14}", fontsize=8.5,
                     family="monospace", color=col[status], fontweight="bold")
            fig.text(0.08, y - 0.014, f"      {note}", fontsize=7.5,
                     family="monospace", color=MUTED)
            y -= 0.040
        finish(pdf, fig, pg)

        # 6. hot-zone gallery (SAM overlays)
        odir = R / "overlays"
        gallery = sorted(crack_by_frame.items(), key=lambda t: -t[1])[:4]
        if gallery:
            for i in range(0, len(gallery), 2):
                pg += 1
                fig = page("Crack hot zones — SAM2 masks",
                           "red = crack mask · yellow = detection box")
                for slot, (fname, dv) in enumerate(gallery[i:i + 2]):
                    p = odir / fname
                    im = cv2.imread(str(p)) if p.exists() else cv2.imread(str(args.frames / fname))
                    if im is None:
                        continue
                    ax = fig.add_axes([0.08, 0.50 - slot * 0.42, 0.84, 0.38])
                    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off")
                    s = station.get(fname, 0)
                    ax.set_title(f"{fname} — STA {s:.1f} m — {dv:.2f}% density",
                                 fontsize=9, color=INK)
                finish(pdf, fig, pg)

        # 7. limitations
        pg += 1
        fig = page("Measurement limitations & upgrade path")
        txt = (
            "Why per-crack width, ASTM severity and PCI are not reported:\n\n"
            "  - Motion blur and grooved-runway texture (the same challenges\n"
            "    noted in published runway-crack studies) blur the thin crack\n"
            "    edge needed for width.\n"
            "  - Vanilla SAM2 segments a broad region, not the 1-3 px crack\n"
            "    line, so mask width overstates (~46 mm median — not physical).\n"
            "  - A dark-line method collapses to a near-constant ~5 mm, i.e. it\n"
            "    measures the groove/texture scale, not real crack width.\n\n"
            "What IS reliable here: crack density & hot zones (prioritisation),\n"
            "crack count/location/length, crack type mix, vegetation coverage,\n"
            "and chainage referencing.\n\n"
            "Upgrade path to certified-style width + ASTM severity + PCI:\n"
            "  - capture deblurred / slower (stop-and-stare) imagery, and/or\n"
            "  - use a crack-fine-tuned segmenter (CrackSAM) instead of vanilla\n"
            "    SAM2; then width-based severity and PCI become trustworthy.\n\n"
            "Other notes: precision/recall need human-verified ground truth;\n"
            "a single video pass yields a strip, not a full-width orthomosaic.")
        fig.text(0.08, 0.86, txt, fontsize=9.5, color=INK, va="top", family="monospace")
        finish(pdf, fig, pg)

        # 8. distress list (by chainage)
        if metrics:
            rows = sorted(metrics, key=lambda c: station.get(c["frame"], 0))
            hdr = f"{'id':>4} {'STA(m)':>7} {'frame':<20}{'type':<13}{'len(m)':>7}"
            per_col, per_page = 46, 92
            for start in range(0, len(rows), per_page):
                pg += 1
                fig = page("Distress list — cracks by chainage",
                           f"{start+1}–{min(start+per_page,len(rows))} of {len(rows)}")
                for col in range(2):
                    cs = start + col * per_col
                    chunk = rows[cs:cs + per_col]
                    if not chunk:
                        break
                    lines = [hdr, "-" * len(hdr)]
                    for c in chunk:
                        s = station.get(c["frame"], 0)
                        lines.append(f"{c['crack_id']:>4} {s:>7.1f} "
                                     f"{c['frame'][:19]:<20}{c['type'][:12]:<13}"
                                     f"{float(c['length_m']):>7.2f}")
                    fig.text(0.05 + col * 0.48, 0.90, "\n".join(lines),
                             fontsize=5.8, color=INK, va="top", family="monospace")
                finish(pdf, fig, pg)

    print(f"[done] {args.out}  ({pg} pages)")


if __name__ == "__main__":
    main()
