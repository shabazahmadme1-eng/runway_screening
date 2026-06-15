#!/usr/bin/env python3
"""Detailed multi-page PDF runway inspection report (matplotlib only).

Standards-aware build: leads with an INDICATIVE condition score (PCI-style,
referencing airfield standard ASTM D5340), real mm crack measurements from
SAM2 masks, ASTM severity bands, crack-type mix, a distress list, and
treatment recommendations. Falls back gracefully if the metrics files are
absent (then it reports detections + density only).

Inputs (under --reports):
  detections.csv, crack_density.csv          (always)
  crack_density_sam.csv                       (SAM2 true density, optional)
  crack_density_bbox_vs_mask.png              (comparison chart, optional)
  crack_metrics.csv, crack_metrics_summary.json (engineering metrics, optional)
  overlays/                                   (SAM2 mask overlays, optional)
"""

import argparse
import csv
import datetime as dt
import json
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
SEV_COL = {"low": "#f1c40f", "medium": "#e67e22", "high": "#c0392b"}


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


def finish(pdf, fig, pageno):
    fig.text(0.92, 0.03, f"page {pageno}", fontsize=8, color=MUTED, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def load(reports):
    d = {}
    d["dets"] = list(csv.DictReader(open(reports / "detections.csv")))
    sam = reports / "crack_density_sam.csv"
    d["sam"] = list(csv.DictReader(open(sam))) if sam.exists() else []
    mj = reports / "crack_metrics_summary.json"
    d["summary"] = json.loads(mj.read_text()) if mj.exists() else None
    mc = reports / "crack_metrics.csv"
    d["metrics"] = list(csv.DictReader(open(mc))) if mc.exists() else []
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_inspection_report.pdf"))
    ap.add_argument("--site", default="Runway survey — UAV imagery, ~1.25 mm GSD")
    args = ap.parse_args()
    d = load(args.reports)
    S = d["summary"]
    cracks = [c for c in d["metrics"] if c.get("plausible_crack") == "True"]

    # class inventory
    cc = Counter(r["class"] for r in d["dets"])
    # true density from SAM
    if d["sam"]:
        mk = sum(float(r["mask_area_px"]) for r in d["sam"])
        surf = sum(float(r["frame_area_px"]) for r in d["sam"])
        true_density = 100 * mk / surf if surf else 0
        sam_by_frame = {r["frame"]: float(r["mask_density_pct"]) for r in d["sam"]}
    else:
        true_density = 0
        sam_by_frame = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pg = 0
    with PdfPages(args.out) as pdf:
        # ---------- page 1: executive summary ----------
        pg += 1
        fig = page("Runway Surface Condition Report",
                   f"{args.site}   ·   generated {dt.date.today().isoformat()}")
        if S:
            score = S["condition_score_indicative"]
            kpis = [
                (f"{score}", f"condition score ({S['condition_class']})"),
                (f"{S['total_crack_length_m']} m", "total crack length"),
                (f"{S['max_crack_width_mm']:.0f} mm", "max crack width"),
                (f"{S['fod_risk_cracks']}", "FOD-risk cracks"),
                (f"{true_density:.2f}%", "true crack density (SAM2)"),
                (f"{S['n_cracks']}", "cracks measured"),
            ]
        else:
            kpis = [(f"{len(set(r['frame'] for r in d['dets']))}", "frames"),
                    (f"{cc.get('crack',0)}", "crack detections"),
                    (f"{true_density:.2f}%", "crack density")]
        for i, (num, lbl) in enumerate(kpis):
            x = 0.08 + (i % 3) * 0.29
            y = 0.82 - (i // 3) * 0.13
            fig.text(x, y, num, fontsize=21, fontweight="bold", color=ACCENT)
            fig.text(x, y - 0.034, lbl, fontsize=8.5, color=MUTED)
        body = (
            "Automated surface screening of the runway from drone imagery "
            "(custom YOLOv8 detection; crack masks refined with SAM2). Crack "
            "dimensions are measured from the masks using the ground sample "
            "distance, classified by severity to airfield standard ASTM D5340 "
            "width bands (low <3.2 mm, medium 3.2–6.4 mm, high >6.4 mm), and "
            "summarised as an indicative condition score with hot zones to "
            "prioritise maintenance under ICAO Annex 14 / CAA CAP 168.\n\n")
        if S:
            body += (
                f"Of {S['n_detections']} crack detections, {S['n_cracks']} were "
                f"measurable thin cracks ({S['wide_features_excluded']} wide "
                "linear features — pavement edges / joints / region fills — were "
                "excluded from width statistics). Total mapped crack length is "
                f"{S['total_crack_length_m']} m; widest crack {S['max_crack_width_mm']:.0f} mm. "
                f"The indicative condition score is {S['condition_score_indicative']}/100 "
                f"({S['condition_class']}).\n\n")
        body += (
            "Limitations: the condition score is INDICATIVE and not a certified "
            "ASTM D5340 PCI (which requires the standard deduct-value curves). "
            "Width accuracy depends on detection quality and pixels-per-crack; "
            "very thin hairlines under-measure. Precision/recall are not "
            "quantified (no human-verified ground truth). Classes without "
            "training data (rubber deposits, joints) are not assessed.")
        fig.text(0.08, 0.55, body, fontsize=9.5, color=INK, va="top", wrap=True)
        finish(pdf, fig, pg)

        # ---------- page 2: density profile ----------
        pg += 1
        fig = page("Crack-density profile along the runway",
                   "Capture order ≈ distance along the pass.")
        cmp_img = args.reports / "crack_density_bbox_vs_mask.png"
        prof = cmp_img if cmp_img.exists() else (args.reports / "crack_density_profile.png")
        if prof.exists():
            ax = fig.add_axes([0.08, 0.46, 0.84, 0.40])
            ax.imshow(plt.imread(str(prof))); ax.axis("off")
        if sam_by_frame:
            hot = sorted(sam_by_frame.items(), key=lambda t: -t[1])[:10]
            lines = ["Top hot zones (frame · true density):", ""]
            lines += [f"   {f:<26}{v:>7.3f}%" for f, v in hot if v > 0]
            fig.text(0.08, 0.40, "\n".join(lines), fontsize=9.5, color=INK,
                     va="top", family="monospace")
        finish(pdf, fig, pg)

        # ---------- page 3: condition, severity, type ----------
        if S:
            pg += 1
            fig = page("Condition assessment (ASTM D5340-aligned)")
            # severity bar
            ax1 = fig.add_axes([0.08, 0.56, 0.38, 0.28])
            sev = S["severity_counts"]
            order = [s for s in ["low", "medium", "high"] if s in sev] or list(sev)
            ax1.bar(order, [sev.get(s, 0) for s in order],
                    color=[SEV_COL.get(s, ACCENT) for s in order])
            ax1.set_title("Cracks by ASTM severity", fontsize=11)
            ax1.tick_params(labelsize=8)
            # type bar
            ax2 = fig.add_axes([0.56, 0.56, 0.36, 0.28])
            typ = S["type_counts"]
            ax2.barh(list(typ)[::-1], [typ[t] for t in list(typ)[::-1]], color=ACCENT)
            ax2.set_title("Cracks by type", fontsize=11)
            ax2.tick_params(labelsize=8)
            txt = [
                f"Indicative condition score: {S['condition_score_indicative']}/100"
                f"  ({S['condition_class']})",
                "  (PCI-style; NOT certified ASTM D5340)",
                "",
                f"Measurable cracks: {S['n_cracks']}   "
                f"wide features excluded: {S['wide_features_excluded']}",
                f"Total crack length: {S['total_crack_length_m']} m",
                f"Crack width mm — median {S['width_mm_p50']:.1f}, "
                f"p90 {S['width_mm_p90']:.1f}, max {S['max_crack_width_mm']:.1f}",
                f"FOD-risk cracks (high severity / alligator): {S['fod_risk_cracks']}",
                "",
                "ASTM width bands: low <3.2 mm · medium 3.2–6.4 mm · high >6.4 mm",
            ]
            fig.text(0.08, 0.48, "\n".join(txt), fontsize=10, color=INK,
                     va="top", family="monospace")
            # treatments
            tr = S["treatments"]
            tl = ["Recommended treatment by severity:", ""]
            tl += [f"  high / alligator : {tr['high/alligator']}",
                   f"  medium          : {tr['medium']}",
                   f"  low             : {tr['low']}"]
            fig.text(0.08, 0.26, "\n".join(tl), fontsize=9, color=INK,
                     va="top", family="monospace")
            finish(pdf, fig, pg)

        # ---------- page 4: inventory ----------
        pg += 1
        fig = page("Defect inventory")
        ax = fig.add_axes([0.08, 0.52, 0.5, 0.32])
        cls = sorted(cc, key=lambda c: -cc[c])
        ax.barh(cls[::-1], [cc[c] for c in cls[::-1]], color=ACCENT)
        ax.set_title("Detections by class", fontsize=11); ax.tick_params(labelsize=8)
        inv = ["Class detail:", ""]
        for c in cls:
            inv.append(f"  {c:<22}{cc[c]:>6} detections")
        fig.text(0.62, 0.84, "\n".join(inv), fontsize=10, color=INK,
                 va="top", family="monospace")
        finish(pdf, fig, pg)

        # ---------- hot-zone gallery ----------
        odir = args.reports / "overlays"
        gallery = sorted(sam_by_frame.items(), key=lambda t: -t[1])[:6] \
            if sam_by_frame else []
        for i in range(0, len(gallery), 2):
            pg += 1
            fig = page("Hot zones — SAM2 crack masks",
                       "red = crack mask · yellow = detection box")
            for slot, (fname, dv) in enumerate(gallery[i:i + 2]):
                p = odir / fname
                img = cv2.imread(str(p)) if p.exists() else cv2.imread(str(args.frames / fname))
                if img is None:
                    continue
                ax = fig.add_axes([0.08, 0.50 - slot * 0.42, 0.84, 0.38])
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
                ax.set_title(f"{fname} — {dv:.2f}% true density", fontsize=9, color=INK)
            finish(pdf, fig, pg)

        # ---------- distress list appendix (measured cracks) ----------
        if cracks:
            rows = sorted(cracks, key=lambda c: -float(c["max_width_mm"]))
            hdr = (f"{'id':>4} {'frame':<20}{'type':<13}{'sev':<7}"
                   f"{'len(m)':>7}{'meanW':>7}{'maxW':>7}{'FOD':>4}")
            per_col, per_page = 46, 92
            for start in range(0, len(rows), per_page):
                pg += 1
                fig = page("Distress list — measured cracks",
                           f"sorted by width · {start+1}–"
                           f"{min(start+per_page,len(rows))} of {len(rows)}")
                for col in range(2):
                    cs = start + col * per_col
                    chunk = rows[cs:cs + per_col]
                    if not chunk:
                        break
                    lines = [hdr, "-" * len(hdr)]
                    for c in chunk:
                        lines.append(
                            f"{c['crack_id']:>4} {c['frame'][:19]:<20}"
                            f"{c['type'][:12]:<13}{c['severity']:<7}"
                            f"{float(c['length_m']):>7.2f}"
                            f"{float(c['mean_width_mm']):>7.1f}"
                            f"{float(c['max_width_mm']):>7.1f}"
                            f"{'Y' if c['fod_risk']=='True' else '':>4}")
                    fig.text(0.05 + col * 0.48, 0.90, "\n".join(lines),
                             fontsize=5.6, color=INK, va="top", family="monospace")
                finish(pdf, fig, pg)

    print(f"[done] {args.out}  ({pg} pages)")


if __name__ == "__main__":
    main()
