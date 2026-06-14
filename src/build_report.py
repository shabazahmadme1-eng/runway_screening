#!/usr/bin/env python3
"""Build a client-facing runway inspection report from the report outputs.

Reads the CSVs + density profile produced by runway_report.py and renders a
single self-contained HTML file (images embedded as base64, so it's one
portable file you can email or print to PDF from any browser). The framing
follows how runway condition is actually reported — crack density, hot
zones, and a severity breakdown referenced to routine ICAO Annex 14 / CAA
CAP 168 aerodrome surface-maintenance expectations.
"""

import argparse
import base64
import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

import cv2

CRACK_COLOR = (0, 0, 255)        # BGR red
VEG_COLOR = (0, 200, 0)          # BGR green
OTHER_COLOR = (255, 150, 0)


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def b64_jpg_from_array(im) -> str:
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def annotate(frame_path: Path, boxes, max_w=900):
    im = cv2.imread(str(frame_path))
    if im is None:
        return None
    h, w = im.shape[:2]
    thick = max(2, w // 600)
    for cls, x1, y1, x2, y2 in boxes:
        color = (CRACK_COLOR if cls == "crack"
                 else VEG_COLOR if cls == "gap_vegetation" else OTHER_COLOR)
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    if w > max_w:
        s = max_w / w
        im = cv2.resize(im, (max_w, int(h * s)))
    return im


def severity(d):  # frame crack-density % -> tier
    if d <= 0:
        return "clean"
    if d < 1.0:
        return "low"
    if d < 4.0:
        return "moderate"
    return "high"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports/runway_defects"))
    p.add_argument("--frames", type=Path, default=Path("frames"))
    p.add_argument("--out", type=Path, default=Path("reports/inspection_report"))
    p.add_argument("--site", default="Runway survey (UAV, ~1.25 mm GSD)")
    p.add_argument("--top", type=int, default=8, help="hot-zone frames to show")
    args = p.parse_args()

    dens = list(csv.DictReader(open(args.reports / "crack_density.csv")))
    dets = list(csv.DictReader(open(args.reports / "detections.csv")))
    if not dens:
        raise SystemExit("no crack_density.csv — run runway_report.py first")

    boxes_by_frame = defaultdict(list)
    class_counts = Counter()
    for r in dets:
        boxes_by_frame[r["frame"]].append(
            (r["class"], float(r["x1"]), float(r["y1"]),
             float(r["x2"]), float(r["y2"])))
        class_counts[r["class"]] += 1

    n = len(dens)
    densities = [float(r["crack_density_pct"]) for r in dens]
    affected = [d for d in densities if d > 0]
    total_area = sum(float(r["crack_bbox_area_px"]) for r in dens)
    total_surface = sum(float(r["frame_area_px"]) for r in dens)
    overall = 100.0 * total_area / total_surface if total_surface else 0.0
    mean_aff = sum(affected) / len(affected) if affected else 0.0
    tiers = Counter(severity(d) for d in densities)

    hottest = sorted(dens, key=lambda r: -float(r["crack_density_pct"]))
    hottest = [r for r in hottest if float(r["crack_density_pct"]) > 0][: args.top]

    args.out.mkdir(parents=True, exist_ok=True)
    profile_b64 = b64_png(args.reports / "crack_density_profile.png")

    gallery = []
    for r in hottest:
        fname = r["frame"]
        im = annotate(args.frames / fname, boxes_by_frame.get(fname, []))
        if im is None:
            continue
        gallery.append((fname, float(r["crack_density_pct"]),
                        int(r["crack_count"]), b64_jpg_from_array(im)))

    today = dt.date.today().isoformat()
    html = render(args.site, today, n, class_counts, overall, mean_aff,
                  len(affected), tiers, profile_b64, gallery)
    out_html = args.out / "index.html"
    out_html.write_text(html)
    print(f"[done] inspection report -> {out_html}")
    print(f"       {n} frames | overall crack density {overall:.2f}% | "
          f"{len(affected)} crack-bearing | {len(gallery)} hot-zone frames shown")


def render(site, date, n, cc, overall, mean_aff, n_aff, tiers, profile, gallery):
    def card(v, label):
        return (f'<div class="card"><div class="num">{v}</div>'
                f'<div class="lbl">{label}</div></div>')

    cards = "".join([
        card(n, "frames analysed"),
        card(f"{overall:.2f}%", "overall crack density"),
        card(f"{mean_aff:.2f}%", "mean density · affected frames"),
        card(f"{n_aff}/{n}", "crack-bearing frames"),
    ])

    inv = "".join(
        f"<tr><td>{c}</td><td class='r'>{cc[c]}</td></tr>"
        for c in sorted(cc, key=lambda k: -cc[k]))

    tier_order = ["high", "moderate", "low", "clean"]
    tier_rows = "".join(
        f"<tr><td><span class='dot {t}'></span>{t}</td>"
        f"<td class='r'>{tiers.get(t,0)}</td>"
        f"<td class='r'>{100*tiers.get(t,0)//n}%</td></tr>"
        for t in tier_order)

    cells = "".join(
        f'<figure><img src="data:image/jpeg;base64,{b}">'
        f'<figcaption><b>{f}</b> — {d:.1f}% density · {c} cracks</figcaption>'
        f'</figure>' for f, d, c, b in gallery)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Runway Surface Condition Report</title>
<style>
:root{{--ink:#1f2a33;--accent:#e67e22;--muted:#6b7785;--line:#e3e8ee;}}
*{{box-sizing:border-box}}
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);margin:0;background:#f4f6f8}}
.page{{max-width:960px;margin:0 auto;background:#fff;padding:44px 52px;
box-shadow:0 1px 4px rgba(0,0,0,.08)}}
header{{border-bottom:3px solid var(--accent);padding-bottom:14px;margin-bottom:6px}}
h1{{font-size:25px;margin:0 0 2px}}
h2{{font-size:17px;margin:30px 0 10px;color:var(--accent);
border-bottom:1px solid var(--line);padding-bottom:5px}}
.sub{{color:var(--muted);font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}
.card{{background:#fafbfc;border:1px solid var(--line);border-radius:8px;
padding:14px;text-align:center}}
.num{{font-size:22px;font-weight:700;color:var(--accent)}}
.lbl{{font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;
letter-spacing:.03em}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}}
.r{{text-align:right}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
img{{max-width:100%;border-radius:6px;border:1px solid var(--line);display:block}}
figure{{margin:0 0 16px}}
figcaption{{font-size:12px;color:var(--muted);margin-top:5px}}
.gallery{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;
margin-right:7px;vertical-align:middle}}
.dot.high{{background:#c0392b}}.dot.moderate{{background:#e67e22}}
.dot.low{{background:#f1c40f}}.dot.clean{{background:#2ecc71}}
.note{{background:#fff8ef;border:1px solid #f3d9b8;border-radius:8px;
padding:12px 15px;font-size:13px;color:#7a5b2e;margin-top:10px}}
footer{{margin-top:34px;border-top:1px solid var(--line);padding-top:12px;
font-size:12px;color:var(--muted)}}
@media print{{body{{background:#fff}}.page{{box-shadow:none;padding:0}}}}
</style></head><body><div class="page">

<header>
<h1>Runway Surface Condition Report</h1>
<div class="sub">{site} &nbsp;·&nbsp; generated {date} &nbsp;·&nbsp;
automated UAV imagery analysis</div>
</header>

<div class="cards">{cards}</div>

<p>This report summarises an automated surface-crack screening of the runway
from drone imagery. Detection is performed by a custom YOLOv8 model;
findings are reported as <b>crack density</b> (share of surface covered by
crack detections) and ranked into <b>hot zones</b> to prioritise inspection
and maintenance, consistent with routine aerodrome pavement-condition
monitoring under ICAO Annex 14 / CAA CAP 168.</p>

<h2>Crack-density profile along the runway</h2>
<img src="data:image/png;base64,{profile}">
<p class="sub">Density is plotted in capture order, which for a continuous
pass approximates distance along the runway. Shaded bands mark hot zones
(more than one standard deviation above the mean) — the stretches to
inspect first.</p>

<div class="two">
<div><h2>Defect inventory</h2>
<table><tr><th>class</th><th class="r">detections</th></tr>{inv}</table></div>
<div><h2>Severity breakdown (by frame)</h2>
<table><tr><th>tier</th><th class="r">frames</th><th class="r">share</th></tr>
{tier_rows}</table>
<p class="sub">Tiers by per-frame crack density: high &ge;4%, moderate
1–4%, low &lt;1%, clean none.</p></div>
</div>

<h2>Hot zones — highest-density frames</h2>
<div class="gallery">{cells}</div>

<div class="note"><b>Method &amp; limitations.</b> Crack density is computed
from detection bounding boxes; because cracks are thin lines within their
boxes, the percentage over-states true crack area and should be read as a
consistent <i>relative</i> index for ranking and trend-tracking, not an
absolute area. Recall/precision are not quantified here as no human-verified
ground truth was available for this survey; figures reflect the model's
detections. Classes without training data (e.g. rubber deposits, band
joints) are not assessed.</div>

<footer>Automated screening output · for maintenance prioritisation, not a
substitute for certified inspection where required.</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
