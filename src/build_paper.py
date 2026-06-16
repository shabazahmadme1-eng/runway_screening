#!/usr/bin/env python3
"""Render the runway-screening study as a single-column research paper (PDF).

A small flowing-layout engine places headings, justified-width paragraphs and
referenced figures, paginating automatically. Content and numbers are drawn
from the analysis under reports/runway_defects_full/.
"""

import csv
import datetime as dt
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4 = (8.27, 11.69)
INK = "#1f2a33"
MUTED = "#6b7785"
ACCENT = "#c0392b"
LEFT, RIGHT = 0.11, 0.89
WIDTH = RIGHT - LEFT
LINE = 0.0166           # vertical step per text line (fraction of page)
WRAP = 96               # characters per line at body size
R = Path("reports/runway_defects_full")
FIG = Path("reports/paper_figures")


class Paper:
    def __init__(self, out, title):
        self.pdf = PdfPages(out)
        self.title = title
        self.pageno = 0
        self._new_page()

    def _new_page(self):
        if self.pageno:
            self._footer()
            self.pdf.savefig(self.fig); plt.close(self.fig)
        self.fig = plt.figure(figsize=A4)
        self.pageno += 1
        self.y = 0.94
        if self.pageno > 1:
            self.fig.text(LEFT, 0.965, self.title, fontsize=7.5, color=MUTED)
            self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [0.958, 0.958],
                                color="#e3e8ee", lw=0.8, transform=self.fig.transFigure))

    def _footer(self):
        self.fig.text(RIGHT, 0.035, str(self.pageno), fontsize=8, color=MUTED, ha="right")

    def _room(self, need):
        if self.y - need < 0.06:
            self._new_page()

    def gap(self, h=0.012):
        self.y -= h

    def heading(self, text, size=12.5):
        self._room(0.04)
        self.gap(0.012)
        self.fig.text(LEFT, self.y, text, fontsize=size, fontweight="bold", color=INK)
        self.y -= 0.026

    def para(self, text, size=9.7, color=INK, just=True):
        lines = textwrap.wrap(text, WRAP)
        self._room(len(lines) * LINE + 0.006)
        # re-wrap if a page break splits — recompute room per chunk
        i = 0
        while i < len(lines):
            avail = int((self.y - 0.07) / LINE)
            chunk = lines[i:i + max(1, avail)]
            self.fig.text(LEFT, self.y, "\n".join(chunk), fontsize=size,
                          color=color, va="top", linespacing=1.5,
                          ha="left", wrap=False)
            self.y -= len(chunk) * LINE
            i += len(chunk)
            if i < len(lines):
                self._new_page()
        self.y -= 0.006

    def figure(self, path, caption, max_h=0.30):
        img = plt.imread(str(path))
        ar = img.shape[0] / img.shape[1]
        w_in = WIDTH * A4[0]
        h_frac = min(max_h, (w_in * ar) / A4[1])
        cap_lines = textwrap.wrap(caption, 110)
        need = h_frac + 0.01 + len(cap_lines) * 0.013 + 0.02
        self._room(need)
        self.gap(0.012)
        ax = self.fig.add_axes([LEFT, self.y - h_frac, WIDTH, h_frac])
        ax.imshow(img); ax.axis("off")
        self.y -= h_frac + 0.012
        self.fig.text(LEFT, self.y, "\n".join(cap_lines), fontsize=8.2,
                      color=MUTED, va="top", style="italic", linespacing=1.4)
        self.y -= len(cap_lines) * 0.0132 + 0.012

    def finish(self):
        self._footer()
        self.pdf.savefig(self.fig); plt.close(self.fig)
        self.pdf.close()


def load():
    s = json.loads((R / "crack_metrics_summary.json").read_text())
    chain = list(csv.DictReader(open(R / "frame_chainage.csv")))
    surveyed = max(float(r["station_m"]) for r in chain)
    veg = list(csv.DictReader(open(R / "surface_features.csv")))
    veg_cov = sum(float(r["vegetation_pct"]) for r in veg) / len(veg)
    veg_frames = sum(1 for r in veg if float(r["vegetation_pct"]) > 0.05)
    mark_frames = sum(1 for r in veg if float(r["marking_pct"]) > 0.05)
    return s, surveyed, len(veg), veg_cov, veg_frames, mark_frames


def main():
    s, surveyed, nframes, veg_cov, veg_frames, mark_frames = load()
    tc = s["type_counts"]
    title = "Automated Condition Assessment of Runway Surfaces from Low-Altitude UAV Imagery"
    p = Paper("reports/runway_screening_paper.pdf", title)

    # ---- title block
    p.fig.text(LEFT, 0.93, title, fontsize=15.5, fontweight="bold", color=INK,
               va="top", wrap=True)
    p.y = 0.875
    p.fig.text(LEFT, p.y, "Shabaz Ahmad Hasib", fontsize=10.5, color=INK)
    p.y -= 0.02
    p.fig.text(LEFT, p.y, f"Sequetrics  ·  {dt.date.today().strftime('%B %Y')}",
               fontsize=9, color=MUTED)
    p.y -= 0.03

    p.heading("Abstract", 11)
    p.para(
        "Runway pavements wear under aircraft loading, weather and age, and the "
        "cracking, vegetation ingress and faded markings that follow are still "
        "found mostly by slow, disruptive walking surveys. This paper sets out a "
        "vision-based alternative that reads a single low-altitude drone pass and "
        "returns a located, prioritised picture of the surface. Frames captured at "
        "roughly 1.25 mm per pixel are examined along three independent tracks — a "
        "crack-specialised segmentation network and two colour routines for "
        "vegetation and painted markings — and every finding is tied to a real "
        "distance along the runway by registering successive frames against one "
        f"another. Over {surveyed:.0f} m of surveyed pavement the method maps "
        f"{s['n_cracks']} individual cracks, locates grass growing through the "
        "eastern joints, and ranks the worst stretches for follow-up. We report "
        "only the quantities the imagery genuinely supports, set them against the "
        "ASTM D5340 airfield framework, and are candid about where motion blur and "
        "the grooved surface place a ceiling on millimetre-level crack width.")

    p.heading("1.  Introduction")
    p.para(
        "Runway surface condition is a safety matter as much as a maintenance one. "
        "Cracking lets water into the pavement and, once it opens up, sheds "
        "fragments that become foreign object debris; vegetation in the joints "
        "signals both moisture and neglect; worn markings reduce pilot guidance. "
        "International practice (ICAO Annex 14, and CAA CAP 168 in the United "
        "Kingdom) expects aerodromes to inspect these surfaces regularly, yet the "
        "conventional walking survey is labour-intensive, subjective, and forces "
        "runway closures that airports are reluctant to grant.")
    p.para(
        "A drone flying low over the runway offers a way out: it can cover the full "
        "length in minutes and record the surface at a resolution fine enough to "
        "see individual cracks. The difficulty is turning that imagery into "
        "decisions. This paper describes an end-to-end approach that does so, and "
        "is deliberately honest about its limits. Our contribution is threefold: a "
        "three-track detection design suited to top-down, near-macro imagery; a "
        "GPS-free way of placing every finding at a real distance along the runway; "
        "and a measured account of which engineering quantities such imagery can, "
        "and cannot, be trusted to deliver.")

    p.heading("2.  Data acquisition")
    p.para(
        f"The study uses a single continuous drone pass over an asphalt runway, "
        f"sampled into {nframes} frames at five frames per second. The ground "
        "sampling distance is about 1.25 mm per pixel — fine enough that a "
        "millimetre-scale crack spans several pixels. This places the data in an "
        "unusual regime: top-down like aerial survey work, but near-macro like "
        "close-range inspection. The runway is grooved for skid resistance, and "
        "the longitudinal grooves, together with the slight motion blur of a "
        "moving platform, are the two features that most complicate analysis.")

    p.heading("3.  System architecture")
    p.figure(FIG / "fig_architecture.png",
             "Figure 1. The pipeline reads a single UAV pass and splits the work "
             "across three independent detection tracks — crack segmentation, "
             "vegetation, and painted markings — before placing each result at a "
             "real chainage and drawing the findings together into a prioritised "
             "assessment.", max_h=0.30)
    p.para(
        "Figure 1 shows the overall design. The choice to run three separate "
        "tracks, rather than one multi-class detector, is deliberate. The three "
        "targets have very different visual signatures and very different data "
        "situations: cracks are thin, low-contrast and need a learned model; "
        "vegetation and intact paint are strongly coloured and separate cleanly "
        "from grey asphalt with simple, transparent colour rules that need no "
        "training. Keeping them apart lets each use the most reliable method for "
        "its target and keeps the failure modes independent, so a weakness in one "
        "track does not corrupt the others.")
    p.para(
        "The first track sends each frame through a crack-specialised "
        "convolutional network that outlines cracking as thin shapes rather than "
        "coarse boxes, so the area genuinely affected — not the rectangle around "
        "it — is what gets measured. The second and third tracks separate green "
        "vegetation and bright white or yellow paint from the surrounding asphalt "
        "in colour space. A fourth stage recovers how far the platform travelled "
        "between consecutive frames and converts frame numbers into metres along "
        "the runway. The final stage gathers everything by location into hot zones "
        "and a distress inventory, framed against the airfield condition standard.")

    p.heading("4.  Detection methods")
    p.heading("4.1  Surface cracking", 11)
    p.figure(FIG / "fig_crack.png",
             "Figure 2. Source frame (left) and the crack outlines returned by the "
             "segmentation network (right). The model traces the cracking as thin "
             "shapes, which is what allows length and orientation to be measured "
             "rather than merely counted.", max_h=0.20)
    p.para(
        "Cracks are the primary target and the hardest to see. Each frame passes "
        "through a segmentation network specialised for cracking (Figure 2), which "
        "returns one outline per crack. From each outline we reduce the shape to "
        "its centre-line and measure length along that line; the orientation of "
        "the line, relative to the runway axis, sorts the crack into longitudinal, "
        "transverse, or an interconnected 'alligator' pattern — the last being the "
        "one the airfield standard flags for debris risk. Across the survey the "
        f"network maps {s['n_cracks']} cracks totalling roughly "
        f"{s['total_crack_length_m']:.0f} m, dominated by longitudinal cracking "
        f"({tc.get('longitudinal', 0)} cracks) with {tc.get('alligator', 0)} "
        "alligator patches flagged for attention.")

    p.heading("4.2  Vegetation encroachment", 11)
    p.figure(FIG / "fig_vegetation.png",
             "Figure 3. Grass growing through the joints along the verge is "
             "isolated by its colour, which is well separated from grey asphalt "
             "and needs no trained model.", max_h=0.12)
    p.para(
        "Vegetation is the clearest of the three to pick out: living plant matter "
        "is green, and green is far from the grey of asphalt in colour space, so a "
        "simple hue–saturation rule isolates it reliably (Figure 3). It is also "
        "operationally meaningful — grass in the joints points to trapped moisture, "
        "is itself a debris source, and is a visible sign that maintenance has "
        f"lapsed. Vegetation appears in {veg_frames} of {nframes} frames, "
        "concentrated along one verge and around the two-thirds mark of the pass, "
        f"covering about {veg_cov:.2f}% of the imaged surface overall.")

    p.heading("4.3  Painted markings", 11)
    p.figure(FIG / "fig_marking.png",
             "Figure 4. Bright paint is isolated by colour (magenta). Intact "
             "white and yellow lines are caught well, but the same rule also "
             "responds to the bright concrete shoulder on the right — a limitation "
             "discussed below.", max_h=0.12)
    p.para(
        "The third track looks for painted markings by the same colour logic, "
        "keying on bright white and yellow (Figure 4). Intact lines are caught "
        "cleanly. The method is, however, the least dependable of the three: the "
        "bright concrete shoulder and verge satisfy the same brightness rule and "
        "are picked up alongside the paint, and faded markings — which are the "
        "ones an inspector most wants flagged — are missed because they are no "
        f"longer bright. Markings register in {mark_frames} frames, but we treat "
        "this track as indicative of where paint is present rather than as a "
        "measure of its condition.")

    p.heading("5.  Spatial referencing")
    p.figure(R / "runway_linear_map.png",
             "Figure 5. Crack and vegetation coverage plotted against chainage — "
             "distance along the pass in metres. The concentration of cracking in "
             "the first few metres marks the priority stretch; vegetation rises "
             "toward the far end.", max_h=0.22)
    p.para(
        "A list of defects is only useful if each one can be found again on the "
        "ground. With no satellite positioning recorded, we recover position from "
        "the imagery itself: consecutive frames overlap heavily, so aligning each "
        "frame with the next gives the distance the platform moved between them, "
        "which accumulates into a chainage — a running distance along the runway. "
        "The platform advanced about half a metre per frame, and the surveyed "
        f"length comes to {surveyed:.0f} m. Every crack and patch of vegetation "
        "then carries a station value, and the survey reads as a map rather than a "
        "list (Figure 5).")

    p.heading("6.  Results")
    p.figure(R / "crack_density_bbox_vs_mask.png",
             "Figure 6. Crack density along the runway, comparing a coarse "
             "bounding-box estimate with the true area from segmentation. The "
             "outline measurement corrects the magnitude while preserving the "
             "pattern of where cracking concentrates.", max_h=0.22)
    p.para(
        "Crack density — the share of surface taken up by cracking — is the "
        "headline figure for prioritisation. Measuring it from outlines rather "
        "than boxes matters: a box drawn round a thin crack vastly overstates the "
        "affected area, whereas the outline gives the true figure of about "
        f"{s.get('overall_density_pct', 0):.2f}% of surface (Figure 6). Both views "
        "agree on where the trouble is — the heaviest cracking sits in the first "
        "few metres of the pass — but only the outline gives a defensible number. "
        "The vegetation track adds a second, independent map of where the verge is "
        "breaking down. Together they let a maintenance team see, at a glance, "
        "which stretches to walk first.")

    p.heading("7.  Limitations")
    p.para(
        "We are deliberate about what this survey does not claim. We do not publish "
        "a per-crack width, an ASTM D5340 severity grade, or a single condition "
        "index, because the imagery cannot support them honestly. The grooved "
        "surface and the motion blur of a moving platform both blur the thin edge "
        "of a crack — exactly the edge a width measurement depends on — and across "
        "three independent attempts the width either inflated to physically "
        "impossible values or collapsed onto the spacing of the grooves rather "
        "than the cracks. Since width drives severity and the index, we leave them "
        "out rather than report a number we cannot stand behind. Two further "
        "caveats: formal accuracy figures would need a human to confirm a sample of "
        "frames, and a single straight pass yields a strip down the runway rather "
        "than a full-width map of the whole surface.")

    p.heading("8.  Conclusions and outlook")
    p.para(
        "From one short drone pass, the approach delivers a located, prioritised "
        "read of a runway's surface: where the cracking is and how it runs, where "
        "vegetation is taking hold, and which stretches deserve the first walk — "
        "all tied to a real distance along the runway and framed against airfield "
        "practice. The clear next steps follow directly from the limitations. "
        "Sharper capture — a slower or stop-and-stare pass that removes the motion "
        "blur — would unlock trustworthy crack width and, with it, a defensible "
        "severity grade and condition index. A crack model fine-tuned on this "
        "specific surface would sharpen the outlines further, and a second pass "
        "offset across the runway would extend the strip into a full-width map. "
        "None of these change the architecture; each simply strengthens a track "
        "that is already in place.")

    p.heading("References")
    for ref in [
        "[1] ICAO. Annex 14 to the Convention on International Civil Aviation — Aerodromes.",
        "[2] UK CAA. CAP 168: Licensing of Aerodromes.",
        "[3] ASTM D5340, Standard Test Method for Airport Pavement Condition Index Surveys.",
        "[4] FAA AC 150/5320-17A, Airfield Pavement Surface Evaluation and Rating.",
        "[5] Li et al., Automated quantification of crack length and width in asphalt "
        "pavements, Computer-Aided Civil and Infrastructure Engineering, 2024.",
        "[6] OpenSistemas, YOLOv8 crack-segmentation models, Hugging Face.",
        "[7] Reddy et al., Real-time 2D orthomosaic mapping from drone imagery via "
        "sequential image registration, 2023.",
    ]:
        p.para(ref, size=8.6, color=MUTED)

    p.finish()
    print("paper -> reports/runway_screening_paper.pdf")


if __name__ == "__main__":
    main()
