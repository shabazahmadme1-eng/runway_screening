# Report enhancement roadmap — toward an industry-grade runway report

Research into how professional pavement/runway condition reports are built,
and what we can add given our current assets (v1 detector + SAM2 crack masks
+ ~1.25 mm GSD). Goal: move from "annotated frames + density" to a
standards-based condition assessment.

## The anchor: ASTM D5340 (airfield PCI) — not D6433

- **ASTM D5340** is the *airport* Pavement Condition Index standard;
  [ASTM D6433](https://store.astm.org/d6433-20.html) is roads/parking lots.
  For a runway, **D5340** is the correct reference.
  [ASTM D5340](https://store.astm.org/d5340-20.html) ·
  [FAA AC 150/5380-7B airport pavement management](https://www.faa.gov/documentlibrary/media/advisory_circular/150-5380-7b.pdf)
- **PCI** is a 0–100 index ([Wikipedia/PCI](https://en.wikipedia.org/wiki/Pavement_condition_index)):
  distress *type* × *severity* × *extent* → deduct values → PCI. Classes:
  Good 86–100, Satisfactory 71–85, Fair 56–70, Poor 41–55, Very Poor 26–40,
  Serious 11–25, Failed 0–10.
- D5340 severity is partly **width-based** for linear cracks and
  **pattern-based** for alligator cracking — and notably flags **FOD
  potential at high-severity** cracking, which ties directly to the runway
  safety narrative.
- Crack width severity thresholds (from D6433/D5340 practice):
  **low < 1/8" (3.2 mm), medium 1/8–1/4" (3.2–6.4 mm), high > 1/4" (6.4 mm)**.

## Why our data can hit it: GSD → millimetres

- At **~1.25 mm/px**, a 6 mm crack spans ~5 px — measurable. mm-scale GSD is
  exactly what crack-width measurement requires.
  [GSD for crack inspection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12900104/) ·
  [Wingtra GSD](https://wingtra.com/surveying-gis/ground-sample-distance/)
- Caveat: resolution/pixels-per-crack matters as much as GSD; very thin
  hairlines (<2 px) under-measure. Report widths with that limitation noted.

## From SAM2 masks → engineering measurements

The SAM2 crack masks we already produce can be turned into real numbers
([crack quantification via skeleton + distance transform](https://onlinelibrary.wiley.com/doi/10.1111/mice.13344),
[end-to-end foundation-model + skeletonization](https://www.sciencedirect.com/science/article/abs/pii/S0952197626005208)):
- **Length** = skeletonise the mask, count connected skeleton path × GSD → metres.
- **Width** = 2 × Euclidean distance transform at each skeleton pixel × GSD →
  mm (report mean and max width per crack).
- **Type** = orientation of the skeleton vs runway axis →
  longitudinal / transverse; high local mask density + interconnection →
  alligator (the FOD-risk pattern).

## What professional UAV pavement reports contain

[Drone pavement deliverables](https://www.dronedeploy.com/blog/pavement-inspections-with-atlas10) ·
[UAV runway inspection (Sensors 2026)](https://doi.org/10.3390/s26041100):
- georeferenced distress **map** (orthomosaic + defect masks → real coords)
- a **distress list** (per defect: type, severity, length, width, location)
- a **PCI score** + condition class
- a **summary** and a multi-year **treatment / budget plan**

## Prioritised plan

**Tier 1 — build now (high impact, feasible with what we have):**
1. **Crack mm-quantification** from SAM2 masks: per-crack length (m), mean/max
   width (mm) via skeleton + distance transform × GSD.
2. **ASTM D5340 severity** per crack from width (low/med/high) — replaces our
   ad-hoc density tiers with the recognised airfield standard.
3. **PCI-style condition score** (0–100 + class) — even a simplified,
   clearly-labelled deduct index gives airports the headline number they expect.

**Tier 2 — strong adds, moderate effort:**
4. **Crack-type classification** (longitudinal / transverse / alligator) with
   the alligator → FOD-potential safety flag.
5. **Distress-list table** matching industry deliverables.
6. **Treatment recommendations** mapped from severity (seal / rout-and-seal /
   patch / reconstruct).

**Tier 3 — bigger lift, needs more inputs:**
7. **True georeferencing** — needs GPS/EXIF + an orthomosaic (ODM); our video
   frames likely lack GPS, so until then report relative position (frame index)
   + mm measurements.
8. **Trend / time-series** — needs repeat surveys.

**Recommendation:** build Tier 1 before regenerating the PDF, so the report
leads with a standards-based PCI + real mm crack measurements rather than a
proxy density. Confirm the true GSD first (assumed ~1.25 mm/px) so the mm
conversions are accurate.

## Reality check: per-crack WIDTH / PCI is NOT reliable on this imagery

Implementing Tier 1 exposed a hard limit. Two width methods both fail to give
trustworthy per-crack width, so ASTM severity grading and a width-based PCI
are **not defensible** on the current data:

- **SAM2 mask width** over-segments: vanilla SAM2 with a box prompt fills a
  broad region, not the thin crack line → median "width" 46 mm, every crack
  flagged "high". Physically impossible (a 46 mm crack is not a crack).
- **Dark-line (blackhat+Otsu) width** is more plausible (~3.5–5 mm) but
  collapses to a near-constant 5 mm (median 5.0, p90 5.0) — it's measuring the
  grooved-texture / motion-blur scale, not real per-crack width variation.

Root causes (both match case study #2's stated challenges): **motion blur**
and **grooved-runway texture**, plus vanilla SAM2 not being a thin-crack
segmenter. Inspection of raw crops confirms some genuine cracks (e.g.
frame_000274) but also many ambiguous blurred/grooved detections.

**Decision:** do not publish width-based severity or a numeric PCI from this
survey. Report the **defensible** metrics instead:
- crack **density** (SAM2 area; relative) → hot-zone prioritisation
- crack **count / locations / hot zones**
- crack **type mix** (longitudinal vs transverse — geometry is robust)
- **total crack length** (approximate)
- an explicit **measurement-limitations** section (blur, groove texture)

**Upgrade path to true width/PCI:** deblurred or slower/stop-and-stare capture
at the stated ~1.25 mm GSD, and/or a **fine-tuned** crack segmenter
(CrackSAM-style) rather than vanilla SAM2. Then ASTM D5340 severity + PCI
become trustworthy.

## Adding features APART from cracks — what's reliable

Airfield reports cover more than cracks. ASTM D5340's distress catalogue
(asphalt) groups distresses as load-related (alligator, corrugation, rutting,
shoving), climate-related (block / joint-reflective / longitudinal /
transverse cracking, raveling, weathering) and other (bleeding, depression,
jet-blast erosion, oil spillage, polished aggregate, patching, slippage
cracking, swell). Vegetation, FOD and rubber deposits are operational/safety
items, not D5340 distresses.
[ASTM D5340](https://store.astm.org/d5340-20.html) ·
[FAA AC 150/5320-17A](https://www.faa.gov/documentLibrary/media/Advisory_Circular/150-5320-17a.pdf)

What is reliably detectable from our RGB top-down imagery (no new training),
tested in `src/surface_features.py`:

- **Vegetation — RELIABLE, ADD.** Green is colour-separable from grey asphalt.
  Overall ~0.10% coverage, present in 85/274 frames, concentrated around frames
  186–190 and 252. Operationally meaningful (plants in joints = water ingress,
  FOD source, sign of neglect). Reported as coverage % + hot zones, like cracks.
- **Paint markings — NOT reliable.** White/yellow thresholds catch the bright
  concrete shoulder/verge and miss faded lines (the real yellow line in
  frame_000188 isn't caught). Same flooding issue seen earlier. Report as a
  limitation, not a metric.
- **FOD** — needs training (no plug-in; FOD-A is dataset-only).
- **Rubber deposits** — touchdown-zone glossy darkening; detection normally
  starts from friction/texture data, needs the TDZ location and isn't reliable
  from single RGB. [SKYbrary: rubber removal](https://skybrary.aero/articles/removal-rubber-runway)
- **Standing water / ponding** — none in a dry survey; little RGB-only method.
- **Raveling / weathering / oxidation** — texture/colour; unreliable on
  motion-blurred grooved RGB.

**Net:** the one solid non-crack add is **vegetation**. The report should also
carry an **ASTM D5340 coverage matrix** — what we assess (cracking, vegetation)
vs what needs more (FOD, rubber, raveling, marking condition) and why — which
is honest and shows command of the standard.
