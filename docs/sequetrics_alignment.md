# Aligning the pipeline with Sequetrics' product (case-study findings)

Notes from the three public Sequetrics case studies (sequetrics.co.uk),
and what they imply for this repo. The headline: Sequetrics sells a **crack
density + hot-zone inspection product benchmarked on recall and framed
around aviation compliance** — not raw bounding boxes. Our model is the
right engine; the reporting layer is what makes it match the offering.

## The three case studies

| # | Title | Tier | Key facts |
|---|-------|------|-----------|
| 1 | Next-Generation Airfield Inspection (Fife Airport) | UAV + platform | **93% crack recall**; cycle 7–12 min vs 45 min manual; 1 operator; CO₂ 1.2 kg; **crack density** map; ICAO/CAA compliance |
| 2 | Runway Crack Detection from Aerial Imagery (Univ. Edinburgh / CENSIS) | UAV, ~1 mm GSD | orthomosaic → tile crack/no-crack classification + ISO-cluster map (Cracks/Dark/Medium/Light); challenges = motion blur, crack connectivity, scarce labels |
| 3 | Surface Assessment using Geospatial Imagery (Sentinel-2) | Satellite | broad change-detection; flags rubber deposits, patching, grass growth, flooding for UAV/manual follow-up |

The product vision is a **multi-scale funnel**: satellite (broad, cheap) →
UAV (detailed, us) → manual (verify). We are the UAV detection tier.

## What this changes for us

1. **Crack density is the headline KPI, not detection counts.**
   Implemented in `src/runway_report.py`: per-frame `crack_density.csv`
   (crack bbox area ÷ surface area) + an overall figure. It's a bbox-area
   proxy (cracks are thin within their boxes), so it's a consistent
   *relative* index for ranking/trend tracking, not absolute area.

2. **Hot-zone mapping.** `runway_report.py` writes
   `crack_density_profile.png` — density vs frame order, which (for a
   continuous pass) approximates distance along the runway and shades the
   high-density zones. This is the 1-D stand-in for their georeferenced
   density map; the full version needs a stitched orthomosaic (see below).

3. **Recall is the advertised number (93%).** We cannot honestly report
   runway recall without verified ground truth (we skipped CVAT). The report
   computes per-class precision/recall *if* `--gt <verified-labels>` is
   supplied, so the capability is ready the moment any frames are verified.

4. **Compliance framing.** Reports should reference ICAO/CAA and a severity
   grade. A simple size-based low/med/high crack severity is the next add.

## Gaps worth flagging

- **Rubber deposits** (touchdown-zone friction loss) are a class Sequetrics
  actively reports; we have no class or data for it. Candidate for a future
  class once data exists.
- **Standing water / flooding** (post-event, case study 3) — out of scope
  for a single dry survey, satellite-tier anyway.
- **Orthomosaic step.** Case study 2 stitches frames into an orthomosaic
  before tiling, which is what enables true georeferenced density maps and
  resolves crack connectivity across frame edges. If the client supplies a
  full survey, adding a stitch step (ODM / Metashape) is the upgrade path
  from our per-frame profile to their map.
