# Transfer-learning datasets for runway surface screening

Evaluation of public datasets to improve the model beyond street-level
RDD2022. Our imagery is unusual: **top-down AND near-macro (1.25 mm GSD)**.
No single public set matches both axes, so we blend a top-down source (for
viewpoint) with crack-rich data (for recall).

## Recommended — wired into `merge_datasets.py`

| Dataset | Images / instances | Format | View / altitude | License | Maps to |
|---------|--------------------|--------|-----------------|---------|---------|
| **UAV-PDD2023** | 2,440 / 11,158 | VOC bbox | vertical top-down, 30 m, 2592×1944 | CC-BY | crack (LC/TC/AC/OC), repair_patch (RP), spalling (PH) |
| **HighRPD** | 11,696 / ~22k | YOLO bbox | top-down, 50 m, 45 MP tiled to 640 | CC-BY | crack (line+block), spalling (pothole) |
| **FOD-A** | 31 cats / 30k+ | VOC bbox | ground-level, runway/taxiway bg | **MIT** | fod (all 31 → fod) |

Downloads:
- UAV-PDD2023 — https://zenodo.org/records/8429208 (DOI 10.5281/zenodo.8429208)
- HighRPD — https://data.mendeley.com/datasets/sywswj7djj/1 (DOI 10.17632/sywswj7djj.1)
- FOD-A — https://github.com/FOD-UNOmaha/FOD-data

## Why these

- **UAV-PDD2023** is the only genuinely vertical top-down set — closest
  viewpoint to our drone, directly attacks the crack domain gap.
- **HighRPD** adds volume (~20k crack annotations) for crack recall.
- **FOD-A** is the only viable FOD source and is MIT-licensed (the most
  commercial-friendly, relevant for a Sequetrics product).

## Considered but lower priority

- **SDNET2018** (56k concrete tiles, cracks 0.06–25 mm) — concrete matches
  runway material and the near-macro crack scale, but it's tile
  *classification*, not detection; would need conversion.
- **Crack segmentation sets** (Crack500, CFD, DeepCrack, GAPs384, bundled in
  CrackSeg9k) — fine crack detail but pixel masks, close-range handheld.
- **Attain / CeyMo** — faded/road markings, but street-level.

## Looks relevant by name, but a poor match (skipped)

- **BARS**, **RLD (Runway Landing Dataset)** — runway *segmentation for
  aircraft landing/navigation*, not surface defects; RLD is synthetic
  (X-Plane 11).
- **PID** — wide street-level view, same gap as RDD2022.

## Caveats

- Even UAV-PDD2023 (30 m) and HighRPD (50 m) are far higher than our
  ultra-low flight — they fix *viewpoint*, not *resolution/scale*. Pair with
  a near-macro concrete set (SDNET/CrackSeg9k) for the best of both.
- `gap_vegetation` has no public dataset — the classical-CV labeler
  (`src/cv_autolabel.py`) remains its only source.
- Licenses must be confirmed for commercial use before shipping.
