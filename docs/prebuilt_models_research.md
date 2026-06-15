# Pre-built models for the classes v1 can't do — research findings

Question: are there pre-trained/off-the-shelf models we can plug in (little or
no retraining) to cover the defect classes our v1 model misses — spalling,
FOD, paint markings, rubber deposits, joints, patching, etc.?

**Short answer: no single off-the-shelf model fits our domain (top-down AND
near-macro ~1.25 mm GSD).** Every relevant pretrained model is either
street-level (domain gap) or dataset-only (no released weights). The one
high-value plug-in is **SAM2**, used as a segmentation refiner on our
existing detections — it doesn't add classes but upgrades crack density from
a bbox proxy to a real measurement.

> Note: re-derived via direct searches after the parallel research agents'
> write-ups were truncated by a session limit.

## By class / category

### Spalling / potholes — pretrained models exist, but street-level
- Roboflow Universe hosts many deployable pothole models with API endpoints
  (e.g. 9k+ image projects). [Roboflow pothole search](https://universe.roboflow.com/search?q=class:pothole)
- Hugging Face RDD2022 models: [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022),
  [cvtechniques/road-damage-detection-yolov11](https://huggingface.co/cvtechniques/road-damage-detection-yolov11),
  [oracl4/RoadDamageDetection (YOLOv8)](https://github.com/oracl4/RoadDamageDetection).
- **Verdict:** all dashcam/street-level — same top-down domain gap as RDD2022,
  which we already fold into our own training. A separate pretrained one adds
  little. Needs fine-tuning regardless.

### FOD (foreign object debris) — dataset only, no public weights
- [FOD-A dataset](https://github.com/FOD-UNOmaha/FOD-data) (31 categories,
  30k+ instances, MIT). Papers build YOLO variants on it (Bi-YOLO,
  [PGDIG-YOLO](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-4/043014/),
  [improved YOLOv8](https://dl.acm.org/doi/10.1145/3656766.3656920)) but
  **none release downloadable weights.**
- FOD-A is ground-level/oblique → poor transfer to top-down high-altitude.
- **Verdict:** no plug-in. We must train on FOD-A ourselves (already the plan).

### Paint / runway markings — no usable pretrained model for top-down
- Lane-detection nets (LaneNet, CLRNet, Ultra-Fast-Lane-Detection) assume a
  forward-facing driving camera; top-down breaks their geometry. Not usable.
- Aerial/remote-sensing models segment **road area, not painted markings**
  ([Massachusetts Roads / SpaceNet](https://github.com/aavek/Satellite-Image-Road-Segmentation),
  [SatlasPretrain](https://github.com/allenai/satlaspretrain_models)).
- **Verdict:** no pretrained marking model fits. Pragmatic path: *intact*
  runway paint is high-contrast white/yellow → a classical-CV colour
  threshold (we already have `cv_autolabel.py`) is the reliable plug-in for
  intact markings; faded markings remain the hard, data-hungry case.

### Foundation / zero-shot — SAM2 is the one worth using
- **SAM2 (Apache-2.0)** — promptable segmentation. With a detector supplying
  box prompts, SAM-based crack segmentation reaches **Dice ≈ 0.7** and
  **mIoU 0.69 (CFD) / 0.59 (Crack500)**, comparable to fully-supervised
  models. Well-documented "YOLO box → SAM mask" two-stage pattern.
  [Segment Any Crack](https://arxiv.org/html/2504.14138v1) ·
  [two-stage YOLOv11+SAM](https://www.mdpi.com/2075-5309/16/4/794) ·
  [Crack SAM](https://link.springer.com/article/10.1186/s43065-024-00103-1) ·
  [SAM2 in Ultralytics](https://docs.ultralytics.com/models/sam-2)
- **Grounding DINO** — text-prompted open-set detection;
  [zero-shot crack + SAM demo](https://github.com/capjamesg/zero-shot-crack-detection).
  Strong on clean macro cracks (e.g. 92% on eggshells) but, like YOLO-World,
  unreliable on grooved-runway micro-texture. Worth a quick try for
  markings/FOD via text prompts; low confidence for fine cracks.
- **Verdict:** SAM2 = high value (as a refiner, not a class-adder);
  Grounding DINO = low-effort experiment only.

### Commercial / SaaS — competitors, not components
- Pavemetrics (laser LCMS), Vaisala, Dynatest etc. are sensor systems /
  services, not licensable models. Relevant as competitive context, not as
  plug-ins.

## Shortlist (most promising)

1. **SAM2 as a crack-mask refiner** — feed v1's crack boxes in as prompts,
   get pixel-accurate masks → **true crack length/area → real crack density**
   (removes the bbox-area-proxy caveat in our report). Biggest, cleanest win.
   Apache-2.0, runs locally. Does *not* add new classes.
2. **Classical-CV colour threshold for intact paint markings** — the only
   reliable way to add the paint class without new labelled data.
3. **Train FOD on FOD-A ourselves** — no shortcut exists; weights aren't
   published and the domain differs.
4. **Skip** redundant street-level RDD/pothole pretrained models — we already
   train on that data.

**Bottom line:** the missing classes can't be bought off the shelf for this
domain. The realistic gains are (a) SAM2 to make cracks *measurably* better,
(b) classical CV for intact paint, and (c) training FOD ourselves.

## Result: SAM2 wired in (implemented)

`src/crack_segment.py` runs SAM2 (sam2_t, Apache-2.0) on v1's crack boxes and
recomputes density from true mask area:

- overall **bbox-proxy density 2.40% → SAM2 true-area 0.76%** (boxes overstate
  by ~3.2×). The hot-zone pattern is preserved; only the magnitude is corrected.
- Overlays also surfaced that some v1 "cracks" are pavement-edge regions, not
  thin cracks — a v1 quality note, separate from the density fix.
- Artifacts: `reports/runway_defects_full/crack_density_sam.csv`,
  `crack_density_bbox_vs_mask.png`, `overlays/`.
