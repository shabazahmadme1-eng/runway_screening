# Runway Screening — Edge ML Pipeline (Sequetrics)

Custom, edge-deployable YOLOv8 pipeline that detects runway surface
degradation (cracks, spalling) and Foreign Object Debris (FOD) in
high-resolution drone imagery (1.25 mm GSD, frames extracted at 5 FPS).

## Class schema

Classes 0–2 are the original spec; 3–11 come from the client annotation
Q&A (see [docs/annotation_guide.md](docs/annotation_guide.md) for when to
use each).

| id | name                  | source                                          |
|----|-----------------------|-------------------------------------------------|
| 0  | crack                 | RDD2022 D00 / D10 / D20 + verified drone labels |
| 1  | spalling              | RDD2022 D40 (potholes ≈ spalling proxy) + drone |
| 2  | fod                   | drone only                                      |
| 3  | faded_paint_marking   | RDD2022 D43 / D44 (blurred markings) + drone    |
| 4  | band_joint            | drone only                                      |
| 5  | gap_vegetation        | drone only                                      |
| 6  | aged_surface          | drone only                                      |
| 7  | repair_patch          | RDD2022 "Repair" + drone                        |
| 8  | weathered_surface     | drone only                                      |
| 9  | surface_discoloration | drone only                                      |
| 10 | paint_marking         | drone only                                      |
| 11 | faded_surface_marking | drone only                                      |

## Pipeline

1. **Data extraction** — `src/extract_frames.py` parses the raw DJI `.MP4`
   into a structured folder of frames at 5 FPS. *(completed)*
2. **Dataset merging** — `src/merge_datasets.py` pulls RDD2022 from Kaggle,
   converts its PASCAL VOC XML annotations to YOLO format with the class
   mapping above, and combines everything into a master YOLO directory
   alongside the drone frames. *(current step)*
3. **AI bootstrapping** — `src/bootstrap_labels.py` runs a YOLOv8 Nano model
   over the unlabelled drone frames at a low confidence threshold to generate
   pseudo-label bounding boxes.
4. **Ground-truth refinement** — upload the pseudo-labels to an annotation
   tool (CVAT / Roboflow), delete false positives, categorise true defects,
   then re-run the merge with `--drone-labels` to fold them into the master
   set.
5. **Production fine-tuning** — `src/train.py` trains YOLOv8 Medium for
   150+ epochs on the merged, verified dataset so it masters the grooved
   runway texture.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle API credentials are required for the download step: place your
`kaggle.json` in `~/.kaggle/` (chmod 600) or export `KAGGLE_USERNAME`
and `KAGGLE_KEY`.

## Usage

```bash
# 1. (already done) extract frames at 5 FPS
python src/extract_frames.py --video raw/DJI_0001.MP4 --out data/drone_frames --fps 5

# 2. download RDD2022, convert VOC -> YOLO, merge with drone frames
python src/merge_datasets.py \
    --drone-frames data/drone_frames \
    --out datasets/master

# 3. bootstrap: quick-train a nano on the RDD portion, then pseudo-label
python src/train.py --model yolov8n.pt --data datasets/master/data.yaml \
    --epochs 30 --imgsz 640 --batch 16 --name nano_bootstrap
python src/bootstrap_labels.py \
    --weights runs/detect/nano_bootstrap/weights/best.pt \
    --frames datasets/master/staging/drone_frames \
    --conf 0.10

# 3b. visual sanity check of the pseudo-labels (writes annotated previews)
python src/preview_labels.py --frames frames --labels pseudo_labels --out previews

# 4. package frames+pseudo-labels for annotation, refine, then re-merge
python src/export_for_annotation.py --format cvat --zip   # -> export/cvat.zip
#   ... refine in CVAT/Roboflow (see docs/annotation_guide.md), export to
#   data/drone_labels_verified/, then:
python src/merge_datasets.py \
    --drone-frames data/drone_frames \
    --drone-labels data/drone_labels_verified \
    --out datasets/master

# 5. production fine-tune
python src/train.py --model yolov8m.pt --data datasets/master/data.yaml --epochs 150 --imgsz 1280 --name runway_m_v1
```

### Fast path — v1 model with no manual annotation

To skip the CVAT/Roboflow refinement and train a usable v1 straight away,
auto-refine the pseudo-labels (precision-tuned re-inference + NMS) and feed
those "silver" labels into the merge. Only the four RDD-backed classes
appear (crack, spalling, faded_paint_marking, repair_patch); the eight
drone-only classes stay undetectable until a human labels them.

```bash
# auto-refine the conf-0.10 bootstrap labels into clean silver labels
python src/auto_refine.py --out pseudo_labels_silver --conf 0.40

# merge RDD2022 + the silver drone labels (only the 110 positive frames fold
# in; the rest are left out rather than trusted as background)
python src/merge_datasets.py --drone-frames frames \
    --drone-labels pseudo_labels_silver --out datasets/master

# train YOLOv8m. On 6 GB VRAM use --imgsz 960; for full 1280 use a cloud GPU
python src/train.py --model yolov8m.pt --data datasets/master/data.yaml \
    --epochs 150 --imgsz 960 --batch -1 --name runway_m_v1
```

### Adding gap_vegetation for free (5-class v1)

Classical CV can auto-label one more class without any manual work —
`gap_vegetation` (green is chromatically distinct from grey asphalt). The
other drone-only classes flood on grooved-runway texture; see
[docs/auto_labeling_findings.md](docs/auto_labeling_findings.md). The
`drone_labels_v1/` set already combines the silver crack labels with these
vegetation labels:

```bash
# (regenerate if needed)
python src/cv_autolabel.py --frames frames --out cv_labels_green --classes 5 9

# train on RDD2022 + crack + gap_vegetation drone labels
python src/merge_datasets.py --drone-frames frames \
    --drone-labels drone_labels_v1 --out datasets/master
python src/train.py --model yolov8m.pt --data datasets/master/data.yaml \
    --epochs 150 --imgsz 960 --batch -1 --name runway_m_v1
```

### Inference, reporting & edge deployment

The trained model is `weights/yolov8m_v1.pt` (150 epochs, YOLOv8m @ 960px,
peak mAP50 0.693). It detects 5 of the 12 schema classes — crack, spalling,
faded_paint_marking, repair_patch, gap_vegetation — the rest have no
training data.

```bash
# scan every frame -> per-detection + per-frame CSVs + printed inventory
python src/runway_report.py --weights weights/yolov8m_v1.pt --conf 0.20

# export for the edge (ONNX fp32 ~104 MB; --half for ~52 MB fp16)
python src/export_edge.py --half
# build TensorRT *on the Jetson/target*, not here:
python src/export_edge.py --format engine --half

# standalone inference — onnxruntime only, no ultralytics (edge-ready)
python src/infer_onnx.py --model weights/yolov8m_v1.onnx \
    --image frames/frame_000273.jpg --conf 0.25
```

### Extra transfer-learning datasets (improve crack recall + add FOD)

`merge_datasets.py` can fold in three more public datasets that better match
the top-down runway domain than street-level RDD2022 (see
[docs/dataset_research.md](docs/dataset_research.md) for the full evaluation).
Download/extract each, then point the merge at it:

| dataset | download | view | maps into |
|---------|----------|------|-----------|
| UAV-PDD2023 | Zenodo `10.5281/zenodo.8429208` | top-down 30 m | crack, repair_patch, spalling |
| HighRPD | Mendeley `10.17632/sywswj7djj.1` | top-down 50 m | crack, spalling |
| FOD-A | github.com/FOD-UNOmaha/FOD-data | runway/taxiway | fod (all 31 → fod) |

```bash
python src/merge_datasets.py \
    --drone-frames frames --drone-labels drone_labels_v1 \
    --uav-pdd-dir data/uav_pdd2023 \
    --highrpd-dir data/highrpd \
    --fod-a-dir   data/fod_a \
    --out datasets/master
# then train as before (now with crack-rich top-down data + a real fod class)
python src/train.py --model yolov8m.pt --data datasets/master/data.yaml \
    --epochs 150 --imgsz 960 --batch -1 --workers 4 --name runway_m_v2
```

HighRPD ships YOLO indices 0/1/2 = line/block/pothole; override with the
`HIGHRPD_INDEX_MAP` in the script if a mirror reorders them.

## Master dataset layout

```
datasets/master/
├── data.yaml                # YOLO data config (nc=12)
├── images/{train,val}/      # RDD2022 + verified drone frames
├── labels/{train,val}/      # YOLO txt labels
└── staging/drone_frames/    # unlabelled drone frames awaiting pseudo-labels
```

## Hardware notes (6 GB VRAM class GPUs, e.g. RTX 3050 Laptop)

- **Nano bootstrap**: train at `--imgsz 640 --batch 16`. RDD2022 images are
  small; 640 is the standard for that dataset and fits comfortably in 6 GB.
- **Pseudo-label inference** still runs at `--imgsz 1280` (inference is far
  lighter than training, and the drone frames need the resolution).
- **Production YOLOv8m**: 1280 training does not fit in 6 GB. Use
  `--imgsz 960 --batch -1` (auto) and expect small batches — ultralytics
  accumulates gradients to a nominal batch of 64 internally, so small
  batches stay stable. For full 1280 training rent a single A10/T4-class
  cloud GPU or use Colab Pro.

## Notes

- The Kaggle mirror defaults to [`codderboy/rdd2022`](https://www.kaggle.com/datasets/codderboy/rdd2022);
  override with `--kaggle-dataset` if you prefer another mirror or the
  official FigShare release (point `--rdd-dir` at an already-extracted copy
  to skip the download entirely).
- RDD2022 `test` splits ship without annotations and are skipped.
- Non-damage RDD classes (D43/D44/D50 — crosswalk blur, lane blur,
  manholes) are dropped; images left with no boxes are skipped unless
  `--keep-negatives` is passed.
