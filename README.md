# Runway Screening — Edge ML Pipeline (Sequetrics)

Custom, edge-deployable YOLOv8 pipeline that detects runway surface
degradation (cracks, spalling) and Foreign Object Debris (FOD) in
high-resolution drone imagery (1.25 mm GSD, frames extracted at 5 FPS).

## Class schema

| id | name     | source                                              |
|----|----------|-----------------------------------------------------|
| 0  | crack    | RDD2022 D00 / D10 / D20 + verified drone labels     |
| 1  | spalling | RDD2022 D40 (potholes ≈ spalling proxy) + drone     |
| 2  | fod      | drone pseudo-labels only (no RDD2022 equivalent)    |

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
python src/train.py --model yolov8n.pt --data datasets/master/data.yaml --epochs 25 --name nano_bootstrap
python src/bootstrap_labels.py \
    --weights runs/detect/nano_bootstrap/weights/best.pt \
    --frames datasets/master/staging/drone_frames \
    --conf 0.10

# 4. refine in CVAT/Roboflow, export YOLO labels, then re-merge
python src/merge_datasets.py \
    --drone-frames data/drone_frames \
    --drone-labels data/drone_labels_verified \
    --out datasets/master

# 5. production fine-tune
python src/train.py --model yolov8m.pt --data datasets/master/data.yaml --epochs 150 --imgsz 1280 --name runway_m_v1
```

## Master dataset layout

```
datasets/master/
├── data.yaml                # YOLO data config (nc=3)
├── images/{train,val}/      # RDD2022 + verified drone frames
├── labels/{train,val}/      # YOLO txt labels
└── staging/drone_frames/    # unlabelled drone frames awaiting pseudo-labels
```

## Notes

- The Kaggle mirror defaults to [`codderboy/rdd2022`](https://www.kaggle.com/datasets/codderboy/rdd2022);
  override with `--kaggle-dataset` if you prefer another mirror or the
  official FigShare release (point `--rdd-dir` at an already-extracted copy
  to skip the download entirely).
- RDD2022 `test` splits ship without annotations and are skipped.
- Non-damage RDD classes (D43/D44/D50 — crosswalk blur, lane blur,
  manholes) are dropped; images left with no boxes are skipped unless
  `--keep-negatives` is passed.
