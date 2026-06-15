# Adding the FOD class (the safety-critical one)

FOD (Foreign Object Debris) has no training data and can't be auto-labelled
on this texture (the CV anomaly detector floods on aggregate/shadows). 

## 1. Label a focused shortlist (~40 frames, ~1 hour)

`fod_label_shortlist.txt` ranks the frames most likely to contain debris
(by anomaly density). Label FOD on those first — that's enough to teach the
class. Build a Roboflow/CVAT package limited to them:

```bash
python src/export_for_annotation.py --format roboflow --out export/fod
# then keep only the shortlisted frames, or just upload frames/ and label
# the ones in fod_label_shortlist.txt
```

In the tool, draw a tight box around each genuine loose object on the
pavement — stones, bolts, litter, broken pieces — and label it `fod`
(class id 2). Ignore painted markings, cracks, vegetation, and surface
stains. Aim for 30-50 boxes total; quality over quantity.

## 2. Fold the FOD labels in and retrain

Export the verified labels to `data/fod_labels/` (YOLO format), merge them
with the existing v1 drone labels, and retrain:

```bash
# combine v1 labels (crack + vegetation) with the new FOD labels per frame
python - <<'PY'
from pathlib import Path
out = Path("drone_labels_v2"); out.mkdir(exist_ok=True)
stems = {p.stem for d in ["drone_labels_v1","data/fod_labels"] for p in Path(d).glob("*.txt")}
for s in stems:
    lines = []
    for d in ["drone_labels_v1","data/fod_labels"]:
        p = Path(d)/f"{s}.txt"
        if p.exists():
            lines += [l for l in p.read_text().splitlines() if l.strip()]
    (out/f"{s}.txt").write_text("\n".join(lines)+"\n")
print("wrote", len(stems), "label files -> drone_labels_v2")
PY

python src/merge_datasets.py --drone-frames frames \
    --drone-labels drone_labels_v2 --out datasets/master
python src/train.py --model weights/yolov8m_v1.pt \
    --data datasets/master/data.yaml --epochs 80 --imgsz 960 \
    --batch -1 --workers 4 --name runway_m_v2
```

Starting from `yolov8m_v1.pt` (not `yolov8m.pt`) warm-starts from the v1
model, so 80 epochs is enough to learn the new class without retraining
from scratch.
