# Auto-labeling the drone-only classes — what worked, what didn't

We tested whether the 8 drone-only classes (which have zero training data)
could be labelled automatically instead of by hand in CVAT. Two approaches,
run on the real 274 drone frames:

## 1. Open-vocabulary detection (YOLO-World) — failed

Prompted `yolov8s-worldv2` with "grass", "vegetation", "debris", "rock",
"white paint spot" at conf 0.02. **Detected nothing** on the drone frames.
These models are trained on web/ground-level photos and do not transfer to
a top-down 1.25 mm-GSD aerial view; they also look for discrete objects,
not texture regions like a grass verge. Dead end for this domain.

## 2. Classical computer vision — one solid win, the rest flood

Color/geometry rules (`src/cv_autolabel.py`). The core obstacle: a grooved
runway is, to an edge/brightness detector, a dense field of parallel dark
lines and bright aggregate specks — so naive thresholds produced **106,748
boxes** across 274 frames (≈260 paint specks per frame). After adding a
local-contrast gate and fixing a yellow-line-as-green hue bug:

| class | result | verdict |
|-------|--------|---------|
| gap_vegetation | 188 boxes / 67 frames, on the verge + edge growth | ✅ reliable |
| surface_discoloration | 1 box — barely present in this footage | ➖ no signal here |
| paint_marking | 5,710 boxes — fires on aggregate + faded line edges | ❌ floods |
| fod | 1,822 boxes — some real debris, lots of shadow/texture FPs | ❌ untrustworthy |
| band_joint | dropped — runway grooves are indistinguishable from joints | ❌ not attempted |

**Conclusion.** Classical CV reliably adds exactly **one** class here,
`gap_vegetation`, because green is the only thing chromatically distinct
from grey asphalt. Everything brightness- or edge-based loses to the
grooved texture. The honest limit of label-free methods on this data is a
**5-class v1**: the 4 RDD-backed classes + gap_vegetation.

The remaining classes (fod, paint_marking, band_joint, aged_surface,
weathered_surface, faded_surface_marking) need human annotation or a
trained model — there is no free lunch for them on this imagery.

## Reproduce

```bash
python src/cv_autolabel.py --frames frames --out cv_labels            # all (noisy)
python src/cv_autolabel.py --frames frames --out cv_labels_green --classes 5 9
python src/preview_labels.py --frames frames --labels cv_labels_green --out previews_green
```
