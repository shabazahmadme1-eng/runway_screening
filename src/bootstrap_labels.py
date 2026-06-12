#!/usr/bin/env python3
"""Step 3 — AI bootstrapping.

Runs a lightweight YOLOv8 Nano model over the unlabelled drone frames at a
low confidence threshold and writes YOLO-format pseudo-label .txt files next
to the frames, ready for upload to an annotation tool (CVAT / Roboflow) for
manual refinement.

Recommended weights: a nano briefly fine-tuned on the merged RDD2022 data
(see README step 3) so the detector already knows crack/spalling. With stock
COCO weights use --map-to-fod so generic object hits land in the fod class.
"""

import argparse
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--frames", type=Path, required=True,
                   help="directory of unlabelled drone frames "
                        "(e.g. datasets/master/staging/drone_frames)")
    p.add_argument("--out", type=Path, default=None,
                   help="label output dir (default: alongside the frames)")
    p.add_argument("--conf", type=float, default=0.10,
                   help="low threshold on purpose — recall over precision; "
                        "false positives get deleted during refinement")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--map-to-fod", action="store_true",
                   help="remap every detected class to fod (id 2); for use "
                        "with COCO-pretrained weights")
    args = p.parse_args()

    frames = sorted(f for f in args.frames.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames found in {args.frames}")
    out = args.out or args.frames
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    n_boxes = 0
    for f in tqdm(frames, desc="pseudo-labelling"):
        result = model.predict(f, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lines = []
        for box in result.boxes:
            cls = 2 if args.map_to_fod else int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out / f.with_suffix(".txt").name).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
    print(f"wrote pseudo-labels for {len(frames)} frames "
          f"({n_boxes} boxes @ conf>={args.conf}) -> {out}")


if __name__ == "__main__":
    main()
