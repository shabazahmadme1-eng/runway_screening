#!/usr/bin/env python3
"""Step 5 — Production fine-tuning (also used for the quick nano bootstrap).

Defaults are tuned for fine, thin, top-down runway cracks:

- large input size preserves the 1.25 mm GSD detail (thin cracks vanish when
  downscaled) — the single biggest precision lever; use 1280 on an A100,
  960 on a 6 GB laptop GPU;
- top-down imagery is orientation-invariant, so vertical flips and small
  rotations are *valid* augmentations here (unlike street-level datasets) and
  multiply effective data without distorting labels;
- mosaic is disabled for the final epochs so the model settles on real groove
  geometry and tightens boxes instead of learning stitched composites;
- cosine LR decay for a smoother, higher final-precision convergence.
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8m.pt",
                   help="base weights: yolov8n.pt bootstrap, yolov8m.pt "
                        "production, yolov8l.pt for max capacity on an A100")
    p.add_argument("--data", default="datasets/master/data.yaml")
    p.add_argument("--epochs", type=int, default=100,
                   help="v1 peaked ~100; patience early-stops if it plateaus")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="1280 on A100 for thin cracks; drop to 960 on 6 GB")
    p.add_argument("--batch", type=float, default=-1,
                   help="-1 auto-sizes to VRAM; or a fraction like 0.85")
    p.add_argument("--name", default="runway")
    p.add_argument("--project", default=None,
                   help="output dir for runs; point at a Drive path on Colab "
                        "so checkpoints survive a disconnect and --resume works")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of the training set used per epoch; <1 trades "
                        "a little accuracy for a big speed-up on huge corpora")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8,
                   help="drop to 4 on <=16 GB RAM to avoid the mosaic-close OOM")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted run from its last.pt")
    # --- precision-oriented knobs (sensible defaults; override as needed) ---
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--close-mosaic", type=int, default=20,
                   help="epochs at the end with mosaic off (sharpens boxes)")
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--box", type=float, default=7.5,
                   help="box-loss gain; higher emphasises tight localisation")
    p.add_argument("--degrees", type=float, default=10.0,
                   help="rotation aug — valid for top-down imagery")
    p.add_argument("--flipud", type=float, default=0.5,
                   help="vertical flip prob — valid for top-down imagery")
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--multi-scale", action="store_true",
                   help="train across +/-50%% sizes; robuster, slower")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        fraction=args.fraction,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        lr0=args.lr0,
        box=args.box,
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        scale=args.scale,
        mosaic=args.mosaic,
        multi_scale=args.multi_scale,
        seed=args.seed,
        plots=True,
    )


if __name__ == "__main__":
    main()
