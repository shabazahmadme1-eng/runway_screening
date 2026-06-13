#!/usr/bin/env python3
"""Step 5 — Production fine-tuning (also used for the quick nano bootstrap).

Thin wrapper around ultralytics with defaults tuned for fine-grained runway
surface texture: large input size to preserve the 1.25 mm GSD detail, and
mosaic disabled for the final epochs so the model settles on real groove
geometry rather than stitched composites.
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8m.pt",
                   help="base weights: yolov8n.pt for the bootstrap pass, "
                        "yolov8m.pt for production (default)")
    p.add_argument("--data", default="datasets/master/data.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    p.add_argument("--name", default="runway")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8,
                   help="dataloader workers; drop to 4 on <=16 GB RAM "
                        "machines to avoid the close_mosaic rebuild OOM")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted run from its last.pt")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        close_mosaic=max(10, args.epochs // 10),
        patience=50,
    )


if __name__ == "__main__":
    main()
