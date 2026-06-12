#!/usr/bin/env python3
"""Step 1 — Data extraction (kept for reproducibility; already run).

Parses the raw DJI .MP4 drone video into a structured folder of image
frames at a fixed sampling rate (default 5 FPS).
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True, help="input .MP4")
    p.add_argument("--out", type=Path, required=True, help="frame output dir")
    p.add_argument("--fps", type=float, default=5.0, help="frames per second to keep")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / args.fps))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    saved = 0
    for i in tqdm(range(total), desc="extracting"):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(args.out / f"{stem}_f{i:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
    cap.release()
    print(f"saved {saved} frames ({src_fps:.1f} src FPS, every {step}th frame) -> {args.out}")


if __name__ == "__main__":
    main()
