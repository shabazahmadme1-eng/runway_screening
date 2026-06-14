#!/usr/bin/env python3
"""Standalone ONNX inference — edge deployment without ultralytics.

Runs the exported yolov8m_v1.onnx under onnxruntime with letterbox
preprocessing and YOLOv8 postprocessing (NMS). Depends only on
onnxruntime, numpy and opencv, so it drops onto an edge device cleanly.

  python src/infer_onnx.py --model weights/yolov8m_v1.onnx --image frames/frame_000273.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = [
    "crack", "spalling", "fod", "faded_paint_marking", "band_joint",
    "gap_vegetation", "aged_surface", "repair_patch", "weathered_surface",
    "surface_discoloration", "paint_marking", "faded_surface_marking",
]


def letterbox(im, size):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="weights/yolov8m_v1.onnx")
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default=None, help="annotated output path")
    args = p.parse_args()

    im0 = cv2.imread(args.image)
    H, W = im0.shape[:2]
    img, r, padx, pady = letterbox(im0, args.imgsz)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]  # [1, 4+nc, N]
    out = out[0].T                                               # [N, 4+nc]
    scores = out[:, 4:].max(1)
    classes = out[:, 4:].argmax(1)
    keep = scores >= args.conf
    out, scores, classes = out[keep], scores[keep], classes[keep]

    # xywh (letterbox space) -> xyxy (original image)
    cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    boxes = np.stack([(cx - bw / 2 - padx) / r, (cy - bh / 2 - pady) / r,
                      (cx + bw / 2 - padx) / r, (cy + bh / 2 - pady) / r], 1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    final = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        for k in nms(boxes[idx], scores[idx], args.iou):
            final.append(idx[k])

    for i in final:
        x1, y1, x2, y2 = boxes[i].astype(int)
        label = f"{CLASS_NAMES[classes[i]]} {scores[i]:.2f}"
        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(im0, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        print(label, (x1, y1, x2, y2))
    print(f"{len(final)} detections")
    out_path = args.out or (Path(args.image).stem + "_onnx.jpg")
    cv2.imwrite(out_path, im0)
    print(f"annotated -> {out_path}")


if __name__ == "__main__":
    main()
