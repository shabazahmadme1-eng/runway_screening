#!/usr/bin/env python3
"""Export the production model to an edge-deployable format.

ONNX is portable and runs under onnxruntime on CPU/GPU/NPU. For NVIDIA edge
devices (Jetson) build the TensorRT engine *on the target* — engines are
hardware-specific and must not be cross-built.

  python src/export_edge.py                      # ONNX fp32 (~104 MB)
  python src/export_edge.py --half               # ONNX fp16 (~52 MB)
  python src/export_edge.py --format engine --half  # TensorRT (run on target)
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/yolov8m_v1.pt")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino"])
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--half", action="store_true", help="fp16 (smaller/faster)")
    args = p.parse_args()

    model = YOLO(args.weights)
    path = model.export(format=args.format, imgsz=args.imgsz,
                        half=args.half, simplify=True,
                        opset=12 if args.format == "onnx" else None)
    print(f"exported -> {path}")


if __name__ == "__main__":
    main()
