"""
Script 05 — Export trained model to deployment formats.

Supported formats:
  onnx      — Cross-platform, runs on CPU/GPU with ONNX Runtime
  tflite    — TensorFlow Lite for Android / Raspberry Pi
  coreml    — Apple CoreML for iOS / macOS
  openvino  — Intel OpenVINO for edge devices
  engine    — TensorRT for NVIDIA Jetson (requires TensorRT installation)

Usage:
  python scripts/05_export_model.py --format onnx
  python scripts/05_export_model.py --format tflite
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
IMG_SIZE = 640


def export(format: str, model_path: str = MODEL_PATH, half: bool = False):
    model = YOLO(model_path)
    print(f"Exporting {model_path} → format: {format}")

    exported = model.export(
        format=format,
        imgsz=IMG_SIZE,
        half=half,          # FP16 quantisation (GPU only)
        simplify=True,      # Simplify ONNX graph
        dynamic=False,      # Fixed batch size (better for edge)
    )

    print(f"\nExported to: {exported}")
    print("\nUsage with exported model:")

    if format == "onnx":
        print("""
import onnxruntime as ort
import numpy as np, cv2

session = ort.InferenceSession("models/best.onnx")
img = cv2.imread("plate.jpg")
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)[np.newaxis] / 255.0
outputs = session.run(None, {"images": img.astype(np.float32)})
""")
    elif format == "tflite":
        print("""
import tflite_runtime.interpreter as tflite
interp = tflite.Interpreter("models/best_float32.tflite")
interp.allocate_tensors()
""")

    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "tflite", "coreml", "openvino", "engine"])
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--half", action="store_true",
                        help="FP16 quantisation (requires GPU)")
    args = parser.parse_args()

    export(args.format, args.model, args.half)
