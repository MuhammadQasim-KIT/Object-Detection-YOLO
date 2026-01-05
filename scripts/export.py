import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export a trained YOLO model (e.g., ONNX).")
    p.add_argument("--weights", type=str, default="runs/ppe_yolo11s_cpu/weights/best.pt", help="Path to best.pt")
    p.add_argument("--format", type=str, default="onnx", help="Export format: onnx, torchscript, openvino, etc.")
    p.add_argument("--imgsz", type=int, default=512, help="Image size used for export")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz)
    print(f"Export complete: format={args.format}")


if __name__ == "__main__":
    main()
