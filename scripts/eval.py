import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained YOLO model on Roboflow PPE dataset.")
    p.add_argument("--data", type=str, default="data/PPE/data.yaml", help="Path to dataset YAML")
    p.add_argument("--weights", type=str, default="runs/ppe_yolo11s_cpu/weights/best.pt", help="Path to best.pt")
    p.add_argument("--imgsz", type=int, default=512, help="Image size")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device="cpu",
        verbose=True,
    )

    # Ultralytics prints a summary; below is a small extra print
    try:
        print("\n--- Metrics summary ---")
        print("mAP50-95:", float(metrics.box.map))
        print("mAP50    :", float(metrics.box.map50))
        print("mAP75    :", float(metrics.box.map75))
    except Exception:
        pass


if __name__ == "__main__":
    main()
