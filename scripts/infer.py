import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained YOLO model and save predictions.")
    p.add_argument("--weights", type=str, default="runs/ppe_yolo11s_cpu/weights/best.pt", help="Path to best.pt")
    p.add_argument("--source", type=str, default="data/PPE/test/images", help="Image/dir/video/webcam (0)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=512, help="Image size")
    p.add_argument("--project", type=str, default="runs", help="Output folder root")
    p.add_argument("--name", type=str, default="ppe_preds", help="Output run name")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu",
        save=True,
        project=args.project,
        name=args.name,
        verbose=False,
    )

    out_dir = Path(args.project) / args.name
    print(f"Saved predictions to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
