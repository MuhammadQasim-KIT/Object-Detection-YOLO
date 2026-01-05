"""
Train YOLO (CPU-only) with Early Stopping on a Roboflow dataset.

Example:
  python scripts/train.py --data data/PPE/data.yaml

Early stopping:
- Training stops if validation mAP does not improve for `patience` epochs.
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO with early stopping (CPU-only).")

    # Dataset / model
    parser.add_argument("--data", type=str, default="data/PPE/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model checkpoint (yolo11s.pt, yolo11m.pt)")
    
    # Training params
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (early stopping may stop earlier)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (CPU: keep small)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (set 0 if issues)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers (useful for heavy models on CPU)")
    
    # Output
    parser.add_argument("--project", type=str, default="runs", help="Output project directory")
    parser.add_argument("--name", type=str, default="ppe_yolo11s_cpu", help="Run name")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n[INFO] Training configuration")
    print(f"  Model     : {args.model}")
    print(f"  Dataset   : {args.data}")
    print(f"  Img size  : {args.imgsz}")
    print(f"  Epochs    : {args.epochs} (early stopping patience={args.patience})")
    print(f"  Batch     : {args.batch}")
    print(f"  Workers   : {args.workers}")
    print(f"  Freeze    : {args.freeze}")
    print(f"  Device    : CPU\n")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,   # ✅ EARLY STOPPING
        device="cpu",
        pretrained=True,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # Optional: freeze backbone layers (useful for yolo11m on CPU)
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    model.train(**train_kwargs)

    print("\n[SUCCESS] Training complete.")
    print(f"Best weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
