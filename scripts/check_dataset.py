import argparse
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Sanity-check YOLO dataset YAML paths.")
    p.add_argument("--data", type=str, default="data/PPE/data.yaml", help="Path to dataset YAML")
    return p.parse_args()


def main():
    args = parse_args()
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = yaml_path.parent

    def check(key):
        v = cfg.get(key)
        if not v:
            print(f"[WARN] '{key}' not found in YAML")
            return
        p = (root / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()
        ok = p.exists()
        print(f"{key}: {v} -> {p}  {'OK' if ok else 'MISSING'}")
        if ok:
            # quick counts
            imgs = list(p.glob("*.*"))
            print(f"  images found: {len(imgs)}")

    check("train")
    check("val")
    check("test")
    print("\nIf any path is MISSING, edit data.yaml to use: train/images, valid/images, test/images")


if __name__ == "__main__":
    main()
