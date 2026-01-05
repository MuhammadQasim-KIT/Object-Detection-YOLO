"""
Download Roboflow dataset (Option A) and fix paths for Ultralytics YOLO.

USAGE (recommended):
  set ROBOFLOW_API_KEY=...
  python scripts/download_dataset.py --workspace <ws> --project <proj> --version <ver> --format yolov11

OR:
  python scripts/download_dataset.py --api-key YOUR_KEY --workspace <ws> --project <proj> --version <ver> --format yolov11

After running, dataset will be at:
  data/PPE/data.yaml
"""

import argparse
import os
from pathlib import Path
import yaml
from roboflow import Roboflow


def parse_args():
    p = argparse.ArgumentParser(description="Download a Roboflow dataset and prepare it for Ultralytics YOLO.")
    p.add_argument("--api-key", type=str, default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    p.add_argument("--workspace", type=str, default=None, help="Roboflow workspace name (or ROBOFLOW_WORKSPACE env var)")
    p.add_argument("--project", type=str, default=None, help="Roboflow project name (or ROBOFLOW_PROJECT env var)")
    p.add_argument("--version", type=int, default=None, help="Roboflow dataset version (or ROBOFLOW_VERSION env var)")
    p.add_argument("--format", type=str, default="yolov11", help="Export format: yolov11 / yolov8 / yolov5")
    p.add_argument("--out-dir", type=str, default="data/PPE", help="Output directory (dataset will be placed here)")
    return p.parse_args()


def resolve_args(args):
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    workspace = args.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project = args.project or os.getenv("ROBOFLOW_PROJECT")
    version = args.version or os.getenv("ROBOFLOW_VERSION")

    if version is not None and isinstance(version, str):
        version = int(version)

    missing = []
    if not api_key:
        missing.append("api-key / ROBOFLOW_API_KEY")
    if not workspace:
        missing.append("workspace / ROBOFLOW_WORKSPACE")
    if not project:
        missing.append("project / ROBOFLOW_PROJECT")
    if not version:
        missing.append("version / ROBOFLOW_VERSION")

    if missing:
        raise ValueError(
            "Missing required inputs: " + ", ".join(missing) +
            "\nGet workspace/project/version from Roboflow's Python download snippet."
        )

    return api_key, workspace, project, version


def fix_yaml_paths(yaml_path: Path):
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # Roboflow sometimes exports ../train/images etc.
    # For our folder structure (data/PPE/{train,valid,test}), these must be:
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"

    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[OK] Fixed dataset paths in: {yaml_path}")


def main():
    args = parse_args()
    api_key, workspace, project_name, version = resolve_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading dataset from Roboflow...")
    print(f"       workspace={workspace}, project={project_name}, version={version}, format={args.format}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)

    # Download into out_dir
    dataset = project.version(version).download(args.format, location=str(out_dir))

    yaml_path = out_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found after download at: {yaml_path}")

    fix_yaml_paths(yaml_path)

    print("\n[SUCCESS] Dataset ready!")
    print(f"Dataset location: {out_dir.resolve()}")
    print(f"Dataset YAML     : {yaml_path.resolve()}")
    print("\nNext commands:")
    print(f"  python scripts/check_dataset.py --data {yaml_path.as_posix()}")
    print(f"  python scripts/train.py --data {yaml_path.as_posix()}")


if __name__ == "__main__":
    main()
