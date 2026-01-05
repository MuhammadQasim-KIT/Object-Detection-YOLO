# Object Detection with YOLO (Ultralytics)

This project demonstrates an **end-to-end object detection pipeline** using **Ultralytics YOLO** and **PyTorch**.  
It is designed as a **clean, reproducible portfolio project** that covers dataset handling, training, evaluation, inference, and model export.

The project uses a **custom PPE Helmet Detection dataset** (single class: `helmet`) that is **already included** in the repository.

---

## Project Overview

**Task:** Detect safety helmets in images  
**Model:** YOLO (Ultralytics)  
**Framework:** PyTorch  
**Classes:** 1 (`helmet`)  
**Dataset:** PPE Helmet Detection (custom)

This repository is structured so that it can be **cloned and run immediately**, without external dataset downloads or API keys.

---

## Repository Structure

```
Object_Detection/
│
├── data/
│   └── PPE/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── export.py
│   └── check_dataset.py
│
├── assets/
│   └── results/         # Training curves & prediction images (added manually)
│
├── runs/                # YOLO training outputs (auto-generated, not committed)
├── environment.yml
└── README.md
```

---

## Dataset

- The dataset is included locally under `data/PPE`
- No external download or API keys are required
- Paths in `data.yaml` are **relative** for portability

### `data.yaml`
```yaml
path: data/PPE
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['helmet']
```

---

## Environment Setup

This project uses **conda**.

```bash
conda env create -f environment.yml
conda activate yolo
```

---

## Dataset Sanity Check

Before training, verify the dataset:

```bash
python scripts/check_dataset.py --data data/PPE/data.yaml
```

This checks:
- Image–label matching
- Missing files
- Dataset size

---

## Training

```bash
python scripts/train.py --data data/PPE/data.yaml
```

Outputs:
- Training curves
- Validation metrics
- Saved model weights

All results are saved automatically under `runs/`.

---

## Evaluation

```bash
python scripts/eval.py --data data/PPE/data.yaml
```

Evaluates:
- Precision
- Recall
- mAP

---

## Inference

Run inference on sample images:

```bash
python scripts/infer.py
```

Predictions are saved with bounding boxes and confidence scores.

---

## Model Export

Export the trained model for deployment:

```bash
python scripts/export.py
```

Supported formats include:
- ONNX
- TorchScript

---

## Results

> **Note:** The images below are placeholders.  
> After training, copy selected result images from `runs/` into `assets/results/`
> and update or keep the filenames as shown.

### Training Metrics
![Training Curves](assets/results/training_curves.png)

### Confusion Matrix
![Confusion Matrix](assets/results/confusion_matrix.png)

### Sample Prediction
![Sample Prediction](assets/results/sample_prediction.png)

---

## Key Features

- Clean, modular script-based workflow
- Portable dataset configuration
- End-to-end ML pipeline
- Suitable for recruiters and technical reviews

---

## Notes

- Dataset download scripts were intentionally removed to keep the project fully self-contained
- This reflects real-world ML workflows where datasets are versioned locally

---

## Author

(Add your name / GitHub / LinkedIn here)
