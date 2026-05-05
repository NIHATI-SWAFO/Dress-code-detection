# Enhanced UWear — Dress Code Detection System

> Module 1 of the SWAFO Violation Management System  
> De La Salle University-Dasmariñas (DLSU-D)

## Project Structure

```text
dresscode_detection/
├── contexts/                   # Research documentation & training plan
│   ├── research.md             # Full research context, gaps, and policy alignment
│   └── plan.md                 # Technical training plan
├── dataset/
│   ├── README.md               # Dataset description & class schema
│   ├── full/                   # Full merged dataset (204,927 images — master copy)
│   │   ├── data.yaml
│   │   ├── images/train/
│   │   └── labels/train/
│   └── balanced/               # Subsampled + split dataset (94,300 images — for training)
│       ├── data.yaml
│       ├── images/
│       │   ├── train/          # 66,003 images (70%)
│       │   ├── val/            # 18,854 images (20%)
│       │   └── test/           #  9,443 images (10%)
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── scripts/                    # Data preparation scripts
│   ├── subsample_and_split.py  # Source-aware subsampling + stratified split
│   ├── count_classes.py        # Class distribution counter
│   ├── verify_final_counts.py  # Post-merge verification
│   └── remap_and_merge.py      # Original dataset merge script
├── training/                   # Training & evaluation scripts
│   ├── train_dev.py            # Development run (yolo11s, ~10-12 hrs)
│   ├── train_final.py          # Final thesis run (yolo11m, ~20-25 hrs)
│   └── evaluate.py             # Metrics report vs UWear baseline
├── models/                     # Trained model weights (after training)
└── results/                    # Training outputs (curves, plots, metrics)
```

## Dataset Overview

- **14 classes** covering uniform, civilian, footwear, and prohibited attire
- **Full dataset**: 204,927 images / 315,597 annotations
- **Balanced training set**: 94,300 images / 129,066 annotations (subsampled for class balance)
- **Split**: 70% train / 20% val / 10% test (stratified)

## Environment Setup

A Python 3.13 virtual environment with CUDA-enabled PyTorch is already set up at `.venv/`.  
**Always use this venv when running any script** — it has the GPU drivers wired in.

```bash
# Activate the virtual environment (required before running any script)
.venv\Scripts\activate
```

## Quick Start

```bash
# Make sure venv is active first (.venv\Scripts\activate)

# 2. Run development training (~10-12 hrs on GTX 1660 SUPER)
python training/train_dev.py

# 3. Evaluate results
python training/evaluate.py --model models/yolo11s_dev_best.pt

# 4. Run final training (~20-25 hrs)
python training/train_final.py

# 5. Final evaluation
python training/evaluate.py --model models/yolo11m_final_best.pt
```

## Targets (Must Beat UWear Baseline)

| Metric | UWear (2024) | Target |
| :--- | ---: | :--- |
| mAP@0.5 | 0.7883 | > 0.85 |
| Precision | 0.7336 | > 0.80 |
| Recall | 0.7881 | > 0.85 |
