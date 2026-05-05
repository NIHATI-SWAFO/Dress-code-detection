# Enhanced UWear — Dress Code Detection System
> **Full Technical Documentation**
> Institution: De La Salle University-Dasmariñas (DLSU-D)
> Module: SWAFO Violation Management — Module 1
> Date: May 2026

---

## 1. Project Overview

Enhanced UWear is an AI-powered dress code compliance detection system built for DLSU-D. It uses a YOLO11 object detection model trained on 14 garment classes to determine whether a student is compliant with the DLSU-D School Uniform and Dress Code Policy (effective September 1, 2025).

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MODULE 1 PIPELINE                          │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  Raw Image  │───▶│  YOLO11s     │───▶│  Rule Engine    │ │
│  │  Input      │    │  Detection   │    │  (Handbook      │ │
│  │  (camera /  │    │  Model       │    │   Sec. 27.1.2)  │ │
│  │   upload)   │    │  14 classes  │    │                 │ │
│  └─────────────┘    └──────────────┘    └────────┬────────┘ │
│                                                  │          │
│                                           ┌──────▼──────┐   │
│                                           │ Compliance  │   │
│                                           │ JSON Output │   │
│                                           └──────┬──────┘   │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                                      ┌────────────▼───────────┐
                                      │  MODULE 2 — SWAFO      │
                                      │  Web App               │
                                      │  Violation Recording   │
                                      └────────────────────────┘
```

### What It Does

1. **Detects** what a student is wearing (top, bottom, footwear) via object detection
2. **Classifies** each garment into one of 14 categories
3. **Evaluates compliance** against the institutional dress code policy using a rule engine
4. **Outputs** a structured JSON report with violation flags and handbook references
5. **Annotates** the image with bounding boxes and a compliance banner

---

## 2. Prior Work — UWear (2024)

| Field | Detail |
|---|---|
| Full Title | UWear: A Real-Time Dress Code Compliance System for College Students at DLSU-D Using YOLO-v5 |
| Authors | Rafael Jan M. Dancel, Angel Gabriel T. Kahulugan, Caleb Emmanuel C. Viernes |
| Adviser | Rolando B. Barrameda |
| Year | 2024 |
| Model | YOLOv5 |
| Best mAP@0.5 | 0.7883 (Epoch 45) |
| Precision | 0.7336 |
| Recall | 0.7881 |

### UWear Critical Limitations

| Limitation | Impact |
|---|---|
| No uniform-specific detection | Could not distinguish uniform from civilian clothing |
| No policy-aware compliance logic | No concept of uniform day vs. civilian day |
| No year-level awareness | 1st–3rd year vs 4th year rules not implemented |
| No footwear detection | Slippers — a common violation — were invisible |
| Skirt class severely underperformed | Misidentified as background (~0.0 AP) |
| Predates September 2025 policy update | Dataset not aligned with current rules |
| Standalone local app | No integration with any records or enforcement system |

---

## 3. Research Gaps Addressed

| Gap | UWear (2024) | Enhanced UWear (2026) |
|---|---|---|
| Uniform-specific detection | ❌ None | ✅ `uniform_top`, `uniform_bottom` |
| Policy-aware compliance | ❌ None | ✅ UNIFORM_MODE vs CIVILIAN_MODE |
| Uniform day vs civilian day | ❌ Not implemented | ✅ Core feature of rule engine |
| 4th year exception | ❌ Not implemented | ✅ Year-level rule engine |
| Footwear detection | ❌ Excluded | ✅ `footwear_shoes`, `footwear_slippers` |
| Skirt reliability | ❌ Worst class (~0.0 AP) | ✅ 0.8656 AP |
| Prohibited item coverage | ⚠️ Partial (2 classes) | ✅ 5 prohibited classes |
| Sept 2025 policy alignment | ❌ Predates policy | ✅ Fully aligned |
| Violation recording integration | ❌ Standalone | ✅ JSON output feeds SWAFO |

---

## 4. Technology Stack

| Component | Choice | Version |
|---|---|---|
| Detection Model | YOLO11s (Ultralytics) | Ultralytics 8.4.45 |
| Deep Learning Framework | PyTorch | 2.6+ with CUDA |
| Computer Vision | OpenCV | 4.x |
| Language | Python | 3.13 |
| Training GPU | NVIDIA GTX 1660 SUPER | 6 GB VRAM |
| Dataset Format | YOLOv8/YOLO11 native format | — |

### YOLO11s Model Specs

| Property | Value |
|---|---|
| Parameters | 9,418,218 |
| Layers (fused) | 101 |
| GFLOPs | 21.3 |
| Input size | 640 × 640 |
| Inference speed | 7.3 ms/image (GTX 1660 SUPER) |
| Weight file size | 54.3 MB |

---

## 5. Detection Classes — 14-Class Schema

> Male and female uniforms are merged into single functional classes. The model learns the shape and color of the uniform regardless of gender.

| ID | Class Name | Category | Description |
|---|---|---|---|
| 0 | `uniform_top` | Uniform | White polo / fitted blouse with De La Salle embroidery |
| 1 | `uniform_bottom` | Uniform | Light khaki pants, short pants, or skorts |
| 2 | `civilian_top_short_sleeve` | Civilian | Short sleeve shirt, t-shirt, polo |
| 3 | `civilian_top_long_sleeve` | Civilian | Long sleeve shirt, sweatshirt |
| 4 | `civilian_bottom_trousers` | Civilian | Standard pants/trousers (non-ripped) |
| 5 | `civilian_bottom_shorts` | Civilian | Shorts (≤3 inches above kneecap = valid) |
| 6 | `civilian_bottom_skirt` | Civilian | Skirt or dress bottom (≤2 inches above knee = valid) |
| 7 | `footwear_shoes` | Footwear | Sneakers, topsiders, low-cut shoes |
| 8 | `footwear_slippers` | Footwear | Any type of slippers or open sandals |
| 9 | `prohibited_ripped_pants` | Prohibited | Ripped jeans exposing skin |
| 10 | `prohibited_leggings` | Prohibited | Leggings, jeggings, overly tight pants |
| 11 | `prohibited_sleeveless` | Prohibited | Sling tops, tank tops, spaghetti straps |
| 12 | `prohibited_crop_halter` | Prohibited | Crop tops, halter blouses/dresses |
| 13 | `prohibited_midriff_offshoulder` | Prohibited | Midriff tops, off-shoulder, hanging blouses |

---

## 6. Dress Code Policy Reference

### 6.1 Uniform Schedule (September 1, 2025 Policy)

| Year Level | Mon | Tue | Wed | Thu | Fri | Sat |
|---|---|---|---|---|---|---|
| 1st–3rd Year | Uniform | Uniform | Wash Day | Uniform | Uniform | Wash Day |
| 4th Year | Civilian | Civilian | Wash Day | Civilian | Civilian | Wash Day |

> 4th year students may opt to wear uniform on civilian days — this is always compliant.

### 6.2 Type C Uniform (Daily Standard)

| Component | Male | Female |
|---|---|---|
| Top | White Japanese cotton polo with De La Salle embroidery | White fitted blouse with De La Salle embroidery |
| Bottom | Light khaki Dockers twill pants or short pants | Light khaki Dockers twill pants or skorts |
| Footwear | Low-cut shoes with white socks | Low-cut shoes with white socks |

### 6.3 Prohibited Items (Section 27.1.2 — Minor Offense)

| Handbook Code | Prohibited Item | Mapped Class |
|---|---|---|
| 27.1.2.2 | Ripped jeans exposing skin | `prohibited_ripped_pants` |
| 27.1.2.3 | Shorts more than 3 inches above kneecap | `civilian_bottom_shorts` (length check) |
| 27.1.2.4 | Sleeveless blouses / sling tops | `prohibited_sleeveless` |
| 27.1.2.5 | Plunging neckline or backless blouses | Out of scope |
| 27.1.2.6 | Midriffs, hanging blouses, off-shoulder tops | `prohibited_midriff_offshoulder` |
| 27.1.2.7 | Skirts/dresses more than 2 inches above knee | `civilian_bottom_skirt` (length check) |
| 27.1.2.8 | Leggings, jeggings, overly tight clothing | `prohibited_leggings` |
| 27.1.2.9 | Halter blouses, crop tops | `prohibited_crop_halter` |
| 27.1.2.11 | Slippers (except rain/injury — officer judgment) | `footwear_slippers` |

---

## 7. Dataset

### 7.1 Sources

| Source | Classes Contributed |
|---|---|
| DeepFashion2 | `civilian_top_*`, `civilian_bottom_*`, `prohibited_sleeveless` |
| ModaNet | `civilian_top_*`, `civilian_bottom_*` |
| Roboflow School Uniform datasets | `uniform_top`, `uniform_bottom` |
| USTP School Uniform (Philippine) | `uniform_top`, `uniform_bottom` |
| Roboflow Shoes Dataset | `footwear_shoes`, `footwear_slippers` |
| Open Images V7 | `civilian_bottom_*`, `prohibited_*` |
| Roboflow Clothing/Fashion datasets | `prohibited_*` classes |
| Manual collection (Google Images) | `uniform_top`, `uniform_bottom` |

### 7.2 Dataset Statistics

| Property | Value |
|---|---|
| Total raw images | 204,927 |
| Total annotations | 315,597 |
| Balanced training set (after augmentation) | 82,036 images |
| Validation set | 18,854 images |
| Test set | 9,443 images |
| Split ratio | 70/20/10 (train/val/test) |
| Format | YOLOv8 (class cx cy w h) |

### 7.3 Class Imbalance Problem & Solution

The raw dataset had extreme imbalance — uniform classes had ~1,000 samples vs ~71,000 for civilian tops (an 88:1 ratio).

**Solution — Two-stage approach:**

**Stage 1: Subsampling** — Capped dominant civilian classes at 5,000 samples per class to reduce ratio.

**Stage 2: Offline Augmentation** — Boosted 4 minority classes to 5,000 each:

| Class | Before | After | Augmentation Applied |
|---|---|---|---|
| `uniform_top` | 684 | 5,000 | +4,316 (flips, rotation, brightness, contrast) |
| `uniform_bottom` | 565 | 5,000 | +4,435 |
| `footwear_slippers` | 1,269 | 5,000 | +3,731 |
| `prohibited_ripped_pants` | 1,449 | 5,000 | +3,551 |

Training set grew from 66,003 → 82,036 images after augmentation.

### 7.4 Data Quality Fixes

- **2,190 malformed labels** discovered — uniform classes had segmentation polygon format instead of YOLO bounding box format
- Fixed via `scripts/fix_segmentation_labels.py` — converted polygon coordinates to bounding boxes using min/max x,y
- **94,300 images** audited for integrity across all splits

---

## 8. Training Configuration

### 8.1 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Model | yolo11s.pt | Best accuracy-speed balance for GTX 1660 SUPER |
| Epochs | 100 (max) | With early stopping patience=20 |
| Image size | 640 × 640 | Standard YOLO input |
| Batch size | 8 | Maximum for 6 GB VRAM |
| Classification loss weight | 2.0 | Higher weight to penalize misclassification on minority classes |
| Early stopping patience | 20 | Stops if no mAP improvement for 20 epochs |
| Checkpoint interval | Every 10 epochs | Safety checkpoints for long runs |
| Workers | 4 | Data loading parallelism |
| Device | GPU 0 (CUDA) | GTX 1660 SUPER |

### 8.2 Online Augmentation (Built-in YOLO11)

| Augmentation | Setting | Reason |
|---|---|---|
| HSV Hue | 0.015 | Color variation |
| HSV Saturation | 0.7 | Lighting conditions |
| HSV Value | 0.4 | Brightness variation |
| Rotation | ±15° | Camera angle variation |
| Translation | 0.1 | Position variation |
| Scale | 0.5 | Distance variation |
| Horizontal flip | 0.5 | Left/right doesn't matter for clothing |
| Vertical flip | 0.0 | People aren't upside down |
| Mosaic | 1.0 | Combines 4 images — improves small object detection |
| MixUp | 0.1 | Helps overlapping detections |
| Copy-paste | 0.1 | Synthetic augmentation |

### 8.3 Training Duration

| Metric | Value |
|---|---|
| Epochs completed | 57 (paused mid-epoch 58) |
| Total training time | ~48,550 seconds (~13.5 hours) |
| Average time per epoch | ~14.2 minutes |
| GPU utilization | ~100% during training |

---

## 9. Results — Model Performance

### 9.1 Overall Metrics vs UWear Baseline

| Metric | Enhanced UWear | UWear Baseline | Delta | Status |
|---|---|---|---|---|
| **mAP@0.5** | **0.9031** | 0.7883 | **+0.1148** | ✅ PASS |
| **mAP@0.5:0.95** | **0.7316** | Not reported | — | — |
| **Precision** | **0.8560** | 0.7336 | **+0.1224** | ✅ PASS |
| **Recall** | **0.8573** | 0.7881 | **+0.0692** | ✅ PASS |

> **Enhanced UWear beats the UWear baseline by +11.48 percentage points in mAP@0.5.**

### 9.2 Per-Class Average Precision (AP@0.5) — Test Set

| Class | AP@0.5 | Test Images | Test Objects | Status |
|---|---|---|---|---|
| `uniform_top` ★ | **0.9919** | 99 | 106 | 🔥 Excellent |
| `uniform_bottom` ★ | **0.9950** | 77 | 77 | 🔥 Excellent |
| `civilian_top_short_sleeve` | 0.9579 | 1,764 | 1,785 | 🔥 Excellent |
| `civilian_top_long_sleeve` | 0.8794 | 884 | 889 | 🔥 Excellent |
| `civilian_bottom_trousers` | 0.9746 | 1,354 | 1,360 | 🔥 Excellent |
| `civilian_bottom_shorts` | 0.9502 | 940 | 952 | 🔥 Excellent |
| `civilian_bottom_skirt` | 0.8656 | 751 | 756 | 🔥 Excellent |
| `footwear_shoes` | 0.8631 | 608 | 1,236 | 🔥 Excellent |
| `footwear_slippers` ★ | **0.9597** | 176 | 373 | 🔥 Excellent |
| `prohibited_ripped_pants` ★ | **0.8838** | 208 | 208 | 🔥 Excellent |
| `prohibited_leggings` | 0.8843 | 502 | 502 | 🔥 Excellent |
| `prohibited_sleeveless` | 0.9303 | 3,248 | 3,254 | 🔥 Excellent |
| `prohibited_crop_halter` | 0.6570 | 763 | 763 | ⚠️ Weakest |
| `prohibited_midriff_offshoulder` | 0.8507 | 548 | 548 | 🔥 Excellent |

★ = Augmented minority class

> **13 out of 14 classes scored above 0.85 AP.** Only `prohibited_crop_halter` (0.657) underperformed — likely due to visual similarity with other top garments.

### 9.3 Augmented Class Performance

All 4 augmented minority classes achieved excellent results:

| Class | Pre-Augmentation Count | Post-Augmentation Count | Final AP@0.5 |
|---|---|---|---|
| `uniform_top` | 684 | 5,000 | **0.9919** |
| `uniform_bottom` | 565 | 5,000 | **0.9950** |
| `footwear_slippers` | 1,269 | 5,000 | **0.9597** |
| `prohibited_ripped_pants` | 1,449 | 5,000 | **0.8838** |

### 9.4 Inference Speed

| Stage | Time per Image |
|---|---|
| Preprocessing | 1.1 ms |
| Inference | 7.3 ms |
| Postprocessing | 0.9 ms |
| **Total** | **~9.3 ms (~107 FPS)** |

### 9.5 Comparison with UWear's Weak Classes

| Class | UWear AP | Enhanced UWear AP | Improvement |
|---|---|---|---|
| Skirt | ~0.0 (misidentified as background) | **0.8656** | ✅ Fully resolved |
| Uniform top | N/A (not detected) | **0.9919** | ✅ New capability |
| Uniform bottom | N/A (not detected) | **0.9950** | ✅ New capability |
| Footwear | N/A (not detected) | **0.8631–0.9597** | ✅ New capability |

---

## 10. Compliance Rule Engine

### 10.1 UNIFORM_MODE (Mon/Tue/Thu/Fri — 1st–3rd Year)

```
COMPLIANT if:
  ✅ uniform_top detected
  ✅ uniform_bottom detected
  ✅ footwear_shoes detected
  ✅ No prohibited class detected

NON-COMPLIANT flags:
  ❌ uniform_top not detected       → "Not in proper uniform top"
  ❌ uniform_bottom not detected    → "Not in proper uniform bottom"
  ❌ civilian top/bottom detected   → "Not in uniform on uniform day"
  ❌ footwear_slippers detected     → "Slippers prohibited (Sec. 27.1.2.11)"
  ❌ Any prohibited class detected  → specific violation flag with handbook ref
```

### 10.2 CIVILIAN_MODE (Wed/Sat All + 4th Year All)

```
COMPLIANT if:
  ✅ civilian_top (any) detected
  ✅ civilian_bottom (any) detected
  ✅ footwear_shoes detected
  ✅ No prohibited class detected
  [uniform detected on civilian day] → STILL COMPLIANT (opt-in allowed)

NON-COMPLIANT flags:
  ❌ footwear_slippers detected     → "Slippers prohibited (Sec. 27.1.2.11)"
  ❌ Any prohibited class detected  → specific violation flag with handbook ref
```

### 10.3 Auto-Mode Detection

The system automatically determines the correct mode based on:
- **Current day of the week** (Mon–Sat schedule)
- **Student year level** (1–3 = may require uniform, 4 = always civilian)

### 10.4 Exceptions (Not Automated — Officer Override)

- NSTP/PE shirt on NSTP/PE days
- PE shorts within PE class area
- Approved written exception from OSS Dean or SWAFO Director
- Slippers during heavy rain or injury

---

## 11. Project Structure

```
dresscode_detection/
├── dataset/
│   └── balanced/
│       ├── images/
│       │   ├── train/          # 82,036 images (after augmentation)
│       │   ├── val/            # 18,854 images
│       │   └── test/           # 9,443 images
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml           # Class definitions + paths
│
├── inference/
│   └── detect.py               # Standalone detection + compliance module
│
├── training/
│   ├── train_dev.py            # Dev training script (yolo11s)
│   ├── train_final.py          # Final training script (yolo11m)
│   ├── evaluate.py             # Evaluation + metrics report generator
│   └── dashboard.py            # Live training dashboard (Chart.js)
│
├── scripts/
│   ├── remap_and_merge.py      # Dataset merge + class remapping
│   ├── subsample_and_split.py  # Stratified subsampling + train/val/test split
│   ├── augment_minority_classes.py  # Offline augmentation for minority classes
│   ├── validate_dataset.py     # Label format validation
│   ├── fix_segmentation_labels.py   # Polygon → bbox conversion
│   ├── count_classes.py        # Class distribution counter
│   ├── verify_final_counts.py  # Post-split verification
│   └── visualize_samples.py    # Sample image visualization
│
├── results/
│   ├── enhanced-uwear-dev/     # Training outputs
│   │   ├── weights/
│   │   │   ├── best.pt         # Best model (54.3 MB)
│   │   │   ├── last.pt         # Last checkpoint
│   │   │   └── epoch{N}.pt     # Periodic checkpoints
│   │   └── results.csv         # Per-epoch metrics
│   ├── evaluation_reports/     # Test set evaluation results
│   └── detections/             # Inference output (annotated images + JSON)
│
├── contexts/
│   ├── research.md             # Research context + handbook rules
│   └── plan.md                 # Training & implementation plan
│
├── models/                     # Exported model copies
├── yolo11s.pt                  # Base pretrained weights
└── PROGRESS.md                 # Development progress tracker
```

---

## 12. How to Use

### 12.1 Environment Setup

```bash
# Create Python 3.13 virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 12.2 Run Inference (Single Image)

```bash
# Auto-detect mode (uses current day + year level)
python inference/detect.py --image path/to/photo.jpg --year 2

# Force UNIFORM_MODE
python inference/detect.py --image path/to/photo.jpg --mode UNIFORM_MODE --year 2

# Force CIVILIAN_MODE
python inference/detect.py --image path/to/photo.jpg --mode CIVILIAN_MODE --year 4
```

### 12.3 Run Inference (Batch — Folder of Images)

```bash
python inference/detect.py --folder path/to/photos/ --mode CIVILIAN_MODE --year 1
```

### 12.4 Evaluate Model

```bash
python training/evaluate.py --model results/enhanced-uwear-dev/weights/best.pt --split test
```

### 12.5 Resume Training

```bash
.venv\Scripts\activate
python -c "from ultralytics import YOLO; model = YOLO('results/enhanced-uwear-dev/weights/last.pt'); model.train(resume=True)"
```

### 12.6 Live Training Dashboard

```bash
python training/dashboard.py
# Opens at http://localhost:8080
```

---

## 13. Output Format

### 13.1 JSON Compliance Report

```json
{
  "compliance_status": "NON_COMPLIANT",
  "detection_mode": "UNIFORM_MODE",
  "year_level": 2,
  "violations": [
    {
      "class": "missing_uniform_top",
      "handbook_ref": "27.1.2",
      "description": "Not wearing prescribed school uniform top on uniform day",
      "severity": "MINOR_OFFENSE"
    },
    {
      "class": "footwear_slippers",
      "handbook_ref": "27.1.2.11",
      "description": "Slippers are not allowed within the campus premises",
      "confidence": 0.92,
      "severity": "MINOR_OFFENSE"
    }
  ],
  "violation_count": 2,
  "detected_garments": {
    "tops": [{"class": "civilian_top_short_sleeve", "confidence": 0.87}],
    "bottoms": [{"class": "civilian_bottom_trousers", "confidence": 0.93}],
    "footwear": [{"class": "footwear_slippers", "confidence": 0.92}],
    "prohibited": []
  },
  "detected_classes": ["civilian_top_short_sleeve", "civilian_bottom_trousers", "footwear_slippers"],
  "timestamp": "2026-05-05T12:15:03"
}
```

### 13.2 Annotated Image

The system outputs annotated images with:
- **Green bounding boxes** for uniform items
- **Teal bounding boxes** for civilian items
- **Red bounding boxes** for prohibited items and slippers
- **Top banner**: Green "COMPLIANT" or Red "NON-COMPLIANT" with mode and violation count
- **Bottom overlay**: List of violations with handbook section references

---

## 14. Model Export Options

The trained `best.pt` can be exported for deployment on different platforms:

```python
from ultralytics import YOLO
model = YOLO("best.pt")

model.export(format="onnx")       # Web / general deployment
model.export(format="torchscript") # PyTorch mobile
model.export(format="tflite")     # Android / Edge devices
model.export(format="coreml")     # iOS devices
model.export(format="ncnn")       # Raspberry Pi / embedded
```

---

## 15. Explicitly Out of Scope

| Item | Reason |
|---|---|
| Offensive prints on clothing (27.1.2.10) | Requires OCR/text detection — not reliable with YOLO |
| Slipper type/brand distinction | All slippers prohibited regardless of type |
| Face/identity recognition | SWAFO uses student ID barcode; facial recognition raises ethics concerns |
| Real-time video stream | Image-based for thesis; video stream is a planned future enhancement |
| Shorts/skirt length measurement | Requires pose estimation (YOLO11-Pose) — documented as future work |

---

## 16. UN SDG Alignment

| SDG | Relevance |
|---|---|
| SDG 4 — Quality Education | Promotes discipline and professionalism through fair enforcement |
| SDG 9 — Industry, Innovation, Infrastructure | AI-based detection in campus management |
| SDG 10 — Reduced Inequality | Consistent, unbiased enforcement across all students |
| SDG 16 — Peace, Justice, Strong Institutions | Transparent, policy-aligned enforcement |
| SDG 17 — Partnerships for the Goals | Collaboration with SWAFO, OSS, SDAO |

---

## 17. Evaluation Targets — All Met ✅

| Metric | Target | Achieved | Status |
|---|---|---|---|
| mAP@0.5 | > 0.85 | **0.9031** | ✅ Exceeded |
| Precision | > 0.80 | **0.8560** | ✅ Exceeded |
| Recall | > 0.85 | **0.8573** | ✅ Exceeded |
| uniform_top AP | > 0.75 | **0.9919** | ✅ Exceeded |
| uniform_bottom AP | > 0.75 | **0.9950** | ✅ Exceeded |
| Skirt AP | > 0.70 | **0.8656** | ✅ Exceeded |
| Footwear AP | > 0.80 | **0.8631–0.9597** | ✅ Exceeded |

---

*End of documentation.*
