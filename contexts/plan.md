# plan.md — Module 1: Enhanced UWear Training & Implementation Plan
> Project: SWAFO Violation Management Web App + Enhanced UWear
> Institution: De La Salle University-Dasmariñas (DLSU-D)
> Last Updated: May 2026
> Status: Dataset assembled — entering training phase

---

## 1. ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────────┐
│                    MODULE 1 PIPELINE                      │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │  Raw Image  │───►│  YOLO11      │───►│  Rule       │ │
│  │  Input      │    │  Detection   │    │  Engine     │ │
│  │  (camera /  │    │  Model       │    │  (Policy    │ │
│  │   upload)   │    │  14 classes  │    │   Logic)    │ │
│  └─────────────┘    └──────────────┘    └──────┬──────┘ │
│                                                │        │
│                                         ┌──────▼──────┐ │
│                                         │  Compliance │ │
│                                         │  Output     │ │
│                                         │  (JSON)     │ │
│                                         └──────┬──────┘ │
└────────────────────────────────────────────────┼────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │   MODULE 2 — SWAFO      │
                                    │   Web App               │
                                    │   Violation Recording   │
                                    └─────────────────────────┘
```

---

## 2. CORE TECHNOLOGY CHOICES

| Component | Choice | Justification |
|---|---|---|
| Detection Model | **YOLO11** (Ultralytics) | Latest Ultralytics model; same family as UWear's YOLOv5 enabling direct benchmark comparison; better accuracy with fewer parameters than YOLOv8 |
| Dataset Format | **YOLOv8/YOLO11 format** | Native format; Roboflow exports in this format; identical between v8 and v11 |
| Dataset Platform | **Roboflow** | Same platform as UWear; class management, annotation, augmentation, version control |
| Deep Learning Framework | **PyTorch** | Same as UWear; YOLO11 runs on PyTorch natively |
| CV Library | **OpenCV** | Same as UWear; image preprocessing and inference overlay |
| Programming Language | **Python** | Same as UWear; all ML tooling is Python-native |
| Training Environment | **Google Colab (Pro recommended)** | Free GPU access; sufficient for YOLO11 training |
| Merge & Balancing | **Custom Python script (local)** | Avoids Roboflow free-tier upload limits; full control over class balancing |

---

## 3. YOLO11 — MODEL SIZE SELECTION

| Model | Parameters | mAP (COCO) | Speed | Use For |
|---|---|---|---|---|
| yolo11n | 2.6M | 39.5 | Fastest | Quick testing only |
| yolo11s | 9.4M | 47.0 | Fast | Development & iteration |
| **yolo11m** | **20.1M** | **51.5** | **Medium** | **Recommended for thesis** |
| yolo11l | 25.3M | 53.4 | Slow | Final results if hardware allows |
| yolo11x | 56.9M | 54.7 | Slowest | Only if Colab Pro+ available |

> **Use yolo11s during development. Use yolo11m for final thesis results.**
> yolo11m gives the best balance of accuracy and training time on free Colab GPU.

---

## 4. DATASET PIPELINE

### 4.1 Current Dataset State
- **Total objects:** 315,597 across 14 classes
- **Preprocessing:** Not yet applied
- **Augmentation:** Not yet applied
- **Format:** YOLOv8 compatible (from Roboflow)
- **Primary data:** Pending ethics approval — secondary only at this stage

### 4.2 Class Imbalance — The Critical Problem

```
uniform_top       :    1,018  ← IMPROVED
uniform_bottom    :      811  ← IMPROVED
civilian classes  :  230,546  ← DOMINANT
```

If trained as-is, the model will be heavily biased toward civilian classes
and nearly blind to uniform detection. This MUST be fixed before training.

### 4.3 Balancing Strategy

**Step 1 — Cap civilian classes during merge**
```python
MAX_SAMPLES_PER_CLASS = {
    0: 2000,   # uniform_top — keep ALL + augment
    1: 2000,   # uniform_bottom — keep ALL + augment
    2: 5000,   # civilian_top_short_sleeve — CAP
    3: 5000,   # civilian_top_long_sleeve — CAP
    4: 5000,   # civilian_bottom_trousers — CAP
    5: 5000,   # civilian_bottom_shorts — CAP
    6: 5000,   # civilian_bottom_skirt — CAP
    7: 5000,   # footwear_shoes — CAP
    8: 4054,   # footwear_slippers — keep ALL
    9: 2071,   # prohibited_ripped_pants — keep ALL
    10: 5000,  # prohibited_leggings — CAP
    11: 5000,  # prohibited_sleeveless — CAP
    12: 5000,  # prohibited_crop_halter — CAP
    13: 5000,  # prohibited_midriff_off-shoulder — CAP
}
```

**Step 2 — Augment uniform classes aggressively**

Apply these augmentations specifically to uniform_top and uniform_bottom:

| Augmentation | Setting | Reason |
|---|---|---|
| Horizontal flip | On | Students seen from both sides |
| Rotation | ±15° | Camera angle variation |
| Brightness | ±30% | Indoor vs outdoor lighting |
| Contrast | ±20% | Camera quality variation |
| Saturation | ±20% | Color variation across cameras |
| Blur | Up to 2px | Motion blur, distance |
| Mosaic | On | YOLO11 native, improves small object detection |
| MixUp | On | Helps with overlapping detections |
| Scale | ±50% | Students at different distances from camera |

> Target: minimum 2,000 uniform images per class after augmentation.
> Set augmentation multiplier to **5x** for uniform classes in Roboflow
> before downloading, OR use YOLO11's built-in online augmentation during training.

### 4.4 Dataset Split
```
Train : 70%
Valid : 20%
Test  : 10%
```
Apply split AFTER balancing and BEFORE augmentation to prevent data leakage.

### 4.5 Merge Execution (Local Python Script)

```bash
# Step 1: Download all Roboflow projects separately
python download_datasets.py

# Step 2: Run merge and balancing script
python dataset_merge.py

# Output: /merged_dataset/
#   ├── train/images/ + train/labels/
#   ├── valid/images/ + valid/labels/
#   ├── test/images/ + test/labels/
#   └── data.yaml
```

---

## 5. TRAINING PIPELINE

### 5.1 Google Colab Setup

```python
# Install dependencies
!pip install ultralytics roboflow

# Mount Google Drive (to save checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU
import torch
print(torch.cuda.get_device_name(0))
```

### 5.2 Dataset Download

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

# Download each project separately
rf.workspace("your-workspace").project("uniform-project").version(1).download(
    "yolov8", location="./datasets/uniform"
)
rf.workspace("your-workspace").project("civilian-project").version(1).download(
    "yolov8", location="./datasets/civilian"
)
rf.workspace("your-workspace").project("footwear-project").version(1).download(
    "yolov8", location="./datasets/footwear"
)
rf.workspace("your-workspace").project("prohibited-project").version(1).download(
    "yolov8", location="./datasets/prohibited"
)
```

### 5.3 Run Merge Script

```python
# Upload dataset_merge.py to Colab then run
!python dataset_merge.py
# Output: ./merged_dataset/
```

### 5.4 Training — Development Run (yolo11s)

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Development model

results = model.train(
    data="./merged_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="enhanced-uwear-dev",
    patience=20,            # Early stopping — stops if no improvement for 20 epochs
    save=True,
    save_period=10,         # Save checkpoint every 10 epochs
    val=True,
    plots=True,
    device=0,               # GPU
    workers=4,
    # Class weight — give MORE importance to uniform classes
    # because they are underrepresented even after balancing
    cls=2.0,                # Increase classification loss weight
)
```

### 5.5 Training — Final Thesis Run (yolo11m)

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # Final thesis model

results = model.train(
    data="./merged_dataset/data.yaml",
    epochs=150,             # More epochs than UWear's 45
    imgsz=640,
    batch=8,                # Smaller batch for larger model on Colab
    name="enhanced-uwear-final",
    patience=30,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    device=0,
    workers=4,
    # Augmentation (YOLO11 built-in online augmentation)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)
```

### 5.6 Save Best Model to Google Drive

```python
import shutil

# Save best weights to Drive so they survive Colab session
shutil.copy(
    "./runs/detect/enhanced-uwear-final/weights/best.pt",
    "/content/drive/MyDrive/DLSUD_UWear/best.pt"
)
print("Model saved to Google Drive.")
```

---

## 6. EVALUATION

### 6.1 Metrics to Report

```python
from ultralytics import YOLO

model = YOLO("./runs/detect/enhanced-uwear-final/weights/best.pt")

# Validate on test set
metrics = model.val(data="./merged_dataset/data.yaml", split="test")

print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
```

### 6.2 Per-Class Results (Required for Thesis)

```python
# Per-class AP — required to show improvement on UWear's weak classes
for i, cls_name in enumerate(metrics.names.values()):
    print(f"{cls_name}: AP50 = {metrics.box.ap50[i]:.4f}")
```

### 6.3 Targets vs UWear Baseline

| Metric | UWear (Baseline) | Your Target | Your Result |
|---|---|---|---|
| mAP@0.5 | 0.7883 | > 0.85 | [fill after training] |
| Precision | 0.7336 | > 0.80 | [fill after training] |
| Recall | 0.7881 | > 0.85 | [fill after training] |
| mAP@0.5:0.95 | Not reported | > 0.65 | [fill after training] |
| uniform_top AP | N/A | > 0.75 | [fill after training] |
| uniform_bottom AP | N/A | > 0.75 | [fill after training] |
| skirt AP | ~0.0 (background) | > 0.70 | [fill after training] |
| footwear AP | N/A | > 0.80 | [fill after training] |

---

## 7. INFERENCE & RULE ENGINE

### 7.1 Inference on a Single Image

```python
from ultralytics import YOLO
import json
from datetime import datetime

model = YOLO("best.pt")

def run_compliance_check(image_path, detection_mode, year_level):
    """
    detection_mode: "UNIFORM_MODE" or "CIVILIAN_MODE"
    year_level: 1, 2, 3, or 4
    """
    results = model(image_path)[0]
    detected_classes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf >= 0.5:  # Confidence threshold
            detected_classes.append(cls_id)

    return apply_rule_engine(detected_classes, detection_mode, year_level)


def apply_rule_engine(detected_classes, detection_mode, year_level):
    CLASS_NAMES = [
        "uniform_top", "uniform_bottom",
        "civilian_top_short_sleeve", "civilian_top_long_sleeve",
        "civilian_bottom_trousers", "civilian_bottom_shorts", "civilian_bottom_skirt",
        "footwear_shoes", "footwear_slippers",
        "prohibited_ripped_pants", "prohibited_leggings", "prohibited_sleeveless",
        "prohibited_crop_halter", "prohibited_midriff_off-shoulder"
    ]

    PROHIBITED_IDS = {9, 10, 11, 12, 13}
    UNIFORM_TOP_ID = 0
    UNIFORM_BOTTOM_ID = 1
    FOOTWEAR_SHOES_ID = 7
    FOOTWEAR_SLIPPERS_ID = 8

    violations = []
    detected_set = set(detected_classes)

    # Always check for prohibited items — any mode, any day
    for pid in PROHIBITED_IDS:
        if pid in detected_set:
            violations.append({
                "class": CLASS_NAMES[pid],
                "handbook_ref": "27.1.2",
                "description": f"Prohibited clothing detected: {CLASS_NAMES[pid]}"
            })

    # Always check slippers
    if FOOTWEAR_SLIPPERS_ID in detected_set:
        violations.append({
            "class": "footwear_slippers",
            "handbook_ref": "27.1.2.11",
            "description": "Slippers are prohibited. Officer discretion for rain/injury."
        })

    if detection_mode == "UNIFORM_MODE" and year_level in [1, 2, 3]:
        if UNIFORM_TOP_ID not in detected_set:
            violations.append({
                "class": "missing_uniform_top",
                "handbook_ref": "27.1.2",
                "description": "Not wearing prescribed school uniform top."
            })
        if UNIFORM_BOTTOM_ID not in detected_set:
            violations.append({
                "class": "missing_uniform_bottom",
                "handbook_ref": "27.1.2",
                "description": "Not wearing prescribed school uniform bottom."
            })
        if FOOTWEAR_SHOES_ID not in detected_set and FOOTWEAR_SLIPPERS_ID not in detected_set:
            violations.append({
                "class": "missing_footwear",
                "handbook_ref": "27.1.2",
                "description": "Improper or missing footwear."
            })

    compliance_status = "COMPLIANT" if not violations else "NON_COMPLIANT"

    return {
        "detection_mode": detection_mode,
        "year_level": year_level,
        "detected_classes": [CLASS_NAMES[c] for c in detected_classes],
        "compliance_status": compliance_status,
        "violations": violations,
        "timestamp": datetime.now().isoformat()
    }
```

### 7.2 Module 2 Integration Endpoint

```python
# Django REST Framework view (Module 2 backend)
# POST /api/detection/check/
# Body: { image, student_id, detection_mode, year_level }
# Returns: compliance JSON → pre-fills violation recording form

@api_view(['POST'])
def run_detection(request):
    image = request.FILES['image']
    detection_mode = request.data['detection_mode']
    year_level = int(request.data['year_level'])
    
    result = run_compliance_check(image, detection_mode, year_level)
    return Response(result)
```

---

## 8. DEVELOPMENT PHASES

| Phase | Task | Status |
|---|---|---|
| 1 | Define 14-class schema | ✅ Done |
| 2 | Gather secondary datasets | ✅ Done |
| 3 | Remap and merge datasets | ✅ Done |
| 4 | Class balancing setup | 🔄 In Progress |
| 5 | Preprocessing and augmentation | ⏳ Next |
| 6 | Development training run (yolo11s) | ⏳ Pending |
| 7 | Evaluate and fix weak classes | ⏳ Pending |
| 8 | Final training run (yolo11m) | ⏳ Pending |
| 9 | Report metrics vs UWear baseline | ⏳ Pending |
| 10 | Build rule engine | ⏳ Pending |
| 11 | Integrate with Module 2 API | ⏳ Pending |
| 12 | Primary data collection (post-ethics) | ⏳ After approval |
| 13 | Retrain with primary + secondary data | ⏳ After approval |

---

## 9. PENDING DECISIONS

| Decision | Status | Notes |
|---|---|---|
| Detection algorithm | ✅ YOLO11 | Confirmed |
| Dataset platform | ✅ Roboflow | Confirmed |
| Model size for final run | ✅ yolo11m | Confirmed |
| Hosting for inference API | ⚠️ TBD | Options: Railway, Render, school server |
| Primary data ethics approval | ⚠️ Pending | Required before primary collection |
| Length-based violation (shorts/skirt) | ⚠️ TBD | Requires pose estimation — possibly future work |
| Real-time video stream support | ⚠️ TBD | Image-based for now; video is v2 feature |
| Module 1 + Module 2 integration method | ⚠️ TBD | REST API endpoint confirmed; deployment TBD |

---

## 10. IMMEDIATE NEXT STEPS (IN ORDER)

```
1. Run dataset_merge.py → produce merged_dataset/
2. Check class distribution → verify caps applied correctly
3. Upload merged_dataset/ to Google Colab or Google Drive
4. Run development training with yolo11s (epochs=100)
5. Check training curves → look for overfitting or underfitting
6. Run validation → get per-class AP
7. Identify weak classes → fix with more data or augmentation
8. Run final training with yolo11m (epochs=150)
9. Report all metrics vs UWear baseline
10. Build rule engine on top of best.pt
11. Connect rule engine output to Module 2 Django endpoint
```
