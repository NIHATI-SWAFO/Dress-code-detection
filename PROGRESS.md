# MODULE 1 — Enhanced UWear Progress Tracker
> Project: Dress Code Detection System (DLSU-D)
> Last Updated: May 2026

---

## SUMMARY — ORDERED CHECKLIST

```
SETUP (Pre-Training)
  [x] STEP 1:  Define 14-class schema (uniform, civilian, footwear, prohibited)
  [x] STEP 2:  Gather and audit secondary datasets (204,927 images)
  [x] STEP 3:  Remap all class IDs to unified 0-13 schema
  [x] STEP 4:  Merge 3 source datasets into merged_dataset/
  [x] STEP 5:  Verify final annotation counts (315,597 annotations ✅)
  [x] STEP 6:  Restructure project directory (dataset/, training/, scripts/)
  [x] STEP 7:  Subsample dataset_output to 25% — fix 88:1 class imbalance
  [x] STEP 8:  Stratified split → 66,003 train / 18,854 val / 9,443 test
  [x] STEP 9:  Verify split — no data leakage, all counts match ✅
  [x] STEP 10: Set up Python 3.13 venv with CUDA PyTorch (GTX 1660 SUPER)
  [x] STEP 11: Install Ultralytics 8.4.45 — GPU confirmed ✅

PREPROCESSING (Before Training)
  NOTE: YOLO11 handles resize (640x640) and pixel normalization automatically.
        We only need to validate data integrity before handing it to the trainer.
  [x] STEP 12: Run validate_dataset.py — found 2,190 bad label files
                 - 2,190 uniform_* files had segmentation polygon format
                 - NOT standard YOLO bbox format (class cx cy w h)
  [x] STEP 13: Fixed via fix_segmentation_labels.py
                 - Converted polygon coords → bounding boxes (min/max x,y)
                 - Fixed in both full/ and balanced/ datasets
  [x] STEP 14: Re-validated — ALL 3 SPLITS CLEAN ✅
  [x] STEP 14b: Offline augmentation for minority classes
                 - uniform_top:          684 → 5,000 images (+4,316)
                 - uniform_bottom:       565 → 5,000 images (+4,435)
                 - footwear_slippers:  1,269 → 5,000 images (+3,731)
                 - prohibited_ripped:  1,449 → 5,000 images (+3,551)
                 - Training set grew: 66,003 → 82,036 images

TRAINING
  [ ] STEP 15: Dev run — yolo11s, 100 epochs (~10-12 hrs)
  [ ] STEP 16: Review training curves — check for overfitting
  [ ] STEP 17: Check per-class AP — flag any class below 0.50
  [ ] STEP 18: Fix weak classes if needed (adjust subsample ratio or augmentation)
  [ ] STEP 19: Final run — yolo11m, 150 epochs (~20-25 hrs)
  [ ] STEP 20: Evaluate on TEST SET — report all metrics vs UWear baseline

EVALUATION TARGETS (Must Beat UWear)
  [ ] mAP@0.5      > 0.85  (UWear baseline: 0.7883)
  [ ] Precision    > 0.80  (UWear baseline: 0.7336)
  [ ] Recall       > 0.85  (UWear baseline: 0.7881)
  [ ] uniform_top AP       > 0.75  (not in UWear)
  [ ] uniform_bottom AP    > 0.75  (not in UWear)
  [ ] skirt AP             > 0.70  (UWear: near 0.0)
  [ ] footwear AP          > 0.80  (not in UWear)

POST-TRAINING
  [ ] STEP 18: Build rule engine on top of best.pt
                - UNIFORM_MODE: check for uniform_top, uniform_bottom, footwear_shoes
                - CIVILIAN_MODE: check for prohibited items, slippers
                - Year-level awareness (1st-3rd vs 4th year rules)
  [ ] STEP 19: Add pose estimation for shorts/skirt length check (yolo11n-pose)
                - civilian_bottom_shorts → flag if hem above knee (27.1.2.3)
                - civilian_bottom_skirt  → flag if hem > 2in above knee (27.1.2.7)
  [ ] STEP 20: Connect rule engine output to Module 2 Django API endpoint
                - POST /api/detection/check/
                - Returns compliance JSON → pre-fills SWAFO violation form

FUTURE (Post-Ethics Approval)
  [ ] STEP 21: Collect primary DLSU-D student images (campus cameras)
  [ ] STEP 22: Retrain with primary + secondary combined dataset
  [ ] STEP 23: Re-evaluate and update all metrics
  [ ] STEP 24: Deploy inference API (Railway / Render / school server)
```

---

## Current Status

| Phase | Status |
| :--- | :--- |
| Dataset Assembly | ✅ Complete |
| Environment Setup | ✅ Complete — GTX 1660 SUPER + CUDA ready |
| Dev Training | ⏳ Ready to start |
| Final Training | ⏳ Pending dev results |
| Post-Training Integration | ⏳ Pending |
| Primary Data Collection | ⚠️ Blocked — Ethics approval pending |

---

## Dataset Summary

| Property | Value |
| :--- | :--- |
| Full dataset | 204,927 images / 315,597 annotations |
| Balanced training set | 94,300 images / 129,066 annotations |
| Split | 70% train / 20% val / 10% test |
| Classes | 14 (IDs 0–13) |
| GPU | NVIDIA GTX 1660 SUPER (6GB VRAM) |
| Dev model | yolo11s |
| Final model | yolo11m |

---

## Results (Fill After Training)

| Metric | UWear Baseline | Dev Run (yolo11s) | Final Run (yolo11m) |
| :--- | ---: | :--- | :--- |
| mAP@0.5 | 0.7883 | — | — |
| mAP@0.5:0.95 | N/A | — | — |
| Precision | 0.7336 | — | — |
| Recall | 0.7881 | — | — |
| uniform_top AP | N/A | — | — |
| uniform_bottom AP | N/A | — | — |
| skirt AP | ~0.0 | — | — |
| footwear_shoes AP | N/A | — | — |

---

## How to Run

```bash
# Activate environment (do this every time)
.venv\Scripts\activate

# Start dev training (ready now)
python training/train_dev.py

# Evaluate after training
python training/evaluate.py --model models/yolo11s_dev_best.pt
```
