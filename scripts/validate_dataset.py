"""
Pre-Training Validation Script
================================
Checks the balanced training split for:
1. Corrupted / unreadable image files
2. Label files with invalid YOLO format (wrong columns, out-of-range coords)
3. Images with no annotations (empty label files)
4. Labels with class IDs outside 0-13

Run this BEFORE training. Fix any issues reported before running train_dev.py.

Usage:
    .venv\Scripts\python.exe scripts/validate_dataset.py
"""

import os
import sys
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install pillow")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALANCED_DIR = os.path.join(BASE_DIR, "dataset", "balanced")
SPLITS = ["train", "val", "test"]
NUM_CLASSES = 14

# ============================================================
# VALIDATION
# ============================================================
def validate_split(split):
    img_dir = os.path.join(BALANCED_DIR, "images", split)
    lbl_dir = os.path.join(BALANCED_DIR, "labels", split)

    bad_images = []       # Unreadable image files
    bad_labels = []       # Malformed label lines
    empty_labels = []     # Images with zero annotations
    bad_class_ids = []    # Class ID outside 0-13
    checked = 0

    with os.scandir(img_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue

            checked += 1
            if checked % 10000 == 0:
                print(f"  [{split}] Checked {checked}...", flush=True)

            base = os.path.splitext(entry.name)[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")

            # --- Check 1: Image readable ---
            try:
                with Image.open(entry.path) as img:
                    img.verify()
            except Exception as e:
                bad_images.append((entry.path, str(e)))
                continue

            # --- Check 2: Label file exists ---
            if not os.path.exists(lbl_path):
                bad_labels.append((lbl_path, "Label file missing"))
                continue

            # --- Check 3: Parse label lines ---
            with open(lbl_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]

            if len(lines) == 0:
                empty_labels.append(lbl_path)
                continue

            for i, line in enumerate(lines):
                parts = line.split()

                # Must have exactly 5 values: class cx cy w h
                if len(parts) != 5:
                    bad_labels.append((lbl_path, f"Line {i+1}: expected 5 values, got {len(parts)}"))
                    break

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    bad_labels.append((lbl_path, f"Line {i+1}: non-numeric values"))
                    break

                # Check 4: Class ID in range
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    bad_class_ids.append((lbl_path, f"Line {i+1}: class {cls_id} out of range 0-{NUM_CLASSES-1}"))
                    break

                # Check 5: Coordinates in 0-1 range
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bad_labels.append((
                        lbl_path,
                        f"Line {i+1}: coords out of range — cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
                    ))
                    break

    return {
        "split": split,
        "checked": checked,
        "bad_images": bad_images,
        "bad_labels": bad_labels,
        "empty_labels": empty_labels,
        "bad_class_ids": bad_class_ids,
    }


# ============================================================
# REPORT
# ============================================================
def print_report(results):
    print("\n" + "=" * 65)
    print("VALIDATION REPORT")
    print("=" * 65)

    total_issues = 0

    for r in results:
        split = r["split"]
        n_bad_img = len(r["bad_images"])
        n_bad_lbl = len(r["bad_labels"])
        n_empty = len(r["empty_labels"])
        n_bad_cls = len(r["bad_class_ids"])
        issues = n_bad_img + n_bad_lbl + n_bad_cls  # empty labels are a warning, not error

        status = "✅ CLEAN" if issues == 0 else f"❌ {issues} ISSUES"

        print(f"\n--- {split.upper()} ({r['checked']} files) — {status} ---")

        if n_bad_img > 0:
            print(f"  🔴 Corrupted images: {n_bad_img}")
            for path, err in r["bad_images"][:5]:
                print(f"     {os.path.basename(path)}: {err}")
            if n_bad_img > 5:
                print(f"     ... and {n_bad_img - 5} more")

        if n_bad_lbl > 0:
            print(f"  🔴 Invalid label files: {n_bad_lbl}")
            for path, err in r["bad_labels"][:5]:
                print(f"     {os.path.basename(path)}: {err}")
            if n_bad_lbl > 5:
                print(f"     ... and {n_bad_lbl - 5} more")

        if n_bad_cls > 0:
            print(f"  🔴 Bad class IDs: {n_bad_cls}")
            for path, err in r["bad_class_ids"][:5]:
                print(f"     {os.path.basename(path)}: {err}")

        if n_empty > 0:
            print(f"  🟡 Empty label files (no annotations): {n_empty}")
            print(f"     These images will be used as background negatives during training.")
            print(f"     This is NORMAL and expected — no action needed.")

        if issues == 0 and n_empty == 0:
            print(f"  ✅ No issues found.")

        total_issues += issues

    print("\n" + "=" * 65)
    if total_issues == 0:
        print("✅ ALL SPLITS CLEAN — Safe to proceed with training.")
    else:
        print(f"❌ TOTAL ISSUES: {total_issues} — Fix before training!")
        print("   Delete or repair the listed files, then re-run this script.")
    print("=" * 65)

    return total_issues


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 65)
    print("PRE-TRAINING DATASET VALIDATION")
    print(f"Balanced dataset: {BALANCED_DIR}")
    print("=" * 65)

    start = time.time()
    results = []
    for split in SPLITS:
        print(f"\nValidating {split}...", flush=True)
        r = validate_split(split)
        results.append(r)

    total_issues = print_report(results)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f} seconds")

    sys.exit(0 if total_issues == 0 else 1)
