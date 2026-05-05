"""
Offline Augmentation for Minority Classes
==========================================
Generates augmented copies of under-represented class images
directly into the training set BEFORE training starts.

Augmentations applied (all label-safe):
  1. Horizontal flip         (cx → 1-cx)
  2. Brightness adjustment   (no label change)
  3. Contrast adjustment     (no label change)
  4. HSV color jitter         (no label change)
  5. Slight blur              (no label change)

Only augments images that contain at least one target minority class.
New files are named with _aug{N}_ prefix to avoid collisions.

Usage:
    .venv/Scripts/python.exe scripts/augment_minority_classes.py
"""

import os
import sys
import cv2
import numpy as np
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent.parent
TRAIN_IMG_DIR = BASE_DIR / "dataset" / "balanced" / "images" / "train"
TRAIN_LBL_DIR = BASE_DIR / "dataset" / "balanced" / "labels" / "train"

# Classes to boost and their target image counts
BOOST_TARGETS = {
    0: 5000,   # uniform_top (684 → 5000)
    1: 5000,   # uniform_bottom (565 → 5000)
    8: 5000,   # footwear_slippers (1269 → 5000)
    9: 5000,   # prohibited_ripped_pants (1449 → 5000)
}

CLASS_NAMES = [
    "uniform_top", "uniform_bottom",
    "civilian_top_short_sleeve", "civilian_top_long_sleeve",
    "civilian_bottom_trousers", "civilian_bottom_shorts", "civilian_bottom_skirt",
    "footwear_shoes", "footwear_slippers",
    "prohibited_ripped_pants", "prohibited_leggings", "prohibited_sleeveless",
    "prohibited_crop_halter", "prohibited_midriff_offshoulder"
]

random.seed(42)
np.random.seed(42)


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================
def augment_hflip(img, labels):
    """Horizontal flip — must adjust bbox cx."""
    flipped = cv2.flip(img, 1)
    new_labels = []
    for line in labels:
        parts = line.strip().split()
        cls_id = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx = 1.0 - cx  # Mirror x-center
        new_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return flipped, new_labels


def augment_brightness(img, labels, factor=None):
    """Random brightness adjustment — labels unchanged."""
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, labels


def augment_contrast(img, labels, factor=None):
    """Random contrast adjustment — labels unchanged."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    result = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    return result, labels


def augment_hsv_jitter(img, labels):
    """Random HSV color jitter — labels unchanged."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, labels


def augment_blur(img, labels):
    """Slight Gaussian blur — labels unchanged."""
    ksize = random.choice([3, 5])
    result = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return result, labels


# List of augmentation functions (flip is always applied first when chosen)
AUG_FUNCS = [
    ("hflip", augment_hflip),
    ("bright", augment_brightness),
    ("contrast", augment_contrast),
    ("hsv", augment_hsv_jitter),
    ("blur", augment_blur),
]


def apply_random_augmentation(img, labels, aug_idx):
    """Apply a combination of augmentations based on index for variety."""
    # Always start with original labels as list of strings
    current_img = img.copy()
    current_labels = list(labels)

    if aug_idx % 5 == 0:
        # Horizontal flip only
        current_img, current_labels = augment_hflip(current_img, current_labels)
    elif aug_idx % 5 == 1:
        # Brightness + HSV jitter
        current_img, current_labels = augment_brightness(current_img, current_labels)
        current_img, current_labels = augment_hsv_jitter(current_img, current_labels)
    elif aug_idx % 5 == 2:
        # Horizontal flip + contrast
        current_img, current_labels = augment_hflip(current_img, current_labels)
        current_img, current_labels = augment_contrast(current_img, current_labels)
    elif aug_idx % 5 == 3:
        # Brightness + blur
        current_img, current_labels = augment_brightness(current_img, current_labels)
        current_img, current_labels = augment_blur(current_img, current_labels)
    elif aug_idx % 5 == 4:
        # Flip + HSV + contrast
        current_img, current_labels = augment_hflip(current_img, current_labels)
        current_img, current_labels = augment_hsv_jitter(current_img, current_labels)
        current_img, current_labels = augment_contrast(current_img, current_labels)

    return current_img, current_labels


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("OFFLINE AUGMENTATION — MINORITY CLASS BOOSTING")
    print("=" * 60)

    # Step 1: Find all images containing each minority class
    print("\nScanning training labels...")
    class_to_files = defaultdict(list)  # cls_id → [base_name, ...]

    for lbl_file in TRAIN_LBL_DIR.iterdir():
        if not lbl_file.name.endswith('.txt'):
            continue
        # Skip already-augmented files (in case of re-run)
        if '_aug' in lbl_file.name:
            continue

        base = lbl_file.stem
        with open(lbl_file) as f:
            lines = f.readlines()

        classes_in_file = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes_in_file.add(int(parts[0]))

        for cls_id in classes_in_file:
            if cls_id in BOOST_TARGETS:
                class_to_files[cls_id].append(base)

    # Step 2: Calculate how many augmented copies each class needs
    print("\nAugmentation plan:")
    total_new = 0
    aug_plan = {}  # cls_id → number of new images needed

    for cls_id, target in BOOST_TARGETS.items():
        current = len(class_to_files[cls_id])
        needed = max(0, target - current)
        aug_plan[cls_id] = needed
        total_new += needed
        print(f"  {CLASS_NAMES[cls_id]:<35} {current:>5} → {target:>5} (need {needed} augmented)")

    print(f"\n  Total new images to generate: {total_new}")

    # Step 3: Generate augmented images
    print("\nGenerating augmented images...")
    generated = defaultdict(int)

    for cls_id, needed in aug_plan.items():
        if needed == 0:
            continue

        source_files = class_to_files[cls_id]
        random.shuffle(source_files)

        aug_count = 0
        cycle = 0

        while aug_count < needed:
            cycle += 1
            for base in source_files:
                if aug_count >= needed:
                    break

                # Find the original image
                img_path = TRAIN_IMG_DIR / (base + ".jpg")
                if not img_path.exists():
                    img_path = TRAIN_IMG_DIR / (base + ".png")
                if not img_path.exists():
                    continue

                lbl_path = TRAIN_LBL_DIR / (base + ".txt")
                if not lbl_path.exists():
                    continue

                # Read image and labels
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                with open(lbl_path) as f:
                    labels = f.readlines()

                # Apply augmentation
                aug_idx = aug_count  # Varies the augmentation type
                aug_img, aug_labels = apply_random_augmentation(img, labels, aug_idx)

                # Save with unique name
                ext = img_path.suffix
                new_base = f"{base}_aug{cls_id}_{aug_count}"
                new_img_path = TRAIN_IMG_DIR / (new_base + ext)
                new_lbl_path = TRAIN_LBL_DIR / (new_base + ".txt")

                # Don't overwrite if exists
                if new_img_path.exists():
                    aug_count += 1
                    continue

                cv2.imwrite(str(new_img_path), aug_img)
                with open(new_lbl_path, 'w') as f:
                    f.writelines(aug_labels)

                aug_count += 1
                generated[cls_id] += 1

                if aug_count % 500 == 0:
                    print(f"    [{CLASS_NAMES[cls_id]}] Generated {aug_count}/{needed}...", flush=True)

        print(f"  {CLASS_NAMES[cls_id]}: +{generated[cls_id]} augmented images")

    # Step 4: Verify final counts
    print("\n" + "=" * 60)
    print("VERIFICATION — Post-Augmentation Counts")
    print("=" * 60)

    final_counts = defaultdict(int)
    for lbl_file in TRAIN_LBL_DIR.iterdir():
        if not lbl_file.name.endswith('.txt'):
            continue
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    final_counts[int(parts[0])] += 1

    print(f"\n{'Class':<35} {'Annotations':>12}")
    print("-" * 50)
    for i in range(14):
        flag = " ✅ BOOSTED" if i in BOOST_TARGETS else ""
        print(f"  {CLASS_NAMES[i]:<33} {final_counts[i]:>10}{flag}")

    total_imgs = len(list(TRAIN_IMG_DIR.glob("*.*")))
    total_lbls = len(list(TRAIN_LBL_DIR.glob("*.txt")))
    print(f"\n  Total training images: {total_imgs}")
    print(f"  Total training labels: {total_lbls}")
    print(f"  New images added: {sum(generated.values())}")

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
