"""
Fix Segmentation Labels → YOLO Bounding Box Format
=====================================================
Some uniform_* labels are in YOLO segmentation format (polygon coordinates)
instead of standard detection format (class cx cy w h).

This script converts them in-place:
  - Reads polygon coordinates (class x1 y1 x2 y2 ... xN yN)
  - Computes bounding box: min/max of all x,y polygon points
  - Converts to YOLO bbox: class cx cy w h
  - Overwrites the label file with the corrected format

Runs on BOTH full/ and balanced/ datasets.

Usage:
    .venv\Scripts\python.exe scripts/fix_segmentation_labels.py
"""

import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def polygon_to_bbox(parts):
    """Convert 'class x1 y1 x2 y2 ... xN yN' to 'class cx cy w h'."""
    cls_id = parts[0]
    coords = [float(v) for v in parts[1:]]

    # Polygon coords are pairs: x1, y1, x2, y2, ...
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None  # Can't parse

    xs = coords[0::2]  # Every even index = x
    ys = coords[1::2]  # Every odd index = y

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # YOLO bbox format: center_x, center_y, width, height (all normalized 0-1)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Clamp to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.001, min(1.0, w))
    h = max(0.001, min(1.0, h))

    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def fix_label_file(filepath):
    """Fix a single label file. Returns True if it was modified."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if len(parts) > 5:
            # This is a segmentation line — convert it
            bbox_line = polygon_to_bbox(parts)
            if bbox_line:
                new_lines.append(bbox_line + "\n")
                modified = True
            else:
                new_lines.append(line)  # Keep as-is if can't parse
        else:
            new_lines.append(line)  # Already bbox format

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    return modified


def process_directory(label_dir, label=""):
    """Process all label files in a directory."""
    fixed = 0
    total = 0

    with os.scandir(label_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith('.txt'):
                continue
            if not entry.name.startswith('uniform_'):
                continue  # Only uniform_* labels have this issue

            total += 1
            if fix_label_file(entry.path):
                fixed += 1

            if total % 1000 == 0:
                print(f"  [{label}] Processed {total} uniform labels, fixed {fixed}...", flush=True)

    print(f"  [{label}] Done: {fixed} fixed out of {total} uniform labels")
    return fixed, total


if __name__ == "__main__":
    start = time.time()

    print("=" * 60)
    print("FIX SEGMENTATION LABELS → YOLO BBOX FORMAT")
    print("=" * 60)

    total_fixed = 0

    # Fix full dataset (master copy)
    print("\nProcessing full dataset...")
    full_lbl = os.path.join(BASE_DIR, "dataset", "full", "labels", "train")
    f, t = process_directory(full_lbl, "full")
    total_fixed += f

    # Fix balanced dataset (all three splits)
    for split in ["train", "val", "test"]:
        print(f"\nProcessing balanced/{split}...")
        balanced_lbl = os.path.join(BASE_DIR, "dataset", "balanced", "labels", split)
        f, t = process_directory(balanced_lbl, f"balanced/{split}")
        total_fixed += f

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE — Fixed {total_fixed} label files in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print("\nNext: Re-run validate_dataset.py to confirm all issues resolved.")
