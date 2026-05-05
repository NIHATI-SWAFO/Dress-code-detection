"""
Visualize Dataset Samples
=========================
Generates 3 sample images per class with bounding boxes drawn
to verify that the labels are correct.

Usage:
    .venv\\Scripts\\python.exe scripts/visualize_samples.py
"""

import os
import cv2
import yaml
import numpy as np
import random
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_YAML = BASE_DIR / "dataset" / "balanced" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "verification_samples"
SAMPLES_PER_CLASS = 3

def main():
    print("=" * 60)
    print("DATASET VISUAL VERIFICATION")
    print("=" * 60)

    # Load names from data.yaml
    with open(DATA_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    
    class_names = cfg['names']
    num_classes = cfg['nc']
    base_path = Path(cfg['path'])
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Search for samples across all splits
    splits = ['train', 'val', 'test']
    
    # Store found samples: {class_id: [(img_path, label_path), ...]}
    samples_found = {i: [] for i in range(num_classes)}
    
    print("Searching for samples...")
    for split in splits:
        img_dir = base_path / "images" / split
        lbl_dir = base_path / "labels" / split
        
        if not img_dir.exists(): continue
        
        # Get list of label files and shuffle to get random samples
        lbl_files = list(lbl_dir.glob("*.txt"))
        random.shuffle(lbl_files)
        
        for lbl_file in lbl_files:
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            classes_in_file = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    classes_in_file.add(int(parts[0]))
            
            img_file = img_dir / (lbl_file.stem + ".jpg")
            if not img_file.exists():
                img_file = img_dir / (lbl_file.stem + ".png")
            
            if img_file.exists():
                for cls_id in classes_in_file:
                    if len(samples_found[cls_id]) < SAMPLES_PER_CLASS:
                        samples_found[cls_id].append((img_file, lbl_file))
            
            # Check if we have enough samples for all classes
            if all(len(v) >= SAMPLES_PER_CLASS for v in samples_found.values()):
                break
        
        if all(len(v) >= SAMPLES_PER_CLASS for v in samples_found.values()):
            break

    print("Generating visualizations...")
    # Colors for bounding boxes (Bayer pattern style)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128),
        (0, 128, 128), (0, 0, 128)
    ]

    for cls_id, samples in samples_found.items():
        cls_name = class_names[cls_id]
        print(f"  Class {cls_id} ({cls_name}): {len(samples)} samples")
        
        for i, (img_path, lbl_path) in enumerate(samples):
            img = cv2.imread(str(img_path))
            h, w, _ = img.shape
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                target_cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Convert normalized to pixel coordinates
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Highlight the target class with a thicker box
                is_target = (target_cls == cls_id)
                color = colors[target_cls % len(colors)]
                thickness = 4 if is_target else 1
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Add label text
                label = f"{class_names[target_cls]}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save the result
            out_name = f"class_{cls_id}_{cls_name}_{i+1}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / out_name), img)

    print(f"\nDone! Samples saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
