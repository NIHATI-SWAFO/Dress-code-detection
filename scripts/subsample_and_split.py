"""
Subsample & Stratified Split Script
=====================================
1. Keeps ALL images from uniform_* and violations_* sources
2. Randomly samples 25% of dataset_output_* images
3. Splits the result into 70/20/10 (train/val/test) using stratified sampling
4. Creates hard links (no extra disk space) into dataset/balanced/
5. Generates data.yaml for the balanced split
"""

import os
import random
import shutil
from collections import Counter, defaultdict
import time

# ============================================================
# CONFIG
# ============================================================
SEED = 42
SUBSAMPLE_RATIO = 0.25  # Keep 25% of dataset_output images
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_IMAGES = os.path.join(BASE_DIR, "dataset", "full", "images", "train")
FULL_LABELS = os.path.join(BASE_DIR, "dataset", "full", "labels", "train")
BALANCED_DIR = os.path.join(BASE_DIR, "dataset", "balanced")

# ============================================================
# STEP 1: Gather all image-label pairs, grouped by source
# ============================================================
def gather_pairs():
    """Find all image files and match them with their label files."""
    print("Gathering image-label pairs...", flush=True)
    
    sources = defaultdict(list)  # source_prefix -> [(img_path, lbl_path, dominant_class)]
    missing_labels = 0
    
    with os.scandir(FULL_IMAGES) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            
            base_name = os.path.splitext(entry.name)[0]
            label_path = os.path.join(FULL_LABELS, base_name + ".txt")
            
            if not os.path.exists(label_path):
                missing_labels += 1
                continue
            
            # Determine source prefix
            if entry.name.startswith("dataset_output_"):
                source = "dataset_output"
            elif entry.name.startswith("uniform_"):
                source = "uniform"
            elif entry.name.startswith("violations_"):
                source = "violations"
            else:
                source = "other"
            
            # Find dominant class (most annotations) for stratification
            class_counts = Counter()
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_counts[int(parts[0])] += 1
                        except ValueError:
                            pass
            
            dominant_class = class_counts.most_common(1)[0][0] if class_counts else -1
            sources[source].append((entry.path, label_path, entry.name, dominant_class))
    
    if missing_labels > 0:
        print(f"  Warning: {missing_labels} images had no matching label file")
    
    for src, pairs in sources.items():
        print(f"  {src}: {len(pairs)} image-label pairs")
    
    return sources


# ============================================================
# STEP 2: Subsample dataset_output, keep all others
# ============================================================
def subsample(sources):
    """Keep all uniform/violations, subsample dataset_output."""
    random.seed(SEED)
    
    selected = []
    
    for source, pairs in sources.items():
        if source == "dataset_output":
            n_keep = int(len(pairs) * SUBSAMPLE_RATIO)
            sampled = random.sample(pairs, n_keep)
            print(f"  Subsampled {source}: {len(pairs)} -> {len(sampled)} ({SUBSAMPLE_RATIO*100:.0f}%)")
            selected.extend(sampled)
        else:
            print(f"  Keeping all {source}: {len(pairs)}")
            selected.extend(pairs)
    
    random.shuffle(selected)
    print(f"\n  Total selected: {len(selected)} images")
    return selected


# ============================================================
# STEP 3: Stratified split
# ============================================================
def stratified_split(selected):
    """Split into train/val/test while ensuring proportional class representation."""
    print("\nPerforming stratified split...", flush=True)
    
    # Group by dominant class
    by_class = defaultdict(list)
    for item in selected:
        by_class[item[3]].append(item)
    
    train, val, test = [], [], []
    
    for cls_id in sorted(by_class.keys()):
        items = by_class[cls_id]
        random.shuffle(items)
        
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # test gets the remainder
        
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    
    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    print(f"  Train: {len(train)} images")
    print(f"  Val:   {len(val)} images")
    print(f"  Test:  {len(test)} images")
    print(f"  Total: {len(train) + len(val) + len(test)} images")
    
    return train, val, test


# ============================================================
# STEP 4: Create balanced directory with links
# ============================================================
def create_split_dirs():
    """Create the balanced dataset directory structure."""
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(BALANCED_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BALANCED_DIR, "labels", split), exist_ok=True)


def link_files(items, split_name):
    """Create hard links for images and labels into the split directory."""
    img_dir = os.path.join(BALANCED_DIR, "images", split_name)
    lbl_dir = os.path.join(BALANCED_DIR, "labels", split_name)
    
    count = 0
    for img_path, lbl_path, img_name, _ in items:
        base_name = os.path.splitext(img_name)[0]
        img_ext = os.path.splitext(img_name)[1]
        
        dst_img = os.path.join(img_dir, img_name)
        dst_lbl = os.path.join(lbl_dir, base_name + ".txt")
        
        try:
            os.link(img_path, dst_img)
        except OSError:
            shutil.copy2(img_path, dst_img)
        
        try:
            os.link(lbl_path, dst_lbl)
        except OSError:
            shutil.copy2(lbl_path, dst_lbl)
        
        count += 1
        if count % 10000 == 0:
            print(f"  Linked {count} files to {split_name}...", flush=True)
    
    print(f"  Completed {split_name}: {count} image-label pairs")


# ============================================================
# STEP 5: Generate data.yaml
# ============================================================
def generate_yaml():
    yaml_content = """train: images/train
val: images/val
test: images/test

nc: 14
names:
  0: uniform_top
  1: uniform_bottom
  2: civilian_top_short_sleeve
  3: civilian_top_long_sleeve
  4: civilian_bottom_trousers
  5: civilian_bottom_shorts
  6: civilian_bottom_skirt
  7: footwear_shoes
  8: footwear_slippers
  9: prohibited_ripped_pants
  10: prohibited_leggings
  11: prohibited_sleeveless
  12: prohibited_crop_halter
  13: prohibited_midriff_offshoulder
"""
    yaml_path = os.path.join(BALANCED_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nGenerated {yaml_path}")


# ============================================================
# STEP 6: Verify counts per split
# ============================================================
def verify_split():
    """Count annotations per class per split and print report."""
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    
    grand_total = Counter()
    grand_files = 0
    
    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(BALANCED_DIR, "labels", split)
        img_dir = os.path.join(BALANCED_DIR, "images", split)
        
        counts = Counter()
        n_files = 0
        n_annotations = 0
        
        with os.scandir(lbl_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".txt"):
                    n_files += 1
                    with open(entry.path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    cls_id = int(parts[0])
                                    counts[cls_id] += 1
                                    n_annotations += 1
                                except ValueError:
                                    pass
        
        # Check image count matches label count
        n_images = sum(1 for _ in os.scandir(img_dir) if _.is_file())
        
        print(f"\n--- {split.upper()} ---")
        print(f"  Images: {n_images}  |  Labels: {n_files}  |  Match: {'✅' if n_images == n_files else '❌ MISMATCH'}")
        print(f"  Total annotations: {n_annotations}")
        for cls_id in sorted(counts.keys()):
            print(f"    Class {cls_id:2d}: {counts[cls_id]:6d}")
        
        grand_total.update(counts)
        grand_files += n_files
    
    print(f"\n--- GRAND TOTAL ---")
    print(f"  Total files: {grand_files}")
    total_annotations = sum(grand_total.values())
    print(f"  Total annotations: {total_annotations}")
    for cls_id in sorted(grand_total.keys()):
        print(f"    Class {cls_id:2d}: {grand_total[cls_id]:6d}")
    
    # Check for data leakage
    print("\n--- DATA LEAKAGE CHECK ---")
    all_names = set()
    duplicates = 0
    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(BALANCED_DIR, "labels", split)
        with os.scandir(lbl_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".txt"):
                    if entry.name in all_names:
                        duplicates += 1
                    all_names.add(entry.name)
    
    if duplicates == 0:
        print("  ✅ No data leakage detected — all files unique across splits")
    else:
        print(f"  ❌ LEAKAGE DETECTED: {duplicates} files appear in multiple splits!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    
    print("=" * 60)
    print("SUBSAMPLE & STRATIFIED SPLIT")
    print(f"  Subsample ratio for dataset_output: {SUBSAMPLE_RATIO}")
    print(f"  Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Seed: {SEED}")
    print("=" * 60)
    
    sources = gather_pairs()
    selected = subsample(sources)
    train, val, test = stratified_split(selected)
    
    print("\nCreating balanced dataset directories...")
    create_split_dirs()
    
    print("\nLinking files...")
    link_files(train, "train")
    link_files(val, "val")
    link_files(test, "test")
    
    generate_yaml()
    verify_split()
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f} seconds")
