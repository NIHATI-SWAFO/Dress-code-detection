import os
import sys
from collections import Counter

def count_classes_in_dir(label_dir):
    counts = Counter()
    if not os.path.exists(label_dir):
        print(f"Directory not found: {label_dir}")
        return counts
    
    print(f"Scanning {label_dir}...", flush=True)
    i = 0
    with os.scandir(label_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".txt"):
                i += 1
                if i % 10000 == 0:
                    print(f"  Processed {i} files...", flush=True)
                
                try:
                    with open(entry.path, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    counts[class_id] += 1
                                except ValueError:
                                    continue
                except Exception as e:
                    print(f"Error reading {entry.path}: {e}")
                    
    print(f"Total files processed in this dir: {i}")
    return counts

base_path = r"c:\Users\timde\Downloads\dresscode_detection\final datasets"
dirs_to_check = {
    "DressCode_Violations_v2": os.path.join(base_path, "DressCode_Violations_v2", "labels"),
    "Final Dataset.yolov8 (train)": os.path.join(base_path, "Final Dataset.yolov8", "train", "labels"),
    "dataset_output": os.path.join(base_path, "dataset_output", "labels")
}

total_counts = Counter()

for name, path in dirs_to_check.items():
    print(f"\n--- {name} ---", flush=True)
    counts = count_classes_in_dir(path)
    total_counts.update(counts)
    
    print(f"Counts for {name}:")
    for class_id, count in sorted(counts.items()):
        print(f"  Class {class_id}: {count}")

print("\n--- FINAL TOTAL ACROSS ALL 3 FOLDERS ---")
for class_id, count in sorted(total_counts.items()):
    print(f"  Class {class_id}: {count}")
