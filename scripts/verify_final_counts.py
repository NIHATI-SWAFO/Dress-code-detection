import os
from collections import Counter
import time

MERGED_LABELS = r"c:\Users\timde\Downloads\dresscode_detection\final datasets\merged_dataset\labels\train"

def verify_counts():
    print(f"Scanning labels in {MERGED_LABELS}...\n")
    final_counts = Counter()
    total_files = 0
    total_annotations = 0
    
    start_time = time.time()
    
    with os.scandir(MERGED_LABELS) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".txt"):
                total_files += 1
                if total_files % 20000 == 0:
                    print(f"  Scanned {total_files} files...", flush=True)
                    
                with open(entry.path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                final_counts[class_id] += 1
                                total_annotations += 1
                            except ValueError:
                                pass

    print(f"\nScan completed in {time.time() - start_time:.2f} seconds.")
    print("-" * 30)
    print(f"Final File Count: {total_files}")
    print(f"Final Annotation Count: {total_annotations}")
    print("-" * 30)
    
    print("\nClass Counts:")
    for class_id in sorted(final_counts.keys()):
        count = final_counts[class_id]
        print(f"  Class {class_id:2d}: {count:6d}")

if __name__ == "__main__":
    verify_counts()
