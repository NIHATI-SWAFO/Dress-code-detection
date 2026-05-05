import os
import shutil
from collections import Counter
import time

BASE_PATH = r"c:\Users\timde\Downloads\dresscode_detection\final datasets"
MERGED_DIR = os.path.join(BASE_PATH, "merged_dataset")
MERGED_IMAGES = os.path.join(MERGED_DIR, "images", "train")
MERGED_LABELS = os.path.join(MERGED_DIR, "labels", "train")

# Define the datasets and their mappings
DATASETS = [
    {
        "name": "dataset_output",
        "labels_dir": os.path.join(BASE_PATH, "dataset_output", "labels"),
        "images_dir": os.path.join(BASE_PATH, "dataset_output", "images"),
        "prefix": "dataset_output_",
        "mapping": {
            4: 2,
            5: 3,
            6: 4,
            7: 5,
            8: 6,
            13: 11
        }
    },
    {
        "name": "Final Dataset.yolov8 (train)",
        "labels_dir": os.path.join(BASE_PATH, "Final Dataset.yolov8", "train", "labels"),
        "images_dir": os.path.join(BASE_PATH, "Final Dataset.yolov8", "train", "images"),
        "prefix": "uniform_",
        "mapping": {
            0: 7,
            1: 8,
            2: 1,
            3: 0
        }
    },
    {
        "name": "DressCode_Violations_v2",
        "labels_dir": os.path.join(BASE_PATH, "DressCode_Violations_v2", "labels"),
        "images_dir": os.path.join(BASE_PATH, "DressCode_Violations_v2", "images"),
        "prefix": "violations_",
        "mapping": {
            11: 9,
            12: 10,
            13: 11,
            14: 12,
            15: 13
        }
    }
]

EXPECTED_COUNTS = {
    0: 1018,
    1: 811,
    2: 71644,
    3: 36064,
    4: 55387,
    5: 36616,
    6: 30835,
    7: 12967,
    8: 4054,
    9: 2071,
    10: 5013,
    11: 46039,
    12: 7614,
    13: 5464
}

def create_dirs():
    os.makedirs(MERGED_IMAGES, exist_ok=True)
    os.makedirs(MERGED_LABELS, exist_ok=True)

def link_or_copy(src, dst):
    if not os.path.exists(src):
        return False
    try:
        # Try hardlink first (instant, 0 extra space)
        os.link(src, dst)
        return True
    except OSError:
        try:
            # Fallback to copy if hardlink not supported/allowed
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"Failed to copy {src}: {e}")
            return False

def generate_yaml():
    yaml_path = os.path.join(MERGED_DIR, "data.yaml")
    yaml_content = """train: images/train
val: images/train

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
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print("Generated data.yaml")

def process_dataset(ds):
    print(f"\nProcessing {ds['name']}...")
    prefix = ds['prefix']
    mapping = ds['mapping']
    labels_dir = ds['labels_dir']
    images_dir = ds['images_dir']
    
    if not os.path.exists(labels_dir):
        print(f"Warning: labels_dir not found: {labels_dir}")
        return
        
    i = 0
    with os.scandir(labels_dir) as entries:
        for entry in entries:
            if not (entry.is_file() and entry.name.endswith(".txt")):
                continue
                
            base_name = entry.name[:-4] # remove .txt
            new_base_name = f"{prefix}{base_name}"
            new_label_path = os.path.join(MERGED_LABELS, f"{new_base_name}.txt")
            
            # Read, map, and write label file
            try:
                new_lines = []
                with open(entry.path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                orig_id = int(parts[0])
                                # If mapping exists, use it, else keep original (but print warning)
                                if orig_id in mapping:
                                    new_id = mapping[orig_id]
                                else:
                                    print(f"Warning: unknown ID {orig_id} in {entry.name}")
                                    new_id = orig_id
                                new_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                                new_lines.append(new_line)
                            except ValueError:
                                pass # skip invalid lines
                                
                with open(new_label_path, 'w') as f:
                    f.writelines(new_lines)
            except Exception as e:
                print(f"Error processing label {entry.path}: {e}")
                continue

            # Now find and copy/link the corresponding image file
            # Common extensions
            img_found = False
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                src_img = os.path.join(images_dir, f"{base_name}{ext}")
                if os.path.exists(src_img):
                    dst_img = os.path.join(MERGED_IMAGES, f"{new_base_name}{ext}")
                    link_or_copy(src_img, dst_img)
                    img_found = True
                    break
            
            if not img_found:
                # Silently ignore missing images or print warning?
                # print(f"Warning: No image found for {base_name}")
                pass
                
            i += 1
            if i % 10000 == 0:
                print(f"  Processed {i} files...", flush=True)

    print(f"Completed {ds['name']}: {i} files processed.")

def verify_counts():
    print("\nVerifying final counts...")
    final_counts = Counter()
    total_files = 0
    total_annotations = 0
    
    with os.scandir(MERGED_LABELS) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".txt"):
                total_files += 1
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

    print(f"\nFinal File Count: {total_files}")
    print(f"Final Annotation Count: {total_annotations}\n")
    
    success = True
    for class_id in sorted(EXPECTED_COUNTS.keys()):
        expected = EXPECTED_COUNTS[class_id]
        actual = final_counts.get(class_id, 0)
        status = "MATCH" if expected == actual else f"MISMATCH (Expected: {expected})"
        print(f"Class {class_id:2d}: {actual:6d} - {status}")
        if expected != actual:
            success = False
            
    if success and total_files == 204927 and total_annotations == 315587: # User had 315,597 typo in prompt, but expected counts sum to 315,587. Let's verify sum.
        expected_sum = sum(EXPECTED_COUNTS.values())
        print(f"\nSuccess! All counts perfectly match the expected sum: {expected_sum}")
    else:
        expected_sum = sum(EXPECTED_COUNTS.values())
        print(f"\nFailed! Mismatch found. Expected Total Annotations: {expected_sum}, Actual: {total_annotations}")

if __name__ == "__main__":
    start_time = time.time()
    create_dirs()
    
    for ds in DATASETS:
        process_dataset(ds)
        
    generate_yaml()
    verify_counts()
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
