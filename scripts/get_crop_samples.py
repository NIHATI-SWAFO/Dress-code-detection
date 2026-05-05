import os
import cv2
import glob

artifacts_dir = r"C:\Users\timde\.gemini\antigravity\brain\e97cc1ba-dabb-4f52-be41-71dec69977b3"
dataset_dir = r"c:\Users\timde\Downloads\dresscode_detection\dataset\balanced"
labels_dir = os.path.join(dataset_dir, "labels", "train")
images_dir = os.path.join(dataset_dir, "images", "train")

found = 0
for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    has_class_12 = False
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        if int(parts[0]) == 12:
            has_class_12 = True
            boxes.append([float(x) for x in parts[1:5]])
            
    if has_class_12:
        img_name = os.path.basename(label_file).replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, os.path.basename(label_file).replace(".txt", ".png"))
            
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            for box in boxes:
                cx, cy, bw, bh = box
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, "prohibited_crop_halter", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            out_path = os.path.join(artifacts_dir, f"crop_sample_{found}.jpg")
            cv2.imwrite(out_path, img)
            print(f"Generated {out_path}")
            found += 1
            if found >= 10:
                break

print(f"Total found: {found}")
