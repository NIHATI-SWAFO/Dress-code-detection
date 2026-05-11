# Enhanced UWear — Algorithm Used for Garment Detection
> Module 1: Dress Code Detection System  
> DLSU-D SWAFO Violation Management

---

## YOLOv11 (You Only Look Once, Version 11)

**Purpose:** Real-time object detection — identifies and classifies 14 garment types in a single image.

**How it works:**  
YOLO is a **single-stage detector** that performs detection and classification in **one forward pass** through a convolutional neural network. The input image (resized to 640×640) is passed through three main stages:

1. **Backbone (CSPDarknet)** — Extracts visual features from the image (edges, textures, shapes, patterns) using convolutional layers organized in C3k2 blocks.
2. **Neck (SPPF + C2PSA)** — Aggregates features at multiple scales so the model can detect both large garments (full torso tops) and small garments (slippers at the bottom of the frame).
3. **Head (Decoupled Head)** — Outputs bounding box coordinates and class probabilities for each detected garment.

The model predicts, for every region of the image:
- **Where** the garment is (bounding box: x, y, width, height)
- **What** the garment is (class probability across 14 categories)
- **How confident** it is (objectness score)

### Model Specifications

| Specification | Value |
|---|---|
| Variant | YOLOv11s (small) |
| Parameters | 9,418,218 |
| Input resolution | 640 × 640 pixels |
| Inference speed | 7.3ms per image (~137 FPS) |
| Accuracy | 91.1% mAP@0.50 |
| Training data | 94,413 labeled images across 14 classes |
| Hardware | NVIDIA GTX 1660 SUPER (6GB VRAM) |

### 14 Detected Garment Classes

| ID | Class | Category |
|---|---|---|
| 0 | uniform_top | Uniform |
| 1 | uniform_bottom | Uniform |
| 2 | civilian_top_short_sleeve | Civilian |
| 3 | civilian_top_long_sleeve | Civilian |
| 4 | civilian_bottom_trousers | Civilian |
| 5 | civilian_bottom_shorts | Civilian |
| 6 | civilian_bottom_skirt | Civilian |
| 7 | footwear_shoes | Footwear |
| 8 | footwear_slippers | Footwear |
| 9 | prohibited_ripped_pants | Prohibited |
| 10 | prohibited_leggings | Prohibited |
| 11 | prohibited_sleeveless | Prohibited |
| 12 | prohibited_crop_halter | Prohibited |
| 13 | prohibited_midriff_offshoulder | Prohibited |

---

*End of Algorithm Specification.*
