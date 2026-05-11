# Enhanced UWear — Thesis Defense Cheat Sheet
> Quick-reference Q&A guide for the panel defense.  
> Based on actual project data and configuration.

---

## MODEL SELECTION

**Q: Why did you choose YOLOv11s specifically, out of all available object detection models?**

A: We selected YOLOv11s (small variant) for three reasons:
1. **Real-time requirement.** Our system must process live webcam feeds. YOLO is a single-stage detector that performs localization and classification in one forward pass, achieving 7.3ms per image (~137 FPS) on our hardware. Two-stage detectors like Faster R-CNN run significantly slower because they require a separate region proposal step.
2. **Hardware constraint.** Our training and deployment GPU is an NVIDIA GTX 1660 SUPER with only 6GB VRAM. The "small" variant (9.4M parameters, 21.3 GFLOPs) fits comfortably within this memory budget for both training (batch size 8) and inference. Larger variants like YOLOv11m or YOLOv11l would not fit.
3. **Accuracy-speed balance.** YOLOv11s achieved a 91.1% mAP@0.50, which exceeded our target threshold of 85%. Moving to a larger model variant would provide diminishing accuracy gains at the cost of speed and VRAM that we cannot afford.

---

**Q: Why YOLO11 and not YOLO8 or YOLO9?**

A: We evaluated all three versions during model selection. YOLOv11 was chosen based on three concrete advantages:

**1. Parameter efficiency (does more with less).**  
YOLOv11 achieves higher accuracy while using fewer parameters than YOLOv8 at the same model size. For example, YOLOv11m uses ~22% fewer parameters than YOLOv8m while matching or exceeding its mAP. This matters because our 6GB VRAM budget is tight — fewer parameters means faster training, lower memory usage, and faster inference.

**2. Architectural improvements (Why we switched from YOLOv8 to YOLOv11).**  
We actually began initial testing with YOLOv8, but we transitioned to YOLOv11 because of fundamental upgrades in how the model "looks" at an image:

*   **What YOLOv8 had (C2f Module):** YOLOv8 relied heavily on the C2f (Cross Stage Partial Bottleneck with 2 convolutions) module for feature extraction. While fast, it lacked the ability to deeply analyze very fine details or complex spatial relationships, which caused it to struggle with our visually similar classes (like telling a crop top apart from a normal short-sleeve shirt just by the hemline).
*   **What YOLOv11 has (C3k2 & C2PSA):** YOLOv11 replaced the C2f blocks with two major upgrades:
    *   **C3k2 Backbone Blocks:** A more advanced feature extractor that pulls richer, more detailed visual information (edges, textures, fabric folds) while using fewer computations than YOLOv8's C2f.
    *   **C2PSA (Cross-Stage Partial with Spatial Attention):** This is the biggest difference. YOLOv8 did not have a built-in spatial attention mechanism in its neck. YOLOv11's C2PSA forces the model to mathematically "pay attention" to the most relevant spatial regions of the image. For our project, this means the model explicitly focuses on the boundaries of the garments (hemlines, waistlines) rather than getting distracted by the background (walls, chairs, other students). This was the key to fixing YOLOv8's misclassification issues.

**3. Why not YOLOv9?**  
YOLOv9 introduced Programmable Gradient Information (PGI) and GELAN (Generalized Efficient Layer Aggregation Network), which are academically significant innovations for preventing information loss during training. However, YOLOv9 was developed outside the Ultralytics ecosystem, meaning it lacks the seamless integration with the Ultralytics training pipeline, export tools, and deployment utilities that our entire project relies on. Additionally, YOLOv9's inference speed is slower than YOLOv11 due to its heavier architecture — a critical disadvantage for our real-time webcam requirement.

**Summary comparison:**

| Aspect | YOLOv8 | YOLOv9 | YOLOv11 (ours) |
|---|---|---|---|
| Parameters (small) | 11.2M | 7.2M | **9.4M** |
| Focus | General-purpose | Gradient flow / accuracy | **Efficiency + accuracy** |
| Ecosystem | Ultralytics ✅ | External ❌ | Ultralytics ✅ |
| Inference speed | Fast | Slower (heavier arch) | **Fastest** |
| Deployment support | Excellent | Limited | **Excellent** |
| Release | Jan 2023 | Feb 2024 | **Oct 2024 (latest)** |

Since our project timeline (2025–2026) coincided with YOLOv11's maturity, it was the most current, most efficient, and best-supported release available.

---

**Q: Did you consider any other models besides YOLO?**

A: Yes, we evaluated three alternatives during the literature review:
- **Faster R-CNN** — Higher accuracy potential but too slow for real-time webcam inference (2–5 FPS vs our 137 FPS).
- **SSD (Single Shot Detector)** — Comparable speed but lower accuracy on small objects like footwear.
- **EfficientDet** — Strong accuracy but not optimized for the Ultralytics ecosystem we built our pipeline around.

YOLO was the only model that satisfied all three constraints: real-time speed, 6GB VRAM limit, and >85% mAP accuracy.

---

## DATASET & CLASS IMBALANCE

**Q: How did you handle the class imbalance in your dataset?**

A: The raw merged dataset had severe class imbalance. Here is the exact before-and-after with the actual numbers:

**Before balancing (raw dataset):**
- Largest class: `civilian_top_short_sleeve` = **71,644 annotations**
- Smallest class: `uniform_bottom` = **811 annotations**
- Imbalance ratio = 71,644 ÷ 811 = **88:1**

This is a critical problem because without correction, the model would be trained on 88 civilian top images for every 1 uniform bottom image, causing it to default to predicting "civilian top" for almost everything.

We addressed this with a **two-stage balancing strategy**:

**Stage 1 — Subsampling (cutting down the big classes):**  
We did not just guess a number — we wrote a script (`subsample_and_split.py`) programmed to keep exactly **25% (a 0.25 ratio)** of all images coming from the massive civilian datasets (DeepFashion2 and ModaNet), while keeping 100% of our manual uniform and violation images. 

By applying this 25% cap via random sampling, the dominant classes were cut down to a quarter of their original size. Here is the math of what was removed:

| Class | Raw Count | After 25% Subsampling | Images Removed |
|---|---|---|---|
| `civilian_top_short_sleeve` | 71,644 | 17,689 | **−53,955** |
| `civilian_top_long_sleeve` | 36,064 | 8,835 | **−27,229** |
| `civilian_bottom_trousers` | 55,387 | 13,600 | **−41,787** |
| `civilian_bottom_shorts` | 36,616 | 9,135 | **−27,481** |
| `prohibited_sleeveless` | 46,039 | 32,513 | **−13,526** |

*Note: The script grouped images by their dominant class to maintain a mathematically clean 70/20/10 train/validation/test split.*

**Stage 2 — Offline Augmentation (boosting the small classes):**  
After subsampling, the 4 smallest classes were still critically underrepresented. We generated synthetic copies using geometric (horizontal flips, ±15° rotation) and photometric (brightness, contrast) transformations until each reached 5,000 training samples:

| Class | Before Augmentation | After Augmentation | Added |
|---|---|---|---|
| `uniform_top` | 684 | 5,000 | **+4,316** |
| `uniform_bottom` | 565 | 5,000 | **+4,435** |
| `footwear_slippers` | 1,269 | 5,000 | **+3,731** |
| `prohibited_ripped_pants` | 1,449 | 5,000 | **+3,551** |

**Result after both stages:**  
- Largest training class: `civilian_top_short_sleeve` = **12,376**
- Smallest training class: `uniform_top` / `uniform_bottom` = **5,000** (post-augmentation)
- New ratio = 12,376 ÷ 5,000 = **~2.5:1**

This is a dramatic improvement from the original **88:1** down to **2.5:1**, giving the model a much more balanced view of all 14 garment classes during training.

---

**Q: Why didn't you make every class have exactly 5,000 images? Isn't a 2.5:1 ratio still imbalanced?**

A: In modern deep learning, especially with YOLO models, perfectly equal class counts (1:1 ratio) are actually unnecessary and can sometimes be harmful by forcing you to throw away good training data. A ratio of 2.5:1 is considered a "mild" imbalance, which YOLOv11 handles effortlessly through three built-in mechanisms:

1. **Classification Loss Weighting:** We explicitly set the `cls` loss weight to 2.0. This tells the neural network that getting a class prediction wrong costs twice as much, forcing it to pay attention to the minority classes even if they appear less frequently.
2. **Binary Cross-Entropy (BCE) Loss:** YOLO evaluates each class independently using BCE. It doesn't ask "is this a civilian top OR a uniform top?", it asks "is there a uniform top here? yes/no". This independent evaluation prevents dominant classes from overshadowing minority classes.
3. **Mosaic Augmentation:** During training, YOLO stitches 4 different images together into a single grid. This drastically increases the chance that a minority class (like a uniform) appears in the same training step as a majority class (like civilian clothes), naturally balancing the learning process per batch.

---

**Q: What train/validation/test split ratio did you use?**

A: We used a **70/20/10** split:
- Training set: 82,036 images (70%, includes augmented samples)
- Validation set: 18,854 images (20%)
- Test set: 9,443 images (10%)

The split was applied *before* augmentation to ensure that augmented copies of an image never leak into the validation or test sets.

---

**Q: Where did your dataset come from? Is it reliable?**

A: Our dataset was compiled from 8 publicly available, peer-reviewed sources:
- **DeepFashion2** (CVPR 2019) — civilian clothing classes
- **ModaNet** (ACM Multimedia 2018) — civilian clothing classes
- **Open Images V7** (Google) — prohibited clothing classes
- **Roboflow public datasets** — school uniform, footwear, and fashion violation images
- **USTP Philippine School Uniform dataset** — Philippine-context uniform images
- **Manual collection** — additional uniform images via Google Images for Philippine school context

All images were verified via an integrity audit of 94,300 images across all splits, and 2,190 malformed labels (segmentation polygons instead of YOLO bounding boxes) were programmatically corrected.

---

## OVERFITTING & BIAS

**Q: How can you ensure the model is not overfitting?**

A: We implemented five anti-overfitting measures:

1. **Separate validation set.** The 18,854-image validation set was strictly held out and never used during training. The mAP reported (0.911) is measured entirely on this unseen data.
2. **Early stopping with patience=20.** Training automatically halts if the validation mAP does not improve for 20 consecutive epochs. This prevents the model from memorizing training data.
3. **Heavy online augmentation.** Every training batch is dynamically augmented with mosaic (4-image composites), HSV color shifts, random flips, scaling, rotation, and MixUp. The model never sees the exact same image twice.
4. **Training vs. validation loss convergence.** The training curves show both training loss and validation loss decreasing in parallel without divergence, which would indicate overfitting.
5. **Independent test set evaluation.** Final metrics were confirmed on a completely separate 10% test split that was never used for model selection or hyperparameter tuning.

---

**Q: How do you know the model isn't just biased toward the majority classes?**

A: Three pieces of evidence confirm the model is not biased:

1. **Minority class performance.** The 4 augmented minority classes all achieved strong AP scores: `uniform_top` (0.992), `uniform_bottom` (0.995), `footwear_slippers` (0.960), `prohibited_ripped_pants` (0.884). If the model were biased, these classes would score near zero.
2. **Classification loss weight = 2.0.** We doubled the default classification loss penalty during training, which forces the model to pay extra attention to correctly classifying minority classes rather than defaulting to the majority prediction.
3. **Confusion matrix diagonal.** The normalized confusion matrix shows a strong diagonal across all 14 classes, including the minority ones. There is no systematic pattern of minority classes being absorbed into majority classes.

---

**Q: The `prohibited_crop_halter` class has a low F1-score of 0.63. Why?**

A: This is a **dataset quality issue**, not a model architecture issue. The source data for this class came from external fashion datasets where many images labeled as "crop tops" were actually standard civilian t-shirts photographed from certain angles. The confusion matrix clearly shows bidirectional misclassification between `prohibited_crop_halter` and `civilian_top_short_sleeve` — the model is confused because the training data itself is noisy.

As documented in our future work, remediating this class requires curating a high-quality, verified replacement set of actual crop top and halter images with correct labels.

---

## TRAINING PROCESS

**Q: How long did training take?**

A: The model trained for 57 epochs over approximately 13.5 hours (~14.2 minutes per epoch) on a single NVIDIA GTX 1660 SUPER GPU. Training was configured for 100 epochs with early stopping at patience=20, but was manually paused at epoch 58. The best weights were saved automatically at the peak performance point (epoch 57, mAP@0.50 = 0.911).

---

**Q: What loss functions did you use?**

A: YOLOv11 uses a composite loss with three components:
1. **CIoU Loss (box regression)** — Measures how well predicted bounding boxes align with ground truth, penalizing center distance and aspect ratio mismatch.
2. **Binary Cross-Entropy (classification)** — Measures the error in class predictions. We set the classification loss weight to 2.0 (double the default) to emphasize correct garment classification.
3. **Distribution Focal Loss (coordinate regression)** — Enables fine-grained bounding box coordinate prediction.

---

**Q: What optimizer did you use?**

A: Stochastic Gradient Descent (SGD) with momentum of 0.937. The learning rate followed a cosine annealing schedule from 0.01 (initial) to 0.0001 (final), with a 3-epoch linear warmup to prevent destructive gradient updates during early training.

---

## EVALUATION METRICS

**Q: What does mAP@0.50 mean?**

A: Mean Average Precision at IoU threshold 0.50. A detection is counted as correct if the predicted bounding box overlaps with the ground truth box by at least 50% (measured by Intersection over Union). The AP is computed per class as the area under the Precision-Recall curve, and mAP is the average of all 14 class AP scores. Our model achieved 0.911 (91.1%).

---

**Q: What is the difference between mAP@0.50 and mAP@0.50-0.95?**

A: mAP@0.50 uses a single, lenient overlap threshold (50%). mAP@0.50-0.95 averages across 10 thresholds from 50% to 95% in steps of 5%. It is a much stricter metric because at 95% overlap, the bounding box must almost perfectly match the ground truth. Our model scored 0.911 at @0.50 and 0.736 at @0.50-0.95, which is a typical and healthy gap.

---

**Q: The DLSU-D handbook has specific length rules, like skirts being "no more than 2 inches above the knee." How can an AI model detect something that specific?**

A: While the AI doesn't have a physical ruler, YOLOv11 is uniquely suited for these "fine-grained" checks due to its superior **Localization Precision**. 

1. **Distribution Focal Loss (DFL):** This is a core architectural upgrade in YOLOv11 that allows the model to predict the boundaries of a bounding box with much higher accuracy than older versions. Instead of just guessing a box, it learns a probability distribution for the edges, allowing it to pinpoint exactly where a skirt or short ends relative to the knee.
2. **Spatial Attention (C2PSA):** This mechanism helps the model understand the spatial relationship between different parts of the body. It "attends" to the distance between the hemline of the garment and the detected footwear/leg area, allowing it to accurately classify an item as a "short" (prohibited if too high) versus "trousers" (compliant).
3. **Class-Based Policy Mapping:** We trained the model on a diverse dataset that included various lengths of clothing. By accurately classifying an item as `civilian_bottom_shorts` or `civilian_bottom_skirt` based on its detected boundaries, the **Rule Engine** can then automatically apply the handbook's specific length policy for that category.

In short, YOLOv11's "intelligence" isn't in measuring inches, but in its ability to highly accurately **localize** the edges of garments, which is the foundational requirement for enforcing these handbook rules.

---

**Q: What confidence threshold do you use in the live deployment?**

A: Based on the F1-Confidence curve, the optimal confidence threshold is 0.306 (30.6%), where the model achieves its peak F1-score of 0.86. This provides the best balance between catching violations (recall) and avoiding false alarms (precision). In the live web UI, the default threshold is set to 0.25 to slightly favor recall (catching more violations) at the cost of occasional false positives.

---

## SYSTEM ARCHITECTURE

**Q: How does the rule engine work? Why not just use the AI model directly?**

A: The YOLO model only answers "what garment is this person wearing?" It does not understand institutional policy. The rule engine is a separate, deterministic algorithm that takes YOLO's detections as input and evaluates them against Section 27.1.2 of the DLSU-D Student Handbook. This separation provides two key benefits:
1. **Policy updates without retraining.** If the university changes the dress code (e.g., allows sleeveless tops), we only modify the rule engine code — no expensive model retraining needed.
2. **Context-aware evaluation.** The rule engine considers the day of the week and student year level to determine whether UNIFORM_MODE or CIVILIAN_MODE applies. The AI model has no concept of schedules.

---

**Q: Can the system work offline?**

A: Yes. The YOLO model runs entirely on the local GPU. No internet connection is required for inference. The only network dependency is the optional integration with Module 2 (SWAFO Web App) for recording violations to the institutional database.

---

**Q: What is the inference speed in production?**

A: 7.3ms per image (approximately 137 frames per second). The full pipeline breakdown is:
- Preprocessing: 1.1ms
- Model inference: 7.3ms
- Post-processing (NMS): 1.0ms
- **Total: ~9.4ms per frame**

This is well above the 30 FPS minimum required for smooth real-time webcam processing.

---

## COMPARISON WITH BASELINE

**Q: How does Enhanced UWear compare to the original UWear system?**

A: Enhanced UWear surpasses the UWear baseline across all metrics:

| Metric | Enhanced UWear | UWear Baseline | Improvement |
|---|---|---|---|
| mAP@0.50 | **0.911** | 0.788 | +12.3% |
| Precision | **0.865** | 0.734 | +13.1% |
| Recall | **0.861** | 0.788 | +7.3% |
| Classes | **14** | 8 | +6 new classes |

Key improvements: expanded from 8 to 14 garment classes, added 5 prohibited clothing categories mapped to specific handbook sections, and implemented the rule-based compliance engine for automated policy evaluation.

---

## LIMITATIONS & FUTURE WORK

**Q: What are the known limitations of this system?**

A: Three primary limitations:
1. **Visually similar garments.** The model struggles to distinguish between civilian polo shirts and uniform polo shirts because they share identical structural features (collar, buttons, short sleeves). Differentiating them requires logo-level detection, which is beyond the current model's resolution.
2. **Dataset noise in prohibited classes.** The `prohibited_crop_halter` and `prohibited_midriff_offshoulder` classes contain mislabeled samples from external datasets, reducing their detection accuracy.
3. **Subtle violations.** Minor rips in jeans or slightly above-knee skirt lengths are difficult for the model to detect because these features occupy very few pixels at standard webcam resolution.

---

**Q: What would you do differently if you had more time?**

A: Three improvements:
1. **Local data collection.** Photograph actual DLSU-D students (with consent) wearing various compliant and non-compliant outfits to create a campus-specific dataset that eliminates reliance on external fashion datasets.
2. **Two-stage detection.** Add a secondary lightweight model specifically trained to detect the DLSU-D uniform logo/embroidery, solving the polo shirt confusion problem.
3. **Full training cycle.** Complete the interrupted 100-epoch training run with a cleaned dataset (removing noisy crop halter samples) to maximize the model's potential accuracy.

---

*End of Defense Cheat Sheet.*
