# Enhanced UWear — Model Evaluation & Data Results
> Module 1: Dress Code Detection System  
> Evaluation metrics based on the final YOLO11s model (`best.pt`, Epoch 57).

---

## 1. Overall Performance Metrics

The model achieved an overall Mean Average Precision (mAP) of **91.1%** at an IoU threshold of 0.50 across all 14 classes on a validation set of 25,773 bounding boxes.

| Metric | Score | Description |
|---|---|---|
| **mAP@0.50** | `0.911` | Average accuracy across all 14 classes |
| **mAP@0.50-0.95** | `0.736` | Strict accuracy across varying overlap thresholds |
| **Precision** | `0.865` | Out of all detections made, 86.5% were correct |
| **Recall** | `0.861` | Out of all actual garments, 86.1% were successfully found |

---

## 2. Class-by-Class Classification Report

| Class | Precision | Recall | F1-Score | Instances |
|---|:---:|:---:|:---:|:---:|
| `uniform_top` | 1.00 | 0.99 | 0.99 | 198 |
| `uniform_bottom` | 0.98 | 1.00 | 0.99 | 169 |
| `civilian_top_short_sleeve` | 0.85 | 0.92 | 0.89 | 3,608 |
| `civilian_top_long_sleeve` | 0.82 | 0.81 | 0.81 | 1,765 |
| `civilian_bottom_trousers` | 0.94 | 0.92 | 0.93 | 2,748 |
| `civilian_bottom_shorts` | 0.91 | 0.89 | 0.90 | 1,832 |
| `civilian_bottom_skirt` | 0.84 | 0.86 | 0.85 | 1,527 |
| `footwear_shoes` | 0.88 | 0.78 | 0.83 | 2,535 |
| `footwear_slippers` | 0.92 | 0.92 | 0.92 | 807 |
| `prohibited_ripped_pants` | 0.74 | 0.93 | 0.82 | 414 |
| `prohibited_leggings` | 0.87 | 0.85 | 0.86 | 1,002 |
| `prohibited_sleeveless` | 0.87 | 0.87 | 0.87 | 6,554 |
| `prohibited_crop_halter` | 0.69 | 0.58 | 0.63 | 1,522 |
| `prohibited_midriff_offshoulder` | 0.83 | 0.74 | 0.78 | 1,092 |
| **Weighted Average** | **0.86** | **0.86** | **0.86** | **25,773** |

*(Note: The lower F1-score for `prohibited_crop_halter` is attributed to mislabeling in the source dataset, where generic t-shirts were incorrectly labeled as crop tops, causing confusion with `civilian_top_short_sleeve`.)*

---

## 3. Confusion Matrices

The confusion matrix shows exactly where the model gets confused between two different classes. A perfect model would have a dark blue diagonal line and zero everywhere else.

### Normalized Confusion Matrix (Percentages)
Shows the percentage of correct predictions vs incorrect predictions for each class.
![Normalized Confusion Matrix](results/val_matrix/confusion_matrix_normalized.png)

### Absolute Confusion Matrix (Raw Counts)
Shows the exact number of garments classified.
![Absolute Confusion Matrix](results/val_matrix/confusion_matrix.png)

---

## 4. Evaluation Curves

### F1-Confidence Curve
The F1-score is the harmonic mean of precision and recall. This curve shows that the model achieves its peak overall performance (F1 = 0.86) when the detection confidence threshold is set to **0.306**.
![F1-Confidence Curve](results/val_matrix/BoxF1_curve.png)

### Precision-Recall (PR) Curve
The PR curve plots Precision against Recall. The larger the area under the curve (AUC), the better the model. An ideal model perfectly hugs the top-right corner. Our model achieves an impressive **0.911 mAP@0.50**.
![Precision-Recall Curve](results/val_matrix/BoxPR_curve.png)

### Precision-Confidence Curve
Shows how precision increases as you require higher confidence. At a confidence threshold of **0.872**, every prediction the model makes is 100% correct (Precision = 1.00).
![Precision Curve](results/val_matrix/BoxP_curve.png)

### Recall-Confidence Curve
Shows how recall drops as you require higher confidence. If you set the confidence threshold to 0.00, the model finds 97% of all garments, but it will also generate many false positives.
![Recall Curve](results/val_matrix/BoxR_curve.png)

---
*End of Data Results Report.*
