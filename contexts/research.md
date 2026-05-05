# research.md — Module 1: Enhanced UWear Dress Code Detection System
> Project: SWAFO Violation Management Web App + Enhanced UWear
> Institution: De La Salle University-Dasmariñas (DLSU-D)
> Last Updated: May 2026
> Status: Dataset assembled — entering training phase

---

## 1. PROJECT OVERVIEW

Module 1 is an enhancement of UWear, a prior DLSU-D thesis (2024) that used YOLOv5 for general dress code detection. The enhanced system replaces and significantly expands UWear with a policy-aware, full-body dress code compliance detection model trained on 14 classes covering uniform attire, civilian attire, footwear, and prohibited clothing items — aligned with the DLSU-D School Uniform and Dress Code Policy (effective September 1, 2025).

The end goal is a trained YOLO11 object detection model that:
- Detects what a student is wearing (top, bottom, footwear)
- Classifies it against the current institutional dress code policy
- Determines compliance or violation based on day type and year level
- Outputs violation flags that feed into Module 2 (SWAFO Web App)

---

## 2. PRIOR WORK — UWEAR (2024)

| Field | Detail |
|---|---|
| Full Title | UWear: A Real-Time Dress Code Compliance System for College Students at DLSU-D Using YOLO-v5 |
| Authors | Rafael Jan M. Dancel, Angel Gabriel T. Kahulugan, Caleb Emmanuel C. Viernes |
| Adviser | Rolando B. Barrameda |
| Year | 2024 |
| Model | YOLOv5 |
| Type | Local Python application |
| Tools | OpenCV, PyTorch, Roboflow |

### UWear Best Results (Epoch 45)
| Metric | Value |
|---|---|
| mAP@0.5 | 0.7883 |
| Precision | 0.7336 |
| Recall | 0.7881 |

### UWear's Actual Trained Classes (from confusion matrix)
| Class | Performance |
|---|---|
| valid_top | Good |
| Cropped Top | Often misclassified as valid_top |
| Trousers | Best — 0.99 accuracy |
| Ripped Pants | 0.81 accuracy |
| Shorts | 0.78 accuracy |
| Skirt | Worst — misidentified as Background |

### UWear Critical Limitations
- No uniform-specific detection — all classes were general civilian clothing
- No policy-aware compliance logic (no concept of uniform day vs civilian day)
- No year-level awareness (1st–3rd year vs 4th year rules not implemented)
- Skirt class severely underperformed — misidentified as background
- No footwear detection
- Predates the September 1, 2025 DLSU-D Uniform Policy update
- Standalone local Python app — no integration with any records or enforcement system
- Dataset not aligned with current institutional policy conditions

---

## 3. RESEARCH GAPS ADDRESSED

| Gap | UWear | Enhanced UWear |
|---|---|---|
| Uniform-specific detection | ❌ None | ✅ uniform_top, uniform_bottom |
| Policy-aware compliance | ❌ None | ✅ UNIFORM_MODE vs CIVILIAN_MODE |
| Uniform day vs civilian day | ❌ Not implemented | ✅ Core feature |
| 4th year exception | ❌ Not implemented | ✅ Year-level rule engine |
| Footwear detection | ❌ Excluded | ✅ footwear_shoes, footwear_slippers |
| Skirt reliability | ❌ Worst class | ✅ Improved with more samples |
| Prohibited item coverage | ⚠️ Partial (2 classes) | ✅ 5 prohibited classes |
| Sept 2025 policy alignment | ❌ Predates policy | ✅ Fully aligned |
| Module 2 integration | ❌ Standalone | ✅ Feeds violation recording |

---

## 4. DRESS CODE POLICY REFERENCE

### Uniform Schedule (September 1, 2025 Policy)
| Year Level | Mon | Tue | Wed | Thu | Fri | Sat |
|---|---|---|---|---|---|---|
| 1st–3rd Year | Uniform | Uniform | Wash Day | Uniform | Uniform | Wash Day |
| 4th Year | Civilian | Civilian | Wash Day | Civilian | Civilian | Wash Day |

> 4th year students may opt to wear uniform on civilian days — this is always compliant.

### Type C Uniform (Daily Standard — Uniform Days)
| Component | Male | Female |
|---|---|---|
| Top | White Japanese cotton polo shirt with De La Salle embroidery | White fitted blouse with De La Salle embroidery |
| Bottom | Light khaki Dockers twill pants or short pants | Light khaki Dockers twill pants or skorts |
| Footwear | Low-cut shoes (topsiders or sneakers) with white socks | Low-cut shoes (topsiders or sneakers) with white socks |

### Prohibited Items (Section 27.1.2 — Minor Offense)
| Code | Prohibited Item |
|---|---|
| 27.1.2.2 | Ripped jeans exposing skin |
| 27.1.2.3 | Shorts more than 3 inches above kneecap |
| 27.1.2.4 | Sleeveless blouses / sling tops |
| 27.1.2.5 | Plunging neckline or backless blouses/dresses |
| 27.1.2.6 | Midriffs, hanging blouses, off-shoulder tops |
| 27.1.2.7 | Skirts/dresses more than 2 inches above knee |
| 27.1.2.8 | Leggings, jeggings, overly tight clothing |
| 27.1.2.9 | Halter blouses, crop tops |
| 27.1.2.11 | Slippers (except rain/injury — officer judgment) |

---

## 5. DETECTION CLASSES — 14-CLASS MASTER

> Male and female uniforms are merged into single functional classes.
> The model learns the shape and color of the uniform regardless of gender.
> This is more effective for real-world campus monitoring at DLSU-D.

| Class ID | Class Name | Description | Object Count |
|---|---|---|---|
| 0 | `uniform_top` | White polo shirt / fitted blouse with De La Salle embroidery | 1,018 |
| 1 | `uniform_bottom` | Light khaki pants, short pants, or skorts | 811 |
| 2 | `civilian_top_short_sleeve` | Short sleeve shirt, t-shirt, polo, Lasallian shirt | 71,644 |
| 3 | `civilian_top_long_sleeve` | Long sleeve shirt, sweatshirt, long-sleeved polo | 36,064 |
| 4 | `civilian_bottom_trousers` | Standard pants/trousers (non-ripped, non-tight) | 55,387 |
| 5 | `civilian_bottom_shorts` | Shorts (≤3 inches above kneecap = valid) | 36,616 |
| 6 | `civilian_bottom_skirt` | Skirt or dress bottom (≤2 inches above knee = valid) | 30,835 |
| 7 | `footwear_shoes` | Sneakers, topsiders, low-cut shoes | 12,967 |
| 8 | `footwear_slippers` | Any type of slippers or open sandals | 4,054 |
| 9 | `prohibited_ripped_pants` | Ripped jeans exposing skin | 2,071 |
| 10 | `prohibited_leggings` | Leggings, jeggings, overly tight pants | 5,013 |
| 11 | `prohibited_sleeveless` | Sling tops, tank tops, spaghetti straps, halter necks | 46,039 |
| 12 | `prohibited_crop_halter` | Crop tops, halter blouses/dresses | 7,614 |
| 13 | `prohibited_midriff_off-shoulder` | Midriff tops, off-shoulder tops, hanging blouses | 5,464 |
| **TOTAL** | | | **315,597** |

---

## 6. DATASET STATUS

| Aspect | Status |
|---|---|
| Total object labels | 315,597 |
| Preprocessing | ❌ Not yet applied |
| Augmentation | ❌ Not yet applied |
| Primary data collection | ❌ Pending ethics approval |
| Secondary dataset merge | ✅ Complete |
| Class remapping | ✅ Complete |

### Dataset Sources
| Source | Classes Contributed |
|---|---|
| DeepFashion2 | civilian_top_*, civilian_bottom_*, prohibited_sleeveless |
| ModaNet | civilian_top_*, civilian_bottom_* |
| Roboflow School Uniform datasets | uniform_top, uniform_bottom |
| USTP School Uniform (Philippine) | uniform_top, uniform_bottom |
| Roboflow Shoes Dataset | footwear_shoes, footwear_slippers |
| Open Images V7 | civilian_bottom_*, prohibited_* |
| Roboflow Clothing/Fashion datasets | prohibited_* classes |
| Manual image collection (Google Images) | uniform_top, uniform_bottom |

### Class Imbalance Warning
| Class Group | Object Count | % of Total | Action Required |
|---|---|---|---|
| Uniform (2 classes) | 1,024 | 0.3% | ⚠️ Apply heavy augmentation (5x) |
| Civilian (5 classes) | 230,546 | 77.7% | ⚠️ Cap at 5,000 per class during training |
| Footwear (2 classes) | 17,026 | 5.7% | ✅ Acceptable |
| Prohibited (5 classes) | 48,121 | 16.2% | ✅ Acceptable |

> **Critical:** Uniform classes MUST be augmented aggressively before or during training.
> Civilian classes MUST be capped/sampled to prevent model bias.

---

## 7. COMPLIANCE ASSESSMENT LOGIC (Post-Detection Rule Engine)

### UNIFORM_MODE — Uniform Days (Mon/Tue/Thu/Fri, 1st–3rd Year)

```
COMPLIANT if:
  ✅ uniform_top detected
  ✅ uniform_bottom detected
  ✅ footwear_shoes detected
  ✅ No prohibited class detected

NON-COMPLIANT flags:
  ❌ uniform_top not detected → "Not in proper uniform top"
  ❌ uniform_bottom not detected → "Not in proper uniform bottom"
  ❌ civilian top/bottom detected → "Not in uniform on uniform day"
  ❌ footwear_slippers detected → "Slippers prohibited (Sec. 27.1.2.11)"
  ❌ Any prohibited class detected → specific violation flag
```

### CIVILIAN_MODE — Wash Days (Wed/Sat, All) + 4th Year (All Uniform Days)

```
COMPLIANT if:
  ✅ civilian_top (any) detected
  ✅ civilian_bottom (any) detected
  ✅ footwear_shoes detected
  ✅ No prohibited class detected
  [shorts] length ≤ 3 inches above kneecap
  [skirt] length ≤ 2 inches above knee

NON-COMPLIANT flags:
  ❌ footwear_slippers detected → "Slippers prohibited (Sec. 27.1.2.11)"
  ❌ Any prohibited class detected → specific violation flag
  [uniform top/bottom detected on civilian day] → STILL COMPLIANT (opt-in allowed)
```

### Exceptions (Officer Override — Not Automated)
- NSTP/PE shirt on NSTP/PE days
- PE shorts within PE class area
- Approved written exception from OSS Dean or SWAFO Director
- Slippers during heavy rain or injury

---

## 8. INTEGRATION WITH MODULE 2 (SWAFO WEB APP)

When a violation is detected by Module 1, it outputs:

```json
{
  "student_id": "scanned_barcode",
  "detection_mode": "UNIFORM_MODE",
  "detected_classes": ["civilian_top_short_sleeve", "civilian_bottom_trousers", "footwear_shoes"],
  "compliance_status": "NON_COMPLIANT",
  "violations": [
    {
      "class": "civilian_top_short_sleeve",
      "handbook_ref": "27.1.2",
      "description": "Not in prescribed school uniform on uniform day"
    }
  ],
  "timestamp": "2026-05-01T08:34:00",
  "location": "JFH Building Entrance"
}
```

This JSON output feeds directly into Module 2's violation recording endpoint — the officer confirms, adds evidence, and submits. No manual re-encoding of the violation type needed.

---

## 9. EVALUATION TARGETS (Must Beat UWear)

| Metric | UWear Baseline | Your Target |
|---|---|---|
| mAP@0.5 | 0.7883 | > 0.85 |
| Precision | 0.7336 | > 0.80 |
| Recall | 0.7881 | > 0.85 |
| mAP@0.5:0.95 | Not reported | Report this additionally |
| Skirt class AP | Worst (near 0) | > 0.70 |
| Uniform class AP | N/A (not in UWear) | > 0.75 |
| Footwear class AP | N/A (not in UWear) | > 0.80 |

---

## 10. LENGTH MEASUREMENT — SHORTS & SKIRT (Required by Handbook)

### Policy Basis
| Code | Rule |
|---|---|
| 27.1.2.3 | Shorts must NOT exceed 3 inches above kneecap |
| 27.1.2.7 | Skirts/dresses must NOT exceed 2 inches above knee |

### Implementation Approach — Keypoint-Based Length Estimation

After YOLO11 detects `civilian_bottom_shorts` or `civilian_bottom_skirt`, a secondary measurement step runs using **pose estimation** to determine if the length violates the rule.

**Tool: YOLOv8-Pose or YOLO11-Pose (same Ultralytics ecosystem)**

```python
from ultralytics import YOLO

# Run pose estimation alongside detection
pose_model = YOLO("yolo11n-pose.pt")

def measure_shorts_length(image, shorts_bbox):
    """
    Uses knee keypoint from pose model to determine
    if shorts hem exceeds 3 inches above kneecap.
    Returns: "VALID", "VIOLATION", or "UNCERTAIN"
    """
    pose_results = pose_model(image)[0]

    for person in pose_results.keypoints.xy:
        # Keypoint 13 = left knee, Keypoint 14 = right knee (COCO format)
        left_knee = person[13]
        right_knee = person[14]
        knee_y = min(left_knee[1], right_knee[1])  # Use higher knee in frame

        # Bottom of shorts bounding box = hem of shorts
        shorts_hem_y = shorts_bbox[3]  # y2 of bounding box

        # Pixel distance from hem to knee
        pixel_distance = knee_y - shorts_hem_y

        # Convert pixels to inches using reference (calibrated per camera)
        # As a relative check: if hem is well above knee, flag it
        if pixel_distance < 0:
            return "VIOLATION"  # Hem is above knee keypoint — definitely too short
        elif pixel_distance < THRESHOLD_PIXELS:
            return "VIOLATION"  # Within violation range
        else:
            return "VALID"

    return "UNCERTAIN"  # No pose detected — defer to officer
```

### Measurement Logic Summary

```
IF civilian_bottom_shorts detected:
    → Run pose estimation
    → Find knee keypoint (y-coordinate)
    → Measure pixel gap between shorts hem and knee
    → IF hem is above knee OR within violation threshold → flag 27.1.2.3
    → IF pose not detected → mark UNCERTAIN → defer to officer

IF civilian_bottom_skirt detected:
    → Run pose estimation
    → Find knee keypoint (y-coordinate)
    → Measure pixel gap between skirt hem and knee
    → IF hem is more than 2 inches above knee → flag 27.1.2.7
    → IF pose not detected → mark UNCERTAIN → defer to officer
```

### Limitation to Document in Thesis
- Pixel-to-inch conversion requires camera calibration per deployment location
- Accuracy depends on camera angle and distance from subject
- Partial occlusion (bags, crowds) may prevent accurate pose detection
- In UNCERTAIN cases, system defers to officer judgment — this is documented behavior, not a system failure

---

## 11. OUT OF SCOPE (Explicitly Documented)

| Item | Reason |
|---|---|
| Offensive prints on clothing (27.1.2.10) | Text/print detection not reliable with YOLO — requires OCR pipeline; explicitly documented as limitation |
| Slipper type/brand distinction | Not necessary — ALL slippers are prohibited regardless of type |
| Face/identity recognition | Out of scope — SWAFO uses student ID barcode for identity; facial recognition raises separate ethics concerns |
| Real-time video stream deployment | Image-based inference for proposal phase; video stream is a planned future enhancement |

---

## 11. UN SDG ALIGNMENT

| SDG | Relevance |
|---|---|
| SDG 4 — Quality Education | Promotes discipline and professionalism through fair enforcement |
| SDG 9 — Industry, Innovation, Infrastructure | AI-based detection in campus management |
| SDG 10 — Reduced Inequality | Consistent, unbiased enforcement across all students |
| SDG 16 — Peace, Justice, Strong Institutions | Transparent, policy-aligned enforcement |
| SDG 17 — Partnerships for the Goals | Collaboration with SWAFO, OSS, SDAO |
