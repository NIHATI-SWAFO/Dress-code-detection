"""
Enhanced UWear — Dress Code Detection & Compliance Module
==========================================================
Standalone inference module aligned with the DLSU-D School Uniform
and Dress Code Policy (effective September 1, 2025).

Supports two compliance modes:
  • UNIFORM_MODE  — Mon/Tue/Thu/Fri for 1st–3rd year students
  • CIVILIAN_MODE — Wed/Sat (all) + Mon–Sat for 4th year students

Usage:
  .venv\\Scripts\\python.exe inference/detect.py --image path/to/image.jpg
  .venv\\Scripts\\python.exe inference/detect.py --image path/to/image.jpg --mode CIVILIAN_MODE
  .venv\\Scripts\\python.exe inference/detect.py --folder path/to/images/
  .venv\\Scripts\\python.exe inference/detect.py --image path/to/image.jpg --year 4

References:
  DLSU-D Student Handbook, Section 27.1.2 — Minor Offense (Dress Code)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("Activate the venv: .venv\\Scripts\\activate")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DEFAULT_MODEL = BASE_DIR / "models" / "yolo11s_dev_best.pt"
OUTPUT_DIR = BASE_DIR / "results" / "detections"

CONFIDENCE_THRESHOLD = 0.35

# ── Class definitions ────────────────────────────────────────
CLASS_NAMES = {
    0: "uniform_top",
    1: "uniform_bottom",
    2: "civilian_top_short_sleeve",
    3: "civilian_top_long_sleeve",
    4: "civilian_bottom_trousers",
    5: "civilian_bottom_shorts",
    6: "civilian_bottom_skirt",
    7: "footwear_shoes",
    8: "footwear_slippers",
    9: "prohibited_ripped_pants",
    10: "prohibited_leggings",
    11: "prohibited_sleeveless",
    12: "prohibited_crop_halter",
    13: "prohibited_midriff_offshoulder",
}

# ── Class groupings for compliance logic ─────────────────────
UNIFORM_TOPS    = {0}                   # uniform_top
UNIFORM_BOTTOMS = {1}                   # uniform_bottom
CIVILIAN_TOPS   = {2, 3}               # short_sleeve, long_sleeve
CIVILIAN_BOTTOMS = {4, 5, 6}           # trousers, shorts, skirt
FOOTWEAR_SHOES  = {7}                   # shoes
FOOTWEAR_SLIPPERS = {8}                # slippers
PROHIBITED = {9, 10, 11, 12, 13}       # all prohibited items

ALL_TOPS = UNIFORM_TOPS | CIVILIAN_TOPS | {11, 12, 13}  # includes prohibited tops
ALL_BOTTOMS = UNIFORM_BOTTOMS | CIVILIAN_BOTTOMS | {9, 10}  # includes prohibited bottoms

# ── Handbook violation references (Section 27.1.2) ───────────
VIOLATION_REFS = {
    "footwear_slippers":            {"code": "27.1.2.11", "desc": "Slippers are not allowed within the campus premises"},
    "prohibited_ripped_pants":      {"code": "27.1.2.2",  "desc": "Ripped jeans exposing skin are not permitted"},
    "prohibited_leggings":          {"code": "27.1.2.8",  "desc": "Leggings, jeggings, or overly tight clothing not allowed"},
    "prohibited_sleeveless":        {"code": "27.1.2.4",  "desc": "Sleeveless blouses, sling tops, or spaghetti straps prohibited"},
    "prohibited_crop_halter":       {"code": "27.1.2.9",  "desc": "Halter blouses and crop tops are not allowed"},
    "prohibited_midriff_offshoulder": {"code": "27.1.2.6", "desc": "Midriff tops, hanging blouses, and off-shoulder tops prohibited"},
    "not_in_uniform_top":           {"code": "27.1.2",    "desc": "Not wearing prescribed school uniform top on uniform day"},
    "not_in_uniform_bottom":        {"code": "27.1.2",    "desc": "Not wearing prescribed school uniform bottom on uniform day"},
    "civilian_on_uniform_day":      {"code": "27.1.2",    "desc": "Wearing civilian attire on designated uniform day"},
}

# ── Color palette for bounding boxes (BGR format) ────────────
CLASS_COLORS = {
    "uniform_top":                  (0, 180, 0),       # green
    "uniform_bottom":               (0, 180, 0),       # green
    "civilian_top_short_sleeve":    (200, 160, 0),     # teal
    "civilian_top_long_sleeve":     (200, 160, 0),     # teal
    "civilian_bottom_trousers":     (200, 160, 0),     # teal
    "civilian_bottom_shorts":       (200, 160, 0),     # teal
    "civilian_bottom_skirt":        (200, 160, 0),     # teal
    "footwear_shoes":               (180, 130, 0),     # dark teal
    "footwear_slippers":            (0, 0, 220),       # red
    "prohibited_ripped_pants":      (0, 0, 220),       # red
    "prohibited_leggings":          (0, 0, 220),       # red
    "prohibited_sleeveless":        (0, 0, 220),       # red
    "prohibited_crop_halter":       (0, 0, 220),       # red
    "prohibited_midriff_offshoulder": (0, 0, 220),     # red
}


# ============================================================
# COMPLIANCE ENGINE — Based on DLSU-D Handbook
# ============================================================
def assess_compliance(detected_classes: list[dict], mode: str, year_level: int) -> dict:
    """
    Evaluate dress code compliance based on detected garments.

    Modes:
      UNIFORM_MODE  — 1st–3rd year on Mon/Tue/Thu/Fri
      CIVILIAN_MODE — Wed/Sat (all) + 4th year (all weekdays)

    Args:
        detected_classes: list of {"class_id": int, "class_name": str, "confidence": float, "bbox": list}
        mode: "UNIFORM_MODE" or "CIVILIAN_MODE"
        year_level: 1, 2, 3, or 4

    Returns:
        dict with compliance_status, violations, detected_garments, mode_used
    """
    violations = []
    detected_ids = set()
    detected_names = set()

    for det in detected_classes:
        detected_ids.add(det["class_id"])
        detected_names.add(det["class_name"])

    # ── Check for prohibited items (ALWAYS a violation, any mode) ────
    for det in detected_classes:
        cid = det["class_id"]
        cname = det["class_name"]
        if cid in PROHIBITED:
            ref = VIOLATION_REFS.get(cname, {"code": "27.1.2", "desc": f"Prohibited item: {cname}"})
            violations.append({
                "class": cname,
                "handbook_ref": ref["code"],
                "description": ref["desc"],
                "confidence": det["confidence"],
                "severity": "MINOR_OFFENSE",
            })

    # ── Check for slippers (ALWAYS prohibited — Sec. 27.1.2.11) ─────
    if detected_ids & FOOTWEAR_SLIPPERS:
        ref = VIOLATION_REFS["footwear_slippers"]
        # Avoid duplicate if already added from PROHIBITED check
        if not any(v["class"] == "footwear_slippers" for v in violations):
            slipper_det = next((d for d in detected_classes if d["class_id"] == 8), None)
            violations.append({
                "class": "footwear_slippers",
                "handbook_ref": ref["code"],
                "description": ref["desc"],
                "confidence": slipper_det["confidence"] if slipper_det else 0.0,
                "severity": "MINOR_OFFENSE",
            })

    # ── Mode-specific checks ─────────────────────────────────────────
    if mode == "UNIFORM_MODE":
        # 1st–3rd year on uniform days MUST wear uniform

        has_uniform_top = bool(detected_ids & UNIFORM_TOPS)
        has_uniform_bottom = bool(detected_ids & UNIFORM_BOTTOMS)
        has_civilian_top = bool(detected_ids & CIVILIAN_TOPS)
        has_civilian_bottom = bool(detected_ids & CIVILIAN_BOTTOMS)

        if not has_uniform_top:
            ref = VIOLATION_REFS["not_in_uniform_top"]
            violations.append({
                "class": "missing_uniform_top",
                "handbook_ref": ref["code"],
                "description": ref["desc"],
                "confidence": None,
                "severity": "MINOR_OFFENSE",
            })

        if not has_uniform_bottom:
            ref = VIOLATION_REFS["not_in_uniform_bottom"]
            violations.append({
                "class": "missing_uniform_bottom",
                "handbook_ref": ref["code"],
                "description": ref["desc"],
                "confidence": None,
                "severity": "MINOR_OFFENSE",
            })

        # If civilian clothing is detected on a uniform day, that's a flag
        if has_civilian_top and not has_uniform_top:
            ref = VIOLATION_REFS["civilian_on_uniform_day"]
            violations.append({
                "class": "civilian_top_on_uniform_day",
                "handbook_ref": ref["code"],
                "description": ref["desc"],
                "confidence": None,
                "severity": "MINOR_OFFENSE",
            })

        if has_civilian_bottom and not has_uniform_bottom:
            ref = VIOLATION_REFS["civilian_on_uniform_day"]
            # Avoid duplicate violation description
            if not any(v["class"] == "civilian_top_on_uniform_day" for v in violations):
                violations.append({
                    "class": "civilian_bottom_on_uniform_day",
                    "handbook_ref": ref["code"],
                    "description": "Wearing civilian bottom on designated uniform day",
                    "confidence": None,
                    "severity": "MINOR_OFFENSE",
                })

    elif mode == "CIVILIAN_MODE":
        # Wash days or 4th year — civilian clothing allowed
        # Wearing uniform on civilian day is STILL COMPLIANT (opt-in)
        # Only prohibited items and slippers cause violations
        # (Already handled above)
        pass

    # ── Determine overall status ─────────────────────────────────────
    compliance_status = "COMPLIANT" if len(violations) == 0 else "NON_COMPLIANT"

    # ── Categorize what was detected ─────────────────────────────────
    garment_summary = {
        "tops": [],
        "bottoms": [],
        "footwear": [],
        "prohibited": [],
    }

    for det in detected_classes:
        cid = det["class_id"]
        entry = {"class": det["class_name"], "confidence": det["confidence"]}
        if cid in UNIFORM_TOPS | CIVILIAN_TOPS:
            garment_summary["tops"].append(entry)
        elif cid in UNIFORM_BOTTOMS | CIVILIAN_BOTTOMS:
            garment_summary["bottoms"].append(entry)
        elif cid in FOOTWEAR_SHOES | FOOTWEAR_SLIPPERS:
            garment_summary["footwear"].append(entry)
        if cid in PROHIBITED:
            garment_summary["prohibited"].append(entry)

    return {
        "compliance_status": compliance_status,
        "detection_mode": mode,
        "year_level": year_level,
        "violations": violations,
        "violation_count": len(violations),
        "detected_garments": garment_summary,
        "detected_classes": [d["class_name"] for d in detected_classes],
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# VISUALIZATION
# ============================================================
def draw_detections(image, detections: list[dict], compliance: dict) -> np.ndarray:
    """Draw bounding boxes, labels, and compliance banner on image."""
    img = image.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cls_name, (128, 128, 128))

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Compliance banner at top ─────────────────────────────────────
    status = compliance["compliance_status"]
    mode = compliance["detection_mode"]
    n_violations = compliance["violation_count"]

    if status == "COMPLIANT":
        banner_color = (0, 160, 0)
        banner_text = f"COMPLIANT | {mode} | No Violations"
    else:
        banner_color = (0, 0, 200)
        banner_text = f"NON-COMPLIANT | {mode} | {n_violations} Violation(s)"

    cv2.rectangle(img, (0, 0), (w, 40), banner_color, -1)
    cv2.putText(img, banner_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Violation details at bottom ──────────────────────────────────
    if compliance["violations"]:
        y_offset = h - 10 - (len(compliance["violations"]) * 22)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, y_offset - 10), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        for i, v in enumerate(compliance["violations"]):
            text = f"[{v['handbook_ref']}] {v['description']}"
            y = y_offset + (i * 22)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

    return img


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================
def run_detection(
    image_path: str,
    model: YOLO,
    mode: str = "CIVILIAN_MODE",
    year_level: int = 1,
    save: bool = True,
) -> dict:
    """
    Full pipeline: detect → assess compliance → annotate → save.

    Args:
        image_path: path to input image
        model: loaded YOLO model
        mode: "UNIFORM_MODE" or "CIVILIAN_MODE"
        year_level: 1–4
        save: whether to save annotated image + JSON

    Returns:
        compliance result dict
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found — {image_path}")
        return {}

    # ── Run YOLO inference ───────────────────────────────────────────
    results = model(str(image_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    # ── Parse detections ─────────────────────────────────────────────
    detections = []
    for box in result.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        detections.append({
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, f"unknown_{cid}"),
            "confidence": round(conf, 4),
            "bbox": [round(v, 1) for v in bbox],
        })

    # ── Run compliance assessment ────────────────────────────────────
    compliance = assess_compliance(detections, mode, year_level)
    compliance["image"] = str(image_path)
    compliance["total_detections"] = len(detections)

    # ── Print results to console ─────────────────────────────────────
    status_icon = "✅" if compliance["compliance_status"] == "COMPLIANT" else "❌"
    print(f"\n{status_icon} {compliance['compliance_status']} | {mode} | Year {year_level}")
    print(f"   Image: {image_path.name}")
    print(f"   Detections: {len(detections)}")

    if detections:
        print(f"   Classes: {', '.join(d['class_name'] for d in detections)}")

    if compliance["violations"]:
        print(f"   Violations ({len(compliance['violations'])}):")
        for v in compliance["violations"]:
            print(f"     ⚠ [{v['handbook_ref']}] {v['description']}")

    # ── Save annotated image + JSON ──────────────────────────────────
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Annotated image
        img = cv2.imread(str(image_path))
        if img is not None:
            annotated = draw_detections(img, detections, compliance)
            out_img = OUTPUT_DIR / f"{stem}_{ts}_annotated.jpg"
            cv2.imwrite(str(out_img), annotated)
            print(f"   Saved: {out_img}")

        # JSON report
        out_json = OUTPUT_DIR / f"{stem}_{ts}_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(compliance, f, indent=2, ensure_ascii=False)
        print(f"   Report: {out_json}")

    return compliance


# ============================================================
# AUTO-DETECT MODE FROM DATE
# ============================================================
def auto_detect_mode(year_level: int) -> str:
    """
    Determine UNIFORM_MODE or CIVILIAN_MODE based on current
    day of the week and student year level.

    Schedule (September 1, 2025 Policy):
      1st–3rd Year: Mon/Tue/Thu/Fri = Uniform, Wed/Sat = Wash Day
      4th Year:     All days = Civilian (may opt to wear uniform)
    """
    if year_level >= 4:
        return "CIVILIAN_MODE"

    day = datetime.now().weekday()  # 0=Mon, 6=Sun
    # Mon=0, Tue=1 → Uniform | Wed=2 → Wash | Thu=3, Fri=4 → Uniform | Sat=5 → Wash | Sun=6 → N/A
    uniform_days = {0, 1, 3, 4}

    if day in uniform_days:
        return "UNIFORM_MODE"
    else:
        return "CIVILIAN_MODE"


# ============================================================
# CLI ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced UWear — Dress Code Detection & Compliance Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference/detect.py --image test_photos/student1.jpg
  python inference/detect.py --image test_photos/student1.jpg --mode UNIFORM_MODE --year 2
  python inference/detect.py --folder test_photos/
  python inference/detect.py --image test_photos/student1.jpg --mode auto --year 3
        """
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to YOLO weights (default: best.pt)")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["UNIFORM_MODE", "CIVILIAN_MODE", "auto"],
                        help="Compliance mode (default: auto-detect from date + year)")
    parser.add_argument("--year", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Student year level (default: 1)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated images")

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    # ── Determine mode ───────────────────────────────────────────────
    if args.mode == "auto":
        mode = auto_detect_mode(args.year)
        print(f"Auto-detected mode: {mode} (Year {args.year}, {datetime.now().strftime('%A')})")
    else:
        mode = args.mode

    # ── Load model ───────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Update threshold for this run
    conf_threshold = args.conf

    # ── Run on single image or folder ────────────────────────────────
    all_results = []

    if args.image:
        result = run_detection(args.image, model, mode, args.year, save=not args.no_save)
        all_results.append(result)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"ERROR: Folder not found — {folder}")
            sys.exit(1)

        image_files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        if not image_files:
            print(f"No images found in {folder}")
            sys.exit(1)

        print(f"Processing {len(image_files)} images from {folder}...")
        for img_path in image_files:
            result = run_detection(str(img_path), model, mode, args.year, save=not args.no_save)
            all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────
    if len(all_results) > 1:
        compliant = sum(1 for r in all_results if r.get("compliance_status") == "COMPLIANT")
        total = len(all_results)
        print(f"\n{'='*50}")
        print(f"BATCH SUMMARY: {compliant}/{total} compliant ({compliant/total*100:.1f}%)")
        print(f"{'='*50}")

        # Save batch summary
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_path = OUTPUT_DIR / f"batch_summary_{ts}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump({
                "mode": mode,
                "year_level": args.year,
                "total_images": total,
                "compliant": compliant,
                "non_compliant": total - compliant,
                "compliance_rate": round(compliant / total * 100, 2),
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Batch report: {batch_path}")


if __name__ == "__main__":
    main()
