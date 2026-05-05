"""
Evaluation Script - Generate Thesis Metrics Report
====================================================
Run AFTER training is complete to generate per-class AP,
confusion matrix, and comparison vs UWear baseline.

Usage (ALWAYS use the project venv):
    .venv\\Scripts\\python.exe training/evaluate.py --model results/enhanced-uwear-dev/weights/best.pt
"""

import os
import sys
import argparse
import json
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.")
    print("Activate the venv first: .venv\\Scripts\\activate")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "balanced", "data.yaml")
REPORTS_DIR = os.path.join(BASE_DIR, "results", "evaluation_reports")

CLASS_NAMES = [
    "uniform_top", "uniform_bottom",
    "civilian_top_short_sleeve", "civilian_top_long_sleeve",
    "civilian_bottom_trousers", "civilian_bottom_shorts", "civilian_bottom_skirt",
    "footwear_shoes", "footwear_slippers",
    "prohibited_ripped_pants", "prohibited_leggings", "prohibited_sleeveless",
    "prohibited_crop_halter", "prohibited_midriff_offshoulder"
]

# UWear baseline for comparison
UWEAR_BASELINE = {
    "mAP@0.5":    0.7883,
    "Precision":  0.7336,
    "Recall":     0.7881,
}


# ============================================================
# EVALUATION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate on")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    if not os.path.exists(DATA_YAML):
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"evaluation_{timestamp}.txt")
    json_path   = os.path.join(REPORTS_DIR, f"evaluation_{timestamp}.json")

    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 65)
    p("  ENHANCED UWEAR — MODEL EVALUATION REPORT")
    p(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  Model: {args.model}")
    p(f"  Split: {args.split}")
    p("=" * 65)

    model = YOLO(args.model)
    metrics = model.val(
        data=DATA_YAML,
        split=args.split,
        device=0,
        plots=True,
        save_json=False,
        project=REPORTS_DIR,
        name=f"eval_{timestamp}",
    )

    # ── Overall metrics ──────────────────────────────────────────
    p()
    p("=" * 65)
    p("OVERALL METRICS")
    p("=" * 65)

    results = {
        "mAP@0.5":    float(metrics.box.map50),
        "mAP@0.5:95": float(metrics.box.map),
        "Precision":  float(metrics.box.mp),
        "Recall":     float(metrics.box.mr),
    }

    p(f"{'Metric':<22} {'Your Model':>12} {'UWear':>10} {'Delta':>10}  Status")
    p("-" * 65)
    for metric, value in results.items():
        baseline = UWEAR_BASELINE.get(metric, None)
        if baseline is not None:
            delta = value - baseline
            status = "✓ PASS" if delta > 0 else "✗ FAIL"
            p(f"{metric:<22} {value:>12.4f} {baseline:>10.4f} {delta:>+10.4f}  {status}")
        else:
            p(f"{metric:<22} {value:>12.4f} {'N/A':>10}")

    # ── Per-class AP ─────────────────────────────────────────────
    p()
    p("=" * 65)
    p("PER-CLASS AVERAGE PRECISION (AP@0.5) — TEST SET")
    p("=" * 65)
    p(f"{'Class':<38} {'AP@0.5':>8}  Status")
    p("-" * 60)

    ap50_values = metrics.box.ap50
    per_class_data = {}

    # Minority classes we augmented — highlight these
    augmented = {"uniform_top", "uniform_bottom", "footwear_slippers", "prohibited_ripped_pants"}

    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(ap50_values):
            ap = float(ap50_values[i])
            per_class_data[cls_name] = ap
            if ap >= 0.85:
                status = "🔥 EXCELLENT"
            elif ap >= 0.70:
                status = "✓ GOOD"
            elif ap >= 0.50:
                status = "~ OK"
            else:
                status = "✗ WEAK"
            flag = " ★AUG" if cls_name in augmented else ""
            p(f"{cls_name:<38} {ap:>8.4f}  {status}{flag}")
        else:
            p(f"{cls_name:<38} {'N/A':>8}  ???")
            per_class_data[cls_name] = None

    # ── Augmented class summary ───────────────────────────────────
    p()
    p("=" * 65)
    p("AUGMENTED CLASS PERFORMANCE (★ = was minority, got augmented)")
    p("=" * 65)
    for cls in augmented:
        ap = per_class_data.get(cls)
        if ap is not None:
            p(f"  {cls:<38} AP = {ap:.4f}")

    # ── Final verdict ─────────────────────────────────────────────
    p()
    p("=" * 65)
    overall_map = results["mAP@0.5"]
    uwear_map = UWEAR_BASELINE["mAP@0.5"]
    delta = overall_map - uwear_map

    if overall_map > uwear_map:
        p(f"  RESULT: mAP {overall_map:.4f} > UWear {uwear_map:.4f}  (+{delta:.4f})")
        p(f"  ✓ UWear baseline BEATEN by +{delta*100:.2f} percentage points")
    else:
        p(f"  RESULT: mAP {overall_map:.4f} < UWear {uwear_map:.4f}  ({delta:+.4f})")
        p("  ✗ Below UWear baseline")

    p("=" * 65)
    p()
    p(f"  Confusion matrix + PR curves saved to: {REPORTS_DIR}")
    p(f"  Full report saved to: {report_path}")

    # ── Save report ───────────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_data = {
        "timestamp": timestamp,
        "model": args.model,
        "split": args.split,
        "overall": results,
        "per_class_ap50": per_class_data,
        "uwear_baseline": UWEAR_BASELINE,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nJSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
