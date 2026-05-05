"""
Final Training Script - yolo11m
================================
Thesis-grade model. Run AFTER reviewing dev training results.
Estimated time: ~20-25 hrs on GTX 1660 SUPER.

Usage (ALWAYS use the project venv):
    .venv\\Scripts\\python.exe training/train_final.py
"""

import os
import sys
import shutil

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("ERROR: ultralytics or torch not installed.")
    print("Activate the venv first: .venv\\Scripts\\activate")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "balanced", "data.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "results")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
def preflight():
    """Verify everything is ready before starting a multi-hour training run."""
    errors = []

    if not os.path.exists(DATA_YAML):
        errors.append(f"data.yaml not found at {DATA_YAML}")

    train_imgs = os.path.join(BASE_DIR, "dataset", "balanced", "images", "train")
    if not os.path.isdir(train_imgs):
        errors.append(f"Training images directory not found: {train_imgs}")
    else:
        n_imgs = len([f for f in os.listdir(train_imgs) if os.path.isfile(os.path.join(train_imgs, f))])
        if n_imgs == 0:
            errors.append(f"Training images directory is empty")
        else:
            print(f"  Training images: {n_imgs}")

    val_imgs = os.path.join(BASE_DIR, "dataset", "balanced", "images", "val")
    if os.path.isdir(val_imgs):
        n_imgs = len([f for f in os.listdir(val_imgs) if os.path.isfile(os.path.join(val_imgs, f))])
        print(f"  Validation images: {n_imgs}")

    if not torch.cuda.is_available():
        errors.append("CUDA not available!")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"  GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    # Check dev results exist (final should run AFTER dev)
    dev_best = os.path.join(MODEL_SAVE_DIR, "yolo11s_dev_best.pt")
    if not os.path.exists(dev_best):
        print("  WARNING: Dev model (yolo11s_dev_best.pt) not found.")
        print("           Recommended: Run train_dev.py first to validate the pipeline.")

    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    if errors:
        print("\n  PRE-FLIGHT FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    print("  All pre-flight checks passed\n")


# ============================================================
# TRAINING
# ============================================================
def main():
    print("=" * 60)
    print("ENHANCED UWEAR - Final Training (yolo11m)")
    print("=" * 60)
    print(f"Dataset:  {DATA_YAML}")
    print(f"Output:   {PROJECT_DIR}")
    print()

    print("Running pre-flight checks...")
    preflight()

    model = YOLO("yolo11m.pt")  # Downloads automatically if not cached

    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=4,                # 6GB VRAM - yolo11m needs smaller batch
        name="enhanced-uwear-final",
        project=PROJECT_DIR,
        patience=30,            # More patience for final run
        save=True,
        save_period=10,
        val=True,
        plots=True,
        device=0,
        workers=4,
        exist_ok=True,          # Overwrite previous run with same name

        cls=2.0,

        # Augmentation - slightly more aggressive for final run
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,             # Slightly higher than dev
        copy_paste=0.15,        # Slightly higher than dev
    )

    # Save best weights
    best_weights = os.path.join(PROJECT_DIR, "enhanced-uwear-final", "weights", "best.pt")
    if os.path.exists(best_weights):
        dst = os.path.join(MODEL_SAVE_DIR, "yolo11m_final_best.pt")
        shutil.copy2(best_weights, dst)
        print(f"\n  Best weights saved to: {dst}")
    else:
        print("\n  best.pt not found - training may have been interrupted.")
        weights_dir = os.path.join(PROJECT_DIR, "enhanced-uwear-final", "weights")
        if os.path.isdir(weights_dir):
            print(f"   Available weights: {os.listdir(weights_dir)}")

    # Also save last weights as backup
    last_weights = os.path.join(PROJECT_DIR, "enhanced-uwear-final", "weights", "last.pt")
    if os.path.exists(last_weights):
        dst_last = os.path.join(MODEL_SAVE_DIR, "yolo11m_final_last.pt")
        shutil.copy2(last_weights, dst_last)
        print(f"   Last weights saved to: {dst_last}")

    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print("Next steps:")
    print("  1. Check results/enhanced-uwear-final/ for training curves")
    print("  2. Run: .venv\\Scripts\\python.exe training/evaluate.py --model models/yolo11m_final_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
