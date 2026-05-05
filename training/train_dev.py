"""
Development Training Script - yolo11s
======================================
Quick iteration model to validate dataset quality and identify weak classes.
Do NOT run this until you're ready to train (~10-12 hrs on GTX 1660 SUPER).

Usage (ALWAYS use the project venv):
    .venv\\Scripts\\python.exe training/train_dev.py
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

    # 1. data.yaml exists
    if not os.path.exists(DATA_YAML):
        errors.append(f"data.yaml not found at {DATA_YAML}")

    # 2. Training images exist
    train_imgs = os.path.join(BASE_DIR, "dataset", "balanced", "images", "train")
    if not os.path.isdir(train_imgs):
        errors.append(f"Training images directory not found: {train_imgs}")
    else:
        n_imgs = len([f for f in os.listdir(train_imgs) if os.path.isfile(os.path.join(train_imgs, f))])
        if n_imgs == 0:
            errors.append(f"Training images directory is empty: {train_imgs}")
        else:
            print(f"  Training images: {n_imgs}")

    # 3. Validation images exist
    val_imgs = os.path.join(BASE_DIR, "dataset", "balanced", "images", "val")
    if not os.path.isdir(val_imgs):
        errors.append(f"Validation images directory not found: {val_imgs}")
    else:
        n_imgs = len([f for f in os.listdir(val_imgs) if os.path.isfile(os.path.join(val_imgs, f))])
        print(f"  Validation images: {n_imgs}")

    # 4. CUDA available
    if not torch.cuda.is_available():
        errors.append("CUDA not available! Training will be extremely slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"  GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    # 5. Output directories
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    if errors:
        print("\n❌ PRE-FLIGHT FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    print("  ✅ All pre-flight checks passed\n")


# ============================================================
# TRAINING
# ============================================================
def main():
    print("=" * 60)
    print("ENHANCED UWEAR - Development Training (yolo11s)")
    print("=" * 60)
    print(f"Dataset:  {DATA_YAML}")
    print(f"Output:   {PROJECT_DIR}")
    print()

    print("Running pre-flight checks...")
    preflight()

    model = YOLO("yolo11s.pt")  # Downloads automatically if not cached

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,                # GTX 1660 SUPER (6GB VRAM)
        name="enhanced-uwear-dev",
        project=PROJECT_DIR,
        patience=20,            # Early stopping: no improvement for 20 epochs
        save=True,
        save_period=10,         # Checkpoint every 10 epochs
        val=True,
        plots=True,
        device=0,               # GPU
        workers=4,
        exist_ok=True,          # Overwrite previous run with same name

        # Classification loss weight - penalizes misclassification harder
        # Helps the model pay attention to minority classes
        cls=2.0,

        # Online augmentation (built into YOLO11)
        hsv_h=0.015,            # Hue shift
        hsv_s=0.7,              # Saturation shift
        hsv_v=0.4,              # Value/brightness shift
        degrees=15.0,           # Rotation range
        translate=0.1,          # Translation range
        scale=0.5,              # Scale range
        flipud=0.0,             # No vertical flip (people aren't upside down)
        fliplr=0.5,             # Horizontal flip (left/right doesn't matter)
        mosaic=1.0,             # Mosaic augmentation (combines 4 images)
        mixup=0.1,              # MixUp augmentation
        copy_paste=0.1,         # Copy-paste augmentation
    )

    # Save best weights to models/ directory
    best_weights = os.path.join(PROJECT_DIR, "enhanced-uwear-dev", "weights", "best.pt")
    if os.path.exists(best_weights):
        dst = os.path.join(MODEL_SAVE_DIR, "yolo11s_dev_best.pt")
        shutil.copy2(best_weights, dst)
        print(f"\n✅ Best weights saved to: {dst}")
    else:
        print("\n⚠️  best.pt not found — training may have been interrupted.")
        # Try to find any weights
        weights_dir = os.path.join(PROJECT_DIR, "enhanced-uwear-dev", "weights")
        if os.path.isdir(weights_dir):
            print(f"   Available weights: {os.listdir(weights_dir)}")

    # Also save last weights as backup
    last_weights = os.path.join(PROJECT_DIR, "enhanced-uwear-dev", "weights", "last.pt")
    if os.path.exists(last_weights):
        dst_last = os.path.join(MODEL_SAVE_DIR, "yolo11s_dev_last.pt")
        shutil.copy2(last_weights, dst_last)
        print(f"   Last weights saved to: {dst_last}")

    print("\n" + "=" * 60)
    print("DEV TRAINING COMPLETE")
    print("Next steps:")
    print("  1. Check results/enhanced-uwear-dev/ for training curves")
    print("  2. Run: .venv\\Scripts\\python.exe training/evaluate.py --model models/yolo11s_dev_best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
