import os
import sys
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics or opencv not installed.")
    sys.exit(1)

# Import the compliance logic we built
sys.path.append(str(Path(__file__).parent))
from detect import assess_compliance, draw_detections, CLASS_NAMES

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "results" / "enhanced-uwear-dev" / "weights" / "best.pt"
CONFIDENCE_THRESHOLD = 0.35

# Default settings (you can change these in the UI by pressing keys)
CURRENT_MODE = "CIVILIAN_MODE"
YEAR_LEVEL = 1

def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default laptop webcam
    
    # Increase resolution if camera supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    global CURRENT_MODE, YEAR_LEVEL
    
    print("\n" + "="*50)
    print(" LIVE DRESS CODE TESTER STARTED ")
    print("="*50)
    print(" Controls:")
    print("   [U] - Switch to UNIFORM_MODE")
    print("   [C] - Switch to CIVILIAN_MODE")
    print("   [1-4] - Change Student Year Level")
    print("   [Q] - Quit")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # Parse detections
        detections = []
        for box in result.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            detections.append({
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, f"unknown_{cid}"),
                "confidence": conf,
                "bbox": bbox,
            })

        # Run compliance assessment
        compliance = assess_compliance(detections, CURRENT_MODE, YEAR_LEVEL)
        
        # Add year info to compliance for the banner
        compliance["detection_mode"] = f"{CURRENT_MODE} (Yr {YEAR_LEVEL})"

        # Draw boxes and overlay
        annotated_frame = draw_detections(frame, detections, compliance)

        # Show the frame
        cv2.imshow("Enhanced UWear - Live Tester", annotated_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            CURRENT_MODE = "UNIFORM_MODE"
            print(f"Switched to {CURRENT_MODE}")
        elif key == ord('c'):
            CURRENT_MODE = "CIVILIAN_MODE"
            print(f"Switched to {CURRENT_MODE}")
        elif key == ord('1'):
            YEAR_LEVEL = 1
            print(f"Year Level set to 1")
        elif key == ord('4'):
            YEAR_LEVEL = 4
            CURRENT_MODE = "CIVILIAN_MODE"  # Auto switch for 4th years
            print(f"Year Level set to 4 (Auto-switched to CIVILIAN_MODE)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
