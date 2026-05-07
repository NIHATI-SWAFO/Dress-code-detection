import os
import sys
import cv2
from pathlib import Path
from flask import Flask, render_template_string, Response, request

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.")
    sys.exit(1)

# Import our compliance logic
sys.path.append(str(Path(__file__).parent))
from detect import assess_compliance, draw_detections, CLASS_NAMES

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo11s_dev_best.pt"
CONFIDENCE_THRESHOLD = 0.35

# Global state for the web controls
CURRENT_MODE = "CIVILIAN_MODE"
YEAR_LEVEL = 1

# Initialize Flask app
app = Flask(__name__)

# Load Model
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)
model = YOLO(str(MODEL_PATH))

# Global camera object
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def generate_frames():
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Run YOLO inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # 2. Extract detections
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

        # 3. Assess Compliance
        compliance = assess_compliance(detections, CURRENT_MODE, YEAR_LEVEL)
        compliance["detection_mode"] = f"{CURRENT_MODE} (Yr {YEAR_LEVEL})"

        # 4. Draw Overlay
        annotated_frame = draw_detections(frame, detections, compliance)

        # 5. Convert to JPEG for web streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ============================================================
# WEB ROUTES
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced UWear - Live Web UI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; margin: 0; padding: 0; }
        .header { background-color: #0b4528; color: white; padding: 20px; text-align: center; }
        .container { max-width: 1200px; margin: 20px auto; padding: 20px; text-align: center; }
        .video-container { border: 4px solid #333; border-radius: 8px; overflow: hidden; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
        img { display: block; max-width: 100%; height: auto; }
        .controls { margin-top: 20px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-block; }
        .control-group { margin: 10px 0; font-size: 18px; }
        select, button { padding: 10px 15px; font-size: 16px; margin-left: 10px; border-radius: 4px; border: 1px solid #ccc; }
        button { background-color: #0b4528; color: white; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #0f5e36; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced UWear - Live Dress Code Monitor</h1>
    </div>
    
    <div class="container">
        <div class="video-container">
            <!-- This image tag reads the live MJPEG stream -->
            <img src="/video_feed" width="1280" height="720" alt="Live Camera Feed">
        </div>

        <div class="controls">
            <form action="/update_settings" method="post" target="hidden-iframe">
                <div class="control-group">
                    <label for="mode"><strong>Detection Mode:</strong></label>
                    <select name="mode" id="mode" onchange="this.form.submit()">
                        <option value="UNIFORM_MODE" {% if current_mode == 'UNIFORM_MODE' %}selected{% endif %}>UNIFORM_MODE (Mon/Tue/Thu/Fri)</option>
                        <option value="CIVILIAN_MODE" {% if current_mode == 'CIVILIAN_MODE' %}selected{% endif %}>CIVILIAN_MODE (Wash Days)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="year"><strong>Student Year Level:</strong></label>
                    <select name="year" id="year" onchange="this.form.submit()">
                        <option value="1" {% if year_level == 1 %}selected{% endif %}>1st - 3rd Year</option>
                        <option value="4" {% if year_level == 4 %}selected{% endif %}>4th Year (Always Civilian)</option>
                    </select>
                </div>
            </form>
            <p style="color: #666; font-size: 14px; margin-top: 15px;">Change settings above to instantly update the live camera rules.</p>
        </div>
    </div>
    
    <!-- Hidden iframe to prevent page reload on form submit -->
    <iframe name="hidden-iframe" style="display:none;"></iframe>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        current_mode=CURRENT_MODE, 
        year_level=YEAR_LEVEL
    )

@app.route('/video_feed')
def video_feed():
    # Returns the streaming response using our generator
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global CURRENT_MODE, YEAR_LEVEL
    CURRENT_MODE = request.form.get('mode', CURRENT_MODE)
    YEAR_LEVEL = int(request.form.get('year', YEAR_LEVEL))
    
    # Auto-switch to civilian mode if 4th year is selected
    if YEAR_LEVEL == 4:
        CURRENT_MODE = "CIVILIAN_MODE"
        
    print(f"[*] Settings updated: {CURRENT_MODE}, Year {YEAR_LEVEL}")
    return "OK", 200

if __name__ == "__main__":
    print("="*60)
    print(" STARTING WEB CAMERA SERVER ")
    print("="*60)
    print("1. Open your web browser")
    print("2. Go to: http://127.0.0.1:5000")
    print("="*60)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
