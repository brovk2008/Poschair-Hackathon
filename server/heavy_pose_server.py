"""
PosChair Heavy Pose Server
───────────────────────────
High-accuracy pose estimation backend using YOLOv8-Pose.
Exposes a FastAPI REST endpoint that can be tunneled via Cloudflare Tunnel
to provide desktop-grade deep pose accuracy to PosChair on local or Vercel deployments.
"""

import io
import time
import base64
import math
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="PosChair Heavy Pose Engine", version="1.0.0")

# Enable CORS for local dev and Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (YOLOv8 Medium Pose for high accuracy)
MODEL_NAME = "yolov8m-pose.pt"
print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
print(f"Model {MODEL_NAME} ready on {model.device}!")

class PredictRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string

def angle_between_3_points(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
    v1 = (a["x"] - b["x"], a["y"] - b["y"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 * mag2 < 1e-9:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PosChair Heavy Pose Engine",
        "model": MODEL_NAME,
        "device": str(model.device),
        "timestamp": time.time(),
    }

@app.post("/predict")
def predict_pose(req: PredictRequest):
    t0 = time.perf_counter()
    image_data = req.image

    # Handle data URL prefix
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image base64 data: {str(e)}")

    w, h = img.size
    results = model(img, verbose=False)

    if not results or len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return {
            "person_detected": False,
            "inference_time_ms": round((time.perf_counter() - t0) * 1000, 1),
            "keypoints": [],
            "landmarks_33": [],
            "metrics": None,
        }

    # Extract primary detection (largest / highest box score)
    res = results[0]
    kpts_tensor = res.keypoints.data[0].cpu().numpy()  # shape: (17, 3) -> [x, y, conf]

    keypoints_coco = []
    for i in range(17):
        x_norm = float(kpts_tensor[i][0] / w)
        y_norm = float(kpts_tensor[i][1] / h)
        conf = float(kpts_tensor[i][2])
        keypoints_coco.append({
            "id": i,
            "x": round(x_norm, 4),
            "y": round(y_norm, 4),
            "confidence": round(conf, 3),
        })

    # Map COCO 17 to MediaPipe 33 layout
    # COCO: 0:nose, 3:l_ear, 4:r_ear, 5:l_shoulder, 6:r_shoulder, 7:l_elbow, 8:r_elbow, 11:l_hip, 12:r_hip
    landmarks_33 = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.0} for _ in range(33)]

    mapping = {
        0: 0,   # nose -> nose
        3: 7,   # left ear -> left ear
        4: 8,   # right ear -> right ear
        5: 11,  # left shoulder -> left shoulder
        6: 12,  # right shoulder -> right shoulder
        7: 13,  # left elbow -> left elbow
        8: 14,  # right elbow -> right elbow
        11: 23, # left hip -> left hip
        12: 24, # right hip -> right hip
    }

    for coco_idx, mp_idx in mapping.items():
        landmarks_33[mp_idx] = {
            "x": keypoints_coco[coco_idx]["x"],
            "y": keypoints_coco[coco_idx]["y"],
            "z": 0.0,
            "visibility": keypoints_coco[coco_idx]["confidence"],
        }

    # Compute key biomechanical metrics
    nose = landmarks_33[0]
    l_ear = landmarks_33[7]
    r_ear = landmarks_33[8]
    l_sh = landmarks_33[11]
    r_sh = landmarks_33[12]
    l_hip = landmarks_33[23]
    r_hip = landmarks_33[24]

    mid_ear = {"x": (l_ear["x"] + r_ear["x"]) / 2, "y": (l_ear["y"] + r_ear["y"]) / 2}
    mid_sh = {"x": (l_sh["x"] + r_sh["x"]) / 2, "y": (l_sh["y"] + r_sh["y"]) / 2}
    mid_hip = {"x": (l_hip["x"] + r_hip["x"]) / 2, "y": (l_hip["y"] + r_hip["y"]) / 2}

    # Head angle: angle ear -> shoulder -> hip
    head_neck_angle = angle_between_3_points(mid_ear, mid_sh, mid_hip)

    # Lateral shoulder tilt in degrees
    dx_sh = r_sh["x"] - l_sh["x"]
    dy_sh = r_sh["y"] - l_sh["y"]
    sh_tilt_deg = math.degrees(math.atan2(dy_sh, dx_sh))

    # Shoulder asymmetry
    sh_dist = math.hypot(dx_sh, dy_sh)
    sh_asymmetry = abs(dy_sh) / max(sh_dist, 1e-4)

    # Trunk lean
    dx_trunk = mid_sh["x"] - mid_hip["x"]
    dy_trunk = mid_hip["y"] - mid_sh["y"]
    trunk_lean = dx_trunk / max(dy_trunk, 1e-4)

    # Overall keypoint confidence
    mean_conf = float(np.mean([k["confidence"] for k in keypoints_coco[:7]]))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "person_detected": True,
        "inference_time_ms": elapsed_ms,
        "model": MODEL_NAME,
        "keypoints": keypoints_coco,
        "landmarks_33": landmarks_33,
        "metrics": {
            "headNeckShoulderAngle": round(head_neck_angle, 2),
            "lateralTiltDeg": round(sh_tilt_deg, 2),
            "shoulderAsymmetry": round(sh_asymmetry, 4),
            "trunkLean": round(trunk_lean, 4),
            "confidence": round(mean_conf, 3),
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting PosChair Heavy Pose Server on http://0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
