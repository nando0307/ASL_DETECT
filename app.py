"""
FastAPI web application for real-time ASL (American Sign Language) detection.

Uses MediaPipe hand landmarks + a trained MLP classifier to recognize
ASL alphabet letters (A-Z) from webcam frames sent by the browser.

Run:  uvicorn app:app --reload
Open: http://localhost:8000
"""

import io
import base64

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ──
MODEL_PATH = Path("asl_landmark_mlp.pth")
LANDMARKER_MODEL = Path("hand_landmarker.task")
NUM_FEATURES = 63  # 21 landmarks × 3

# ── MLP Architecture (must match training) ──
class LandmarkMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Load model & landmarker at startup ──
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]
model = LandmarkMLP(NUM_FEATURES, len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

base_opts = mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER_MODEL))
opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
landmarker = mp_vision.HandLandmarker.create_from_options(opts)

# ── Hand connections for drawing ──
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def normalise(landmarks) -> np.ndarray:
    """Normalize 21 hand landmarks: center on wrist, scale to unit."""
    row = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    row -= row[0]
    scale = np.linalg.norm(row, axis=1).max()
    if scale > 0:
        row /= scale
    return row.flatten()


# ── FastAPI app ──
app = FastAPI(title="ASL Detection", description="Real-time ASL alphabet detection")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main webcam page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request):
    """
    Accept a base64-encoded JPEG frame from the browser,
    run hand landmark detection + MLP classification,
    return the predicted letter, confidence, and landmark coordinates.
    """
    body = await request.json()
    img_data = body.get("image", "")

    # Decode base64 image
    if "," in img_data:
        img_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {"letter": None, "confidence": 0, "landmarks": None}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_img)

    if not result.hand_landmarks:
        return {"letter": None, "confidence": 0, "landmarks": None}

    lm = result.hand_landmarks[0]

    # Classify
    feat = normalise(lm)
    tensor = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    conf, idx = probs.max(0)
    letter = classes[idx.item()]
    confidence = round(conf.item(), 3)

    # Return landmark coordinates for frontend drawing
    landmarks = [{"x": p.x, "y": p.y} for p in lm]

    return {
        "letter": letter,
        "confidence": confidence,
        "landmarks": landmarks,
        "connections": HAND_CONNECTIONS,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "classes": len(classes), "device": str(DEVICE)}
