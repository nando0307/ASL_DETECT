"""
Real-time ASL detection using webcam.
Run from terminal:  python realtime_asl.py
Press 'q' to quit.
"""

import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ──
MODEL_PATH       = Path("asl_landmark_mlp.pth")
LANDMARKER_MODEL = Path("hand_landmarker.task")
NUM_FEATURES     = 63  # 21 landmarks × 3

# ── Hand skeleton connections ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── MLP (must match training architecture) ──
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

# ── Helpers ──
def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

def normalise(landmarks) -> np.ndarray:
    row = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    row -= row[0]
    scale = np.linalg.norm(row, axis=1).max()
    if scale > 0:
        row /= scale
    return row.flatten()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = LandmarkMLP(NUM_FEATURES, len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded — {len(classes)} classes: {classes}")

    # MediaPipe hand landmarker
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER_MODEL))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_landmarks(frame, lm)

            feat = normalise(lm)
            tensor = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            conf, idx = probs.max(0)
            label = classes[idx.item()]

            color = (0, 255, 0) if conf.item() > 0.6 else (0, 255, 255)
            cv2.putText(frame, f"{label} ({conf:.0%})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ASL Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
