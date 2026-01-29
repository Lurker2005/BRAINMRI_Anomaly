import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_mri2.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

UPLOAD_DIR = "uploads"
IMG_SIZE = 128
ANOMALY_THRESHOLD = 0.0003   # <-- use YOUR validation threshold

os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# AUTOENCODER MODEL
# =====================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =====================================================
# LOAD MODEL
# =====================================================
model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

criterion = nn.MSELoss(reduction="mean")

# =====================================================
# PREPROCESS FUNCTION (MUST MATCH TRAINING)
# =====================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img.to(device)

# =====================================================
# HEATMAP GENERATION
# =====================================================
def generate_heatmap(original, reconstructed, save_path):
    error = np.abs(original - reconstructed)
    error = (error - error.min()) / (error.max() - error.min() + 1e-8)

    heatmap = cv2.applyColorMap(
        (error * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    original_rgb = cv2.cvtColor(
        (original * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)

# =====================================================
# FLASK API
# =====================================================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)

    with torch.no_grad():
        recon = model(img)
        loss = criterion(recon, img).item()

    is_anomaly = loss > ANOMALY_THRESHOLD

    response = {
        "anomaly": bool(is_anomaly),
        "reconstruction_error": loss,
        "threshold": ANOMALY_THRESHOLD
    }

    if is_anomaly:
        original = img.cpu().numpy()[0, 0]
        reconstructed = recon.cpu().numpy()[0, 0]

        heatmap_path = os.path.join(UPLOAD_DIR, "heatmap.png")
        generate_heatmap(original, reconstructed, heatmap_path)

        response["heatmap_path"] = heatmap_path

    return jsonify(response)

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
