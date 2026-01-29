import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_mri2.pth")

IMG_SIZE = 128
ANOMALY_THRESHOLD = 0.0003

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
# PREPROCESS
# =====================================================
def preprocess_image(image_file):
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    return img.to(device)

# =====================================================
# HEATMAP → BASE64
# =====================================================
def generate_heatmap_base64(original, reconstructed):
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

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = preprocess_image(request.files["file"])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    with torch.no_grad():
        recon = model(img)
        loss = criterion(recon, img).item()

    is_anomaly = loss > ANOMALY_THRESHOLD

    response = {
        "anomaly": bool(is_anomaly),
        "reconstruction_error": loss,
        "threshold": ANOMALY_THRESHOLD,
        "heatmap_base64": None
    }

    if is_anomaly:
        response["heatmap_base64"] = generate_heatmap_base64(
            img.cpu().numpy()[0, 0],
            recon.cpu().numpy()[0, 0]
        )

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
