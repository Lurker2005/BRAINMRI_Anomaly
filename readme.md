# 🧠 Brain MRI Anomaly Detection using Convolutional Autoencoder (CAE)

This project implements an **unsupervised anomaly detection system** for **brain MRI images** using a **Convolutional Autoencoder (CAE)** built with **PyTorch**.

The model is trained **only on normal MRI scans** and detects anomalies based on **reconstruction error**.

---

## 📌 Problem Statement

In medical imaging:
- Labeled abnormal data is limited
- Annotation is expensive and time-consuming
- Abnormal cases are highly diverse

This project addresses these challenges using **unsupervised learning**, where the model learns what *normal brain anatomy* looks like and flags deviations as anomalies.

---

## 🧠 Core Idea

1. Train an autoencoder **only on normal brain MRI slices**
2. The model learns to reconstruct normal anatomy well
3. When an abnormal MRI is passed:
   - Reconstruction quality drops
   - Reconstruction error increases
4. High reconstruction error ⇒ **Potential anomaly**

---

## 🗂️ Dataset & Preprocessing

### Dataset
- Normal brain MRI data (OASIS-style)
- Format: `.nii` (3D MRI volumes)

### Preprocessing Steps
- Convert 3D MRI volumes → **2D axial slices**
- Use **middle 70% slices** (discard empty/background regions)
- Normalize pixel values to `[0, 1]`
- Resize slices to `128 × 128`
- Save processed slices as `.npy`

Only **normal MRI slices** are used for training.

---

## 🏗️ Model Architecture

### Convolutional Autoencoder (CAE)

**Encoder**
- Convolution + ReLU layers
- Downsampling using stride-2 convolutions
- Learns compressed representation of normal brain structure

**Decoder**
- Upsampling layers
- Convolution + ReLU
- Final Sigmoid activation for normalized output

---

## ⚙️ Training Details

- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Learning Rate**: `1e-3`
- **Training Type**: Unsupervised
- **Training Data**: Normal MRI slices only

---

## 🔍 Anomaly Detection Strategy

After training:
1. Pass test MRI slices through the trained CAE
2. Compute reconstruction error
3. Higher error indicates deviation from learned normal patterns
4. These deviations are treated as anomalies

---

## 📁 Project Structure

```text
.
├── Dataset/
│   ├── train/
│   │   └── normal/          # Raw MRI data (.nii)
│   └── processed/
│       └── train/
│           └── normal/      # Processed slices (.npy)
│
├── preprocessing/
│   └── preprocess_oasis.py
│
├── models/
│   └── autoencoder.py
│
├── training/
│   └── train.py
│
├── README.md
└── .gitignore
