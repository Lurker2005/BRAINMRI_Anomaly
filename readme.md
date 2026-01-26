# 🧠 Brain MRI Anomaly Detection using Convolutional Autoencoder (CAE)

This project implements an **unsupervised anomaly detection system** for **brain MRI scans** using a **Convolutional Autoencoder (CAE)**.  
The model is trained **only on normal MRI slices** and detects anomalies based on **reconstruction error**.

---

## 📌 Motivation

In medical imaging, labeled abnormal data is often:
- scarce
- expensive
- biased

Unsupervised anomaly detection solves this by learning **what is normal**, then flagging deviations as anomalies.

This project focuses on:
- learning normal brain anatomy
- reconstructing normal MRI slices
- identifying abnormal regions via poor reconstruction

---

## 🧠 Approach Overview

### 1️⃣ Data Preprocessing
- Input: 3D brain MRI volumes (`.nii`)
- Convert volumes → **2D axial slices**
- Use only **middle 70% slices** (discard empty regions)
- Normalize intensities to `[0, 1]`
- Resize to `128 × 128`
- Save as `.npy`

Only **normal MRI data** is used for training.

---

### 2️⃣ Model: Convolutional Autoencoder (CAE)

**Encoder**
- Extracts hierarchical spatial features
- Progressively downsamples input

**Decoder**
- Upsamples latent representation
- Reconstructs the original MRI slice

The model learns a **compressed representation of normal brain structure**.

---

### 3️⃣ Training Strategy
- **Unsupervised learning**
- Input = Target
- Loss function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Trained only on **normal MRI slices**

---

### 4️⃣ Anomaly Detection (Post-Training)
- Pass test MRI slices through trained CAE
- Compute **reconstruction error**
- High reconstruction error ⇒ potential anomaly

---

## 📂 Project Structure

```text
.
├── Dataset/
│   ├── train/
│   │   └── normal/          # raw normal MRI (.nii)
│   └── processed/
│       └── train/
│           └── normal/      # processed 2D slices (.npy)
│
├── models/
│   └── autoencoder.py
│
├── training/
│   └── train.py
│
├── preprocessing/
│   └── preprocess_oasis.py
│
├── README.md
└── .gitignore
