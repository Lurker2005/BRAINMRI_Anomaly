🧠 Brain MRI Anomaly Detection using Convolutional Autoencoder (CAE)

This project implements an unsupervised anomaly detection system for brain MRI images using a Convolutional Autoencoder (CAE) built with PyTorch.

The model is trained only on normal brain MRI slices and detects anomalies based on reconstruction error, without requiring labeled abnormal data.

📌 Problem Statement

Medical imaging faces several challenges:

Annotated abnormal MRI data is scarce

Manual labeling requires expert radiologists

Abnormalities vary widely in shape, size, and location

To address this, we use unsupervised learning, allowing the model to learn normal brain anatomy and detect deviations automatically.

🧠 Core Idea (How It Works)

Train a Convolutional Autoencoder using only normal MRI slices

The autoencoder learns to accurately reconstruct normal anatomy

When an abnormal MRI is passed:

Reconstruction quality degrades

Reconstruction error increases

High reconstruction error ⇒ Potential anomaly

This approach is commonly used in medical anomaly detection research.

🗂️ Dataset & Preprocessing
Dataset

Normal brain MRI scans (OASIS-style dataset)

Original format: 3D NIfTI volumes (.nii)

Preprocessing Pipeline

Each MRI volume is converted into clean, standardized 2D slices:

Extract axial slices

Discard top and bottom 15% slices

Keep middle 70% (brain-focused region)

Remove empty/background slices

Normalize intensities to [0, 1]

Resize to 128 × 128

Save as .npy for fast loading

⚠️ Only normal MRI slices are used during training

🏗️ Model Architecture
Convolutional Autoencoder (CAE)

Encoder

2D Convolution layers

ReLU activations

Strided convolutions for downsampling

Learns compact representation of normal brain structure

Decoder

Upsampling layers

Convolution + ReLU

Final Sigmoid activation (output in [0,1])

Reconstructs input MRI slice

The model is fully convolutional and works on batch inputs.

⚙️ Training Details

Framework: PyTorch

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Learning Rate: 1e-3

Training Type: Unsupervised

Training Data: Normal MRI slices only

Batch Size: Configurable (e.g., 32)

The model is trained to minimize reconstruction loss on normal samples.

🔍 Anomaly Detection Method

After training:

Freeze the model (model.eval())

Pass MRI slices through the autoencoder

Compute reconstruction error per slice

Use a statistical threshold:

threshold = mean(error) + 3 × std(error)


Error ≤ threshold → Normal

Error > threshold → Anomalous

🔥 Localization using Heatmaps

To visualize anomalies:

Compute pixel-wise reconstruction error

Normalize error map

Overlay error heatmap on original MRI slice

Highlights regions contributing most to anomaly

This provides interpretability, crucial for medical applications.

🌐 Inference API (Flask)

A production-ready Flask API is implemented:

Features

Accepts MRI image input

Runs trained CAE model

Returns:

Anomaly decision

Reconstruction error score

Heatmap (if anomalous)

This allows easy integration with:

Web apps

Hospital systems

Research pipelines

📁 Project Structure
.
├── Dataset/
│   ├── train/
│   │   └── normal/              # Raw MRI volumes (.nii)
│   └── processed/
│       └── train/
│           └── normal/          # Preprocessed slices (.npy)
│
├── preprocessing/
│   ├── preprocess_oasis.py
│   └── sanity_check.py
│
├── model/
│   └── autoencoder.py
│
├── training/
│   └── train.py
│
├── app.py                       # Flask inference API
├── requirements.txt
├── README.md
└── .gitignore

🚀 Future Improvements

Support full 3D MRI volumes

Multi-modal MRI (T1, T2, FLAIR)

Advanced CAE variants (VAE, CAE + Attention)

Proper medical benchmark datasets (BraTS)

Quantitative evaluation (ROC, AUC)

📚 Key Concepts Used

Unsupervised Learning

Convolutional Autoencoders

Medical Image Preprocessing

Reconstruction-based Anomaly Detection

Threshold-based Decision Making

Model Interpretability via Heatmaps

👤 Author

Darshan R (Lurker)
Machine Learning & Medical Imaging Enthusiast