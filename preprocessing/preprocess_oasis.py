import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_DIR = "Dataset/train/normal/OASIS/OASIS_Clean_Data"
OUTPUT_DIR = "Dataset/processed/train/normal"

IMG_SIZE = 128
DROP_RATIO = 0.15        # drop first & last 15%
EMPTY_THRESHOLD = 0.70   # 70% near-zero pixels
EPS = 1e-6
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize(volume):
    volume = volume.astype(np.float32)
    min_val, max_val = volume.min(), volume.max()
    return (volume - min_val) / (max_val - min_val + EPS)

def is_empty(slice_2d):
    zero_pixels = np.sum(slice_2d < 0.05)
    total_pixels = slice_2d.size
    return (zero_pixels / total_pixels) > EMPTY_THRESHOLD

slice_counter = 0

nii_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".nii") or f.endswith(".nii.gz")]

for file in tqdm(nii_files, desc="Processing MRI volumes"):
    path = os.path.join(INPUT_DIR, file)

    # Load MRI
    volume = nib.load(path).get_fdata()
    volume = normalize(volume)

    num_slices = volume.shape[2]
    start = int(num_slices * DROP_RATIO)
    end = int(num_slices * (1 - DROP_RATIO))

    for i in range(start, end):
        slice_2d = volume[:, :, i]

        if is_empty(slice_2d):
            continue

        slice_2d = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE))
        slice_2d = slice_2d.astype(np.float32)

        slice_counter += 1
        filename = f"img_{slice_counter:06d}.npy"
        np.save(os.path.join(OUTPUT_DIR, filename), slice_2d)

print(f"\n✅ Done. Saved {slice_counter} slices to {OUTPUT_DIR}")
