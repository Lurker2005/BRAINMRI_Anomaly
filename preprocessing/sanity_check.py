import numpy as np
import matplotlib.pyplot as plt
import os
import random

path = "Dataset/processed/train/normal"
files = random.sample(os.listdir(path), 10)

for f in files:
    img = np.load(os.path.join(path, f))
    plt.imshow(img, cmap="gray")
    plt.title(f)
    plt.axis("off")
    plt.show()
