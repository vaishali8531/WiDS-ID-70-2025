"""
Week 3 – Feature Extraction & Data Preparation
WildVision – Detecting Life in the Wilderness
WiDS Project | ID 70

This script:
- Divides an image into 8x8 grid cells
- Extracts basic features from each cell
- Stores features in a CSV file
"""

import cv2
import numpy as np
import pandas as pd

# ================================
# 1. Load Image
# ================================

image_path = "../data/sample_image.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found.")

image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ================================
# 2. Feature Functions
# ================================

def extract_features(cell):
    """
    Extracts simple texture and intensity features from an image cell
    """
    mean_intensity = np.mean(cell)
    std_intensity = np.std(cell)

    edges = cv2.Canny(cell, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return mean_intensity, std_intensity, edge_density

# ================================
# 3. Divide Image into 8x8 Grid
# ================================

rows, cols = 8, 8
h, w = gray.shape
row_step = h // rows
col_step = w // cols

feature_data = []

cell_id = 0

for i in range(rows):
    for j in range(cols):
        y1 = i * row_step
        y2 = (i + 1) * row_step
        x1 = j * col_step
        x2 = (j + 1) * col_step

        cell = gray[y1:y2, x1:x2]
        mean_val, std_val, edge_density = extract_features(cell)

        feature_data.append([
            cell_id, i, j, mean_val, std_val, edge_density
        ])

        cell_id += 1

# ================================
# 4. Save Features to CSV
# ================================

columns = [
    "cell_id",
    "row_index",
    "col_index",
    "mean_intensity",
    "std_intensity",
    "edge_density"
]

df_features = pd.DataFrame(feature_data, columns=columns)
df_features.to_csv("../data/image_features.csv", index=False)

print("Feature extraction complete. CSV saved to data/image_features.csv")
