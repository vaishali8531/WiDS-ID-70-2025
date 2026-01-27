import cv2
import os
import numpy as np
import pandas as pd

IMAGE_DIR = "../data/processed_images"
OUTPUT_CSV = "../data/image_features.csv"

ROWS, COLS = 8, 8

def extract_features(cell):
    mean_intensity = np.mean(cell)
    std_intensity = np.std(cell)

    edges = cv2.Canny(cell, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return mean_intensity, std_intensity, edge_density


feature_rows = []

for image_name in os.listdir(IMAGE_DIR):
    if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    cell_h = h // ROWS
    cell_w = w // COLS

    cell_id = 1

    for i in range(ROWS):
        for j in range(COLS):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = gray[y1:y2, x1:x2]

            mean_i, std_i, edge_d = extract_features(cell)

            feature_rows.append([
                image_name,
                cell_id,
                mean_i,
                std_i,
                edge_d
            ])

            cell_id += 1


columns = [
    "image_name",
    "cell_id",
    "mean_intensity",
    "std_intensity",
    "edge_density"
]

df = pd.DataFrame(feature_rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("Feature extraction completed. CSV saved.")