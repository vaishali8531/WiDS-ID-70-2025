import cv2
import os
import numpy as np
import pandas as pd
import pickle

IMAGE_DIR = "../data/processed_images"
MODEL_PATH = "../data/best_model.pkl"
OUTPUT_CSV = "../data/final_predictions.csv"
OUTPUT_VIS_DIR = "../data/visual_outputs"

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

ROWS, COLS = 8, 8

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def extract_features(cell):
    mean_intensity = np.mean(cell)
    std_intensity = np.std(cell)
    edges = cv2.Canny(cell, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    return [mean_intensity, std_intensity, edge_density]

csv_rows = []

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

    predictions = []

    cell_id = 1
    for i in range(ROWS):
        for j in range(COLS):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = gray[y1:y2, x1:x2]
            features = extract_features(cell)

            pred = model.predict([features])[0]
            predictions.append(pred)

            # Highlight wildlife cells
            if pred == 1:
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

            cell_id += 1

    # Save visual output
    save_path = os.path.join(OUTPUT_VIS_DIR, image_name)
    cv2.imwrite(save_path, image)

    # Prepare CSV row
    row = [image_name] + predictions
    csv_rows.append(row)

# Save CSV
columns = ["ImageFileName"] + [f"c{i:02d}" for i in range(1, 65)]
df = pd.DataFrame(csv_rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("Inference completed. Outputs saved.")