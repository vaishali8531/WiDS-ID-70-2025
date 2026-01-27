import cv2
import os

RAW_DIR = "../data/raw_images"
PROCESSED_DIR = "../data/processed_images"

os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET_W, TARGET_H = 800, 600
TARGET_RATIO = 4 / 3

def crop_to_4_3(image):
    h, w, _ = image.shape
    current_ratio = w / h

    if abs(current_ratio - TARGET_RATIO) < 0.01:
        return image

    if current_ratio > TARGET_RATIO:
        new_w = int(h * TARGET_RATIO)
        x_start = (w - new_w) // 2
        return image[:, x_start:x_start + new_w]
    else:
        new_h = int(w / TARGET_RATIO)
        y_start = (h - new_h) // 2
        return image[y_start:y_start + new_h, :]

for filename in os.listdir(RAW_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(RAW_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    image = crop_to_4_3(image)

    h, w, _ = image.shape
    if w > TARGET_W or h > TARGET_H:
        image = cv2.resize(image, (TARGET_W, TARGET_H))

    save_path = os.path.join(PROCESSED_DIR, filename)
    cv2.imwrite(save_path, image)

print("Image preprocessing completed.")