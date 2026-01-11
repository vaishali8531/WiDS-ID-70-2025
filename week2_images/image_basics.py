"""
Week 2 – Image Processing with OpenCV
WildVision – Detecting Life in the Wilderness
WiDS Project | ID 70

This script demonstrates:
- Reading and displaying images
- Resizing images while maintaining aspect ratio
- Drawing an 8x8 grid on an image
"""

import cv2
import numpy as np

# ================================
# 1. Load Image
# ================================

image_path = "../data/sample_image.jpg"  # replace with your image path
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

# ================================
# 2. Resize Image to 800x600
# ================================

target_width = 800
target_height = 600

image_resized = cv2.resize(image, (target_width, target_height))

# ================================
# 3. Draw 8x8 Grid
# ================================

h, w, _ = image_resized.shape
rows = 8
cols = 8

row_step = h // rows
col_step = w // cols

grid_image = image_resized.copy()

# Draw horizontal lines
for i in range(1, rows):
    y = i * row_step
    cv2.line(grid_image, (0, y), (w, y), (0, 255, 0), 1)

# Draw vertical lines
for j in range(1, cols):
    x = j * col_step
    cv2.line(grid_image, (x, 0), (x, h), (0, 255, 0), 1)

# ================================
# 4. Display Results
# ================================

cv2.imshow("Original Image", image)
cv2.imshow("Resized Image with Grid", grid_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
