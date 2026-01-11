"""
Week 1 – Python & Data Science Basics
WildVision – Detecting Life in the Wilderness
WiDS Project | ID 70

This script covers:
- Python fundamentals
- Data handling using NumPy and Pandas
- CSV read/write operations
- Basic data visualization using Matplotlib
"""

# ================================
# 1. Python Basics
# ================================

animal_name = input("Enter animal name: ")
image_count = int(input("Enter number of images captured: "))

if image_count >= 5:
    print("Sufficient data collected")
else:
    print("Need more data")

# ================================
# 2. Lists and Dictionaries
# ================================

animal_list = ["Tiger", "Deer", "Elephant", "Leopard"]
animal_images = {
    "Tiger": 12,
    "Deer": 5,
    "Elephant": 8,
    "Leopard": 2
}

print("\nAnimal List:")
for animal in animal_list:
    print(animal)

print("\nImage Count per Animal:")
for animal, count in animal_images.items():
    print(f"{animal}: {count}")

# ================================
# 3. NumPy Basics
# ================================

import numpy as np

image_counts_array = np.array(list(animal_images.values()))

print("\nNumPy Operations:")
print("Image counts:", image_counts_array)
print("Mean images:", np.mean(image_counts_array))
print("Max images:", np.max(image_counts_array))

# ================================
# 4. Pandas – Tabular Data Handling
# ================================

import pandas as pd

data = {
    "Animal": list(animal_images.keys()),
    "Image_Count": list(animal_images.values())
}

df = pd.DataFrame(data)

print("\nDataFrame:")
print(df)

# Save DataFrame to CSV
df.to_csv("../data/animal_image_data.csv", index=False)
print("\nCSV file saved to data/animal_image_data.csv")

# Read CSV back
df_loaded = pd.read_csv("../data/animal_image_data.csv")
print("\nLoaded CSV Data:")
print(df_loaded)

# ================================
# 5. Basic Plotting with Matplotlib
# ================================

import matplotlib.pyplot as plt

plt.figure()
plt.bar(df_loaded["Animal"], df_loaded["Image_Count"])
plt.xlabel("Animal")
plt.ylabel("Number of Images")
plt.title("Wildlife Image Distribution")
plt.show()
