# WiDS-ID-70-2025
# WildVision – Detecting Life in the Wilderness
WiDS Project | ID 70

## Project Overview

WildVision is a computer vision–oriented data science project developed under the Women in Data Science (WiDS) program. The objective of the project is to establish a strong technical foundation in Python, data handling, image processing, and classical feature extraction techniques that are commonly used in wildlife monitoring and visual analytics. The project follows a structured three-week progression, moving from programming fundamentals to the generation of machine-learning-ready feature datasets from images.

## Objectives

The project aims to enable a smooth transition from C++ to Python while developing proficiency in scientific computing and image analysis. Key objectives include working with structured tabular data using NumPy and Pandas, understanding digital image representation, performing image manipulation using OpenCV, and extracting discriminative numerical features from images that can later be used for machine learning models.

## Technologies Used

WildVision is implemented using Python as the primary language, with NumPy and Pandas for numerical computation and data handling, Matplotlib for visualization, and OpenCV for image processing and feature extraction.

## Weekly Breakdown

### Week 1 – Python and Data Science Basics

Week 1 focuses on Python fundamentals and data analysis concepts. Topics include Python syntax, control flow, functions, and core data structures, followed by data operations using NumPy and Pandas. Reading and writing CSV files and basic data visualization using Matplotlib are also covered. Practical tasks involve writing small Python programs, exploring simple datasets such as Iris or Titanic, and plotting basic graphs. By the end of the week, the learner is expected to write Python scripts confidently and handle tabular datasets efficiently.

### Week 2 – Image Processing with OpenCV

Week 2 introduces the fundamentals of image processing. The focus is on understanding images as numerical arrays composed of pixels, including RGB and grayscale representations. Using OpenCV, the project covers image I/O operations, resizing and cropping with aspect ratio preservation, and drawing geometric primitives such as lines, rectangles, and grids. Tasks include resizing images to 800×600 resolution and overlaying an 8×8 grid, building the foundation for region-based feature extraction.

### Week 3 – Feature Extraction and Data Preparation

Week 3 focuses on converting image data into structured numerical representations suitable for machine learning. Feature extraction techniques such as Histogram of Oriented Gradients (HOG), HSV color features, and Local Binary Patterns (LBP) are explored, along with edge detection methods and basic texture statistics. Images are divided into 8×8 grids, and features are computed per cell and stored in CSV format. By the end of the week, a complete feature dataset is generated, ready for downstream machine learning tasks.

## Methodology

The project follows a bottom-up technical approach, beginning with programming and data handling, progressing to image representation and manipulation, and finally transforming visual information into numerical feature vectors. This staged methodology ensures interpretability, modularity, and scalability of the overall pipeline.

## How to Run the Code

1. Clone the repository or download the project directory.
2. Ensure Python 3.8 or later is installed on your system.
3. Install the required dependencies using:
   pip install numpy pandas matplotlib opencv-python
4. Navigate to the desired week directory.

   * Week 1 scripts are located in `week1_python/`
   * Week 2 scripts are located in `week2_images/`
   * Week 3 scripts are located in `week3_features/`
5. Run the Python files using:
   python filename.py
6. Input datasets and images should be placed in the `data/` directory as required by the scripts.

## Future Scope

Future extensions include training supervised machine learning models on the extracted features, scaling the pipeline to real-world wildlife camera trap datasets, and exploring deep learning approaches such as convolutional neural networks for end-to-end wildlife detection.

## Author

WiDS Project Participant
Project ID: 70
WildVision – Detecting Life in the Wilderness

