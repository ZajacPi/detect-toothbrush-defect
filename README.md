# 🪥 Toothbrush Defect Instance Segmentation

This project focuses on the detection and segmentation of micro-defects on toothbrushes using **YOLOv11/v8** and **SAHI (Slicing Aided Hyper Inference)**. It is specifically optimized for small object detection in high-resolution industrial images.

## 🚀 Overview

Detecting tiny scratches or bristle deformations on a $1024 \times 1024$ image is a challenge for standard object detection models. This repository implements a **Tiling Strategy** to improve model sensitivity:
- **Training:** Done on $320 \times 320$ patches (tiles) to magnify defect features.
- **Inference:** Uses SAHI to "scan" large images window-by-window, ensuring no small defect is missed.

## 📊 Performance Results

The following results represent the best model performance on the validation set after training with optimized hyperparameters and tiling.

| Class | Images | Instances | Box (P) | Box (R) | Box mAP50 | Mask (P) | Mask (R) | Mask mAP50 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **all** | 24 | 25 | **0.887** | 0.630 | **0.727** | **0.776** | 0.554 | **0.573** |

### Key Takeaways:
- **High Precision (88.7%):** The model is extremely reliable in its detections, significantly reducing false positives in a production line environment.
- **Effective Segmentation:** Despite the small size of the defects, the mask precision remains high at 77.6%.

## 🛠️ Installation

Install the required dependencies using pip:

```bash
pip install ultralytics sahi opencv-python tqdm