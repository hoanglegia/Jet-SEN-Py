# tiny-fingerprint-recognition

TinyML-based fingerprint recognition system for resource-constrained embedded systems (MCUs, ESP32, STM32).

## Overview

This project implements a lightweight machine learning pipeline for fingerprint recognition using a Siamese Network architecture. The pipeline is designed to be optimized for deployment on microcontrollers and covers:

1.  **Data loading & preprocessing** – Supports the [SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing), handling both "Real" fingerprint images and "Altered" versions (Easy, Medium, Hard).
2.  **Feature engineering** – Implements image normalization, resizing (90x90), and real-time data augmentation (Gaussian blur, affine transformations) using the `imgaug` library.
3.  **Lightweight model training** – Siamese Neural Network architecture designed to learn embedding similarities between fingerprint pairs.
4.  **Model compression** – Implements Post-Training Symmetric Min-Max Quantization to reduce model weight precision to INT8.
5.  **TFLite export** – Conversion of the Keras model to `.tflite` format with optimizations for embedded inference.
6.  **C/C++ deployment** – Generates a `.h` header file containing the quantized model as a static byte array, ready for integration into firmware toolchains (Arduino, ESP-IDF, STM32Cube).

## Project Structure

```text
fingerprint_ids/
├── data/
│   └── loader.py          # Dataset loading and DataGenerator logic
├── features/
│   └── engineer.py        # Image preprocessing and augmentation
├── models/
│   ├── classifiers.py     # Siamese Network architecture   
│   └── compress.py        # Symmetric Min-Max Quantization
└── deploy/
    ├── export.py          # TFLite conversion utilities
    └── c_export.py        # C array header generation for MCUs

scripts/
└── run_pipeline.py        # End-to-end training and deployment pipeline

requirements.txt           # Project dependencies
```

## Quick Start
```Bash
# Install dependencies
pip install -r requirements.txt

# Ensure your dataset is structured as follows:
# /Real
# /Altered/Altered-Easy
# /Altered/Altered-Medium
# /Altered/Altered-Hard

# Run the full pipeline (Load, Train, Compress, Export to C)
python -m scripts.run_pipeline
```

## Pipeline Workflow
After running the pipeline, the system performs the following steps:
1. Augmentation: Applies random noise and rotations to the training set to improve model robustness.
2. Siamese Training: Trains the model to output a similarity score (0 to 1) between two fingerprint inputs.
3. Quantization: Compresses the 32-bit floats into 8-bit integers using Symmetric Min-Max technique.
4. Export: Transforms the binary TFLite model into a C-compatible header file.

## Embedded Deployment
The ``run_pipeline.py`` script produces the following artifacts for deployment:
- ``fingerprint_model.tflite`` – Optimized model for mobile/embedded interpreters.
- ``fingerprint_model_data.h`` – C header containing the model as **weight** - an ``unsigned char`` array.

To deploy, include the header in your C project and load it using the TFLite Micro library:
```C
#include "fingerprint_model_data.h"

// Initialize TFLite Micro interpreter
model = tflite::GetModel(fingerprint_model);
```

## Memory Footprint (Estimated)
| Component            | Size (FP32)   | Size (INT8/TFLite) |
|----------------------|---------------|--------------------|
| Siamese CNN Model    | ~1.2 MB       | ~300 KB            |
| Input Buffer (90x90) | 31.6 KB       | 8.1 KB             |
| Feature Embeddings   | 0.25 KB       | 0.06 KB            |

## Requirements
- Python 3.8+
- TensorFlow 2.15.0
- OpenCV
- Imgaug
- Scikit-learn
- Numpy