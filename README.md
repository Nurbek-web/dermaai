# Derma Diagnostics: Privacy-Preserving Skin Lesion Triage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Enabled-orange.svg)](https://www.tensorflow.org/js)

A lightweight, browser-based skin lesion classifier designed for rural clinics. This project implements a MobileNetV1 CNN fine-tuned on the HAM10000 dataset to classify 7 types of skin lesions directly on client devices, ensuring patient privacy and enabling offline usage.

## 📄 Abstract

Early melanoma detection saves lives, yet rural clinics often lack dermatologists and reliable connectivity. We present a privacy-preserving, browser-based skin-lesion classifier that returns top-three diagnostic predictions in under 2 seconds on standard smartphones. Using the HAM10000 dataset with leakage-safe lesion-level splitting, we fine-tuned MobileNet with the final 30 layers trainable. The model achieves **97.9% top-3 accuracy** and **90.8% melanoma sensitivity** at 95% specificity. The TensorFlow.js model runs entirely client-side, safeguarding patient privacy while enabling rapid, accurate triage in resource-limited settings.

## 🚀 Key Features

*   **Privacy-Preserving**: All inference happens locally in the browser using TensorFlow.js. No patient images are uploaded to a server.
*   **Offline Capable**: Once loaded, the application works without an internet connection, suitable for remote areas.
*   **High Performance**: 97.9% top-3 accuracy and <2s inference time on mobile devices.
*   **Leakage-Safe Training**: Implements lesion-level stratified splitting to prevent data leakage (different images of the same lesion never appear in both train and validation sets).

## 📂 Repository Structure

```
├── data/                   # Dataset storage (excluded from git)
│   ├── processed_grouped/  # Organized train/val folders after running grouped_split.py
│   └── splits/             # Split metadata (CSVs)
├── src/
│   ├── grouped_split.py    # Script for leakage-safe data preparation
│   └── prepare_data.py     # Basic data preparation script
├── notebooks/
│   └── derma_training.ipynb # Main training and evaluation notebook (Colab compatible)
├── paper/                  # LaTeX source for the manuscript
├── benchmark/              # Browser-based benchmarking tool
└── derma_diagnostics_manuscript.md # Full project manuscript
```

## 🛠️ Getting Started

### Prerequisites

*   Python 3.8+
*   Node.js (for local serving if needed, though Python `http.server` works)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/derma-diagnostics.git
    cd derma-diagnostics
    ```

2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1.  Download the **HAM10000** dataset from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or Kaggle.
2.  Place the unzipped files (`HAM10000_images_part_1`, `HAM10000_images_part_2`, and `HAM10000_metadata.tab`) into the `data/` directory.
3.  Run the grouped split script to prepare the data with leakage prevention:
    ```bash
    python src/grouped_split.py
    ```
    This will create `data/processed_grouped/` with `train` and `val` subdirectories.

### Training

Open `notebooks/derma_training.ipynb` in Jupyter Notebook or Google Colab to reproduce the training process. The notebook covers:
*   Data loading with augmentation
*   MobileNetV1 transfer learning
*   Evaluation metrics (Sensitivity, Specificity, ROC-AUC)
*   Calibration (Temperature Scaling)
*   Conversion to TensorFlow.js format

## 📊 Results

The model was evaluated on a held-out validation set of 159 unique lesions (1,001 images).

| Metric | Value |
|:--- |:--- |
| **Top-3 Accuracy** | **97.88%** |
| **Melanoma Sensitivity** | **90%** (at 95% Specificity) |
| **Macro F1-Score** | **0.721** |
| **Inference Time (Desktop)** | ~447ms |
| **Inference Time (Mobile)** | ~1.1s - 1.2s |

## 🤝 Contributing

Contributions are welcome! Please verify that any data processing changes respect the lesion-level splitting to ensure rigorous evaluation.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this code or methodology, please cite:

> [Authors]. "Derma Diagnostics: Lightweight CNN for Client‐Side Skin‐Lesion Triage in Rural Clinics." (2025).

