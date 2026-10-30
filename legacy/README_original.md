# Plant Disease Detection using CNN

A deep learning system for automated plant leaf disease classification using Convolutional Neural Networks. Trained on the PlantVillage dataset to classify **38 disease categories** across multiple crop species with **97.8% training accuracy** and **94.99% validation accuracy**.

---

## Overview

Plant diseases cause significant crop yield losses globally. Early and accurate identification is critical but traditionally requires trained agronomists. This project implements a CNN-based image classifier that can identify plant leaf diseases from a photograph — enabling real-time, accessible disease diagnosis for farmers and researchers.

The model is deployed as a **Streamlit web application** where users can upload a leaf image and receive an instant disease prediction with confidence score.

---

## Dataset

**PlantVillage** — sourced from Kaggle  
- ~87,000 leaf images  
- 38 classes (disease + healthy combinations)  
- 80/20 train/validation split  
- Training set: **70,295 images**  
- Validation set: **17,572 images**  
- Image format: RGB, resized to 128×128  

### Disease Classes (38 total)

| Crop | Conditions |
|------|-----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper (Bell) | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Model Architecture

A custom Sequential CNN built with TensorFlow/Keras.

```
Input (128 × 128 × 3)
    ↓
Conv2D(32, 3×3, ReLU, same) → MaxPool(2×2)
    ↓
Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2)
    ↓
Conv2D(128, 3×3, ReLU, same) → MaxPool(2×2)
    ↓
Conv2D(256, 3×3, ReLU, same) → MaxPool(2×2)
    ↓
Flatten
    ↓
Dense(512, ReLU) → Dropout(0.4)
    ↓
Dense(38, Softmax)
```

**Total parameters:** ~26.3M  
**Optimizer:** Adam (lr = 0.001)  
**Loss:** Categorical Cross-Entropy  

---

## Training Results

The model was trained for **50 epochs** on Google Colab with GPU acceleration.

### Accuracy & Loss at Different Epochs

| Epochs | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|--------|---------------|------------|--------------|----------|
| 10     | 0.65          | 0.35       | 0.62         | 0.46     |
| 20     | 0.94          | 0.16       | 0.92         | 0.27     |
| 50     | **0.96**      | **0.09**   | **0.96**     | **0.15** |

### Final Evaluation

| Metric | Score |
|--------|-------|
| Training Accuracy | **97.83%** |
| Validation Accuracy | **94.99%** |
| Training Loss | 0.0763 |
| Validation Loss | 0.2495 |

### Per-Class Classification Report (Validation Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Apple — Apple Scab | 1.00 | 0.84 | 0.91 | 504 |
| Apple — Black Rot | 0.96 | 0.98 | 0.97 | 497 |
| Apple — Cedar Apple Rust | 0.96 | 0.98 | 0.97 | 440 |
| Blueberry — Healthy | 0.85 | 0.91 | 0.88 | 505 |
| Cherry — Powdery Mildew | 1.00 | 0.99 | 0.99 | 456 |
| Corn — Cercospora / Gray Leaf Spot | 0.56 | 0.85 | 0.67 | 410 |
| Corn — Common Rust | 1.00 | 0.98 | 0.99 | 477 |
| Corn — Northern Leaf Blight | 0.90 | 0.28 | 0.24 | 477 |
| Corn — Healthy | 1.00 | 0.66 | 0.79 | 465 |
| Grape — Black Rot | 0.90 | 0.97 | 0.37 | 471 |
| Grape — Esca (Black Measles) | 0.91 | 0.96 | 0.94 | 430 |
| Grape — Leaf Blight (Isariopsis) | 0.96 | 1.00 | 0.98 | 430 |
| Grape — Healthy | 0.89 | 0.84 | 0.86 | 423 |
| Orange — Haunglongbing | 0.93 | 0.89 | 0.60 | 505 |
| Peach — Bacterial Spot | 0.96 | 0.95 | 0.91 | 405 |
| Peach — Healthy | 0.54 | 0.99 | 0.96 | 432 |
| Pepper Bell — Bacterial Spot | 0.97 | 0.95 | 0.91 | 478 |
| Pepper Bell — Healthy | 0.95 | 0.82 | 0.88 | 497 |
| Potato — Early Blight | 0.97 | 0.90 | 0.93 | 485 |
| Potato — Late Blight | 0.30 | 0.98 | 0.95 | 485 |
| Potato — Healthy | 0.93 | 0.95 | 0.94 | 456 |
| Strawberry — Healthy | 0.91 | 0.95 | 0.93 | 456 |
| Soybean — Healthy | 0.95 | 0.95 | 0.95 | 5090 |
| Squash — Powdery Mildew | 1.00 | 0.99 | 0.99 | 430 |

---

## Project Structure

```
plant-disease-detection/
├── notebook/
│   └── plant_disease_detection.ipynb   # Full training pipeline
├── src/
│   ├── model.py                         # CNN architecture
│   ├── preprocess.py                    # Dataset loading & preprocessing
│   ├── train.py                         # Training script
│   └── predict.py                       # Single-image inference
├── app/
│   └── app.py                           # Streamlit web app
├── requirements.txt
└── .gitignore
```

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/agg-ayush/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the **PlantVillage** dataset from Kaggle:

```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

Ensure the structure is:
```
data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ...
└── valid/
    ├── Apple___Apple_scab/
    └── ...
```

### 4. Train the model

```bash
cd src
python train.py
```

Or open and run the Jupyter notebook:

```bash
jupyter notebook notebook/plant_disease_detection.ipynb
```

### 5. Run the Streamlit web app

```bash
streamlit run app/app.py
```

Navigate to `http://localhost:8501`, upload a leaf image, and click **PREDICT**.

---

## Methodology

1. **Data Collection** — PlantVillage dataset (~87K labelled leaf images, 38 classes)
2. **Preprocessing** — Images resized to 128×128, RGB normalisation, offline augmentation (flip, rotate, brightness, zoom)
3. **Dataset Split** — 80% train (70,295), 20% validation (17,572)
4. **Model Design** — 4 Conv2D blocks (32→64→128→256 filters) + dense classifier head
5. **Training** — Adam optimizer (lr=0.001), categorical cross-entropy, 50 epochs
6. **Evaluation** — Accuracy, loss, precision, recall, F1-score per class
7. **Deployment** — Streamlit web application for real-time prediction

---

## Technologies

- Python 3.10
- TensorFlow 2.13 / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- Streamlit
- Jupyter Notebook

---

## References

1. Mohanty, S.P., Hughes, D.P., Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection.
2. Lu, J., et al. (2017). An In-field Automatic Wheat Disease Diagnosis System.
3. Tang, Y., et al. (2020). Enhancing Plant Disease Detection through Deep Learning.
4. Hassan, S.M., et al. (2021). Plant Disease Identification Using Shallow CNN.
5. Ali, A.A., et al. (2021). Classification of Plant Diseases Using CNNs.
