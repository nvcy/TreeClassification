# 🌳 TreeSpicesClassification – Multimedia Data Processing Project

**3D → 2D → Feature Extraction → Classical ML → Deep Learning → 3D Learning**

## 📌 Project Overview

**TreeSpicesClassification** is a Multimedia Data Processing project that focuses on the classification of 3D tree leaf/branch models across several species using three major methodological families:

- **Indirect Methods** - Convert 3D to 2D, extract classical features
- **Quasi-Direct Methods** - Multi-view 2D deep learning
- **Direct 3D Deep Learning Methods** - Direct point cloud processing

The goal is to compare how different feature extraction, representation, and learning techniques perform on the same 3D dataset.

**Key Details:**
- The dataset contains **3D point clouds of 7 species of trees**
- Processed models with 2D/3D feature extraction
- Trained classical ML and deep learning models
- Comprehensive evaluation and visualization of results

---

## 🏗️ Project Architecture

```
TreeClassification/
│
├── Utils/
│   ├── data_loader.py              # Load & preprocess point cloud data
│   ├── preprocessing.py            # Data cleaning & normalization
│   ├── projections.py              # 3D → 2D projection converter
│   ├── feature_extraction.py       # Extract LBP, PFH, FPFH, CNN features
│   └── helpers.py                  # Utilities & visualization
│
├── Datasets/
│   ├── Train/                      # 80% training split
│   └── Test/                       # 20% testing split
│
├── Notebooks/
│   ├── LBP_Feature_Extraction.ipynb
│   ├── SVM_Classification.ipynb
│   ├── CNN_From_Scratch.ipynb
│   ├── ResNet_FineTuning.ipynb
│   ├── ResNet_FeatureExtraction_SVM.ipynb
│   ├── ResNet_TransferLearning.ipynb
│   ├── Fusion_CNN.ipynb
│   ├── QuasiDirect_PFH_SVM.ipynb
│   ├── Direct_PointNet.ipynb
│   └── Direct_DGCNN.ipynb
│
├── Outputs/
│   ├── MultiView_Data/
│   │   └── (5 projections per 3D model)
│   │
│   ├── Features/
│   │   ├── 2D_LBP_Features.npy
│   │   ├── 2D_FPFH_Features.npy
│   │   └── Features.csv
│   │
│   ├── Features3D/
│   │   ├── PFH_3D.npy
│   │   ├── FPFH_3D.npy
│   │   └── Descriptors/
│   │
│   └── Models/                     # (for future model saving)
│
└── README.md
```

---

## 🌱 Dataset Description

The dataset contains **3D point cloud files** for 7 different tree species.

### Data Distribution

All samples were preprocessed and split into:
- **Train:** 80%
- **Test:** 20%

### Dataset Location

```
Datasets/Train/     # Training data (80% split)
Datasets/Test/      # Testing data (20% split)
```

Additional intermediate datasets (2D projections, extracted features) are saved under `Outputs/`.

---

## 🧪 Methods & Experiments

Below are the three large families of methods implemented in the notebooks.

### 🔹 1. Indirect Methods (2D Feature Extraction + Classical ML)

These methods convert 3D point clouds into 2D images, then extract classical features.

**Steps:**
1. Convert 3D point cloud → 2D projection images
2. Extract 2D features:
   - **LBP** (Local Binary Patterns)
   - **HOG** (if used)
3. Flatten & normalize features
4. Train classical ML models:
   - **SVM** (primary model used)
   - KNN / Random Forest (optional)

**Related Notebooks:**
- `LBP_Feature_Extraction.ipynb`
- `SVM_Classification.ipynb`

**Outputs Saved:**
Located in `Outputs/Features/`:
- `2D_LBP_Features.npy`
- Feature matrices (.npy / .csv)

---

### 🔹 2. Quasi-Direct Methods (Multi-View 2D Deep Learning)

Quasi-direct methods preserve partial 3D information through multi-view projection.

**Steps:**
1. Generate 5 projections per 3D model (top, bottom, side, oblique, etc.)
2. Train 2D CNN models:
   - **CNN from Scratch**
   - **ResNet** (Transfer Learning)
   - **ResNet** (Fine-Tuning)
3. Extract ResNet deep features and classify with SVM
4. (Optional) Fusion of multi-view CNN predictions

**Related Notebooks:**
- `CNN_From_Scratch.ipynb`
- `ResNet_FineTuning.ipynb`
- `ResNet_FeatureExtraction_SVM.ipynb`
- `ResNet_TransferLearning.ipynb`
- `Fusion_CNN.ipynb`

**Outputs Saved:**
Located in `Outputs/MultiView_Data/`:
- 5 projected images per 3D object

Located in `Outputs/Features/`:
- Feature matrices extracted using CNN / ResNet

---

### 🔹 3. Direct Methods (3D Deep Learning)

These methods operate directly on 3D point cloud data without converting them to 2D.

**State-of-the-Art Architectures:**
- **PointNet** - Pioneering direct point cloud processing
- **DGCNN** (Dynamic Graph CNN) - Graph-based point cloud learning

**Steps:**
1. Load point cloud file
2. Normalize and sample points
3. Train PointNet/DGCNN
4. Evaluate on 20% test set

**Related Notebooks:**
- `Direct_PointNet.ipynb`
- `Direct_DGCNN.ipynb`

**Outputs Saved:**
Located in `Outputs/Features3D/`:
- PFH / FPFH 3D descriptors
- 3D deep features
- Normalized point cloud samples

---

## 🧰 Utilities (Utils Folder)

The `Utils/` directory contains reusable functions:

| File | Purpose |
|------|---------|
| `data_loader.py` | Load point cloud data, normalize, split |
| `preprocessing.py` | Denoising, normalization, format conversion |
| `projections.py` | Convert 3D → 2D (multi-view generator) |
| `feature_extraction.py` | Extract LBP, PFH, FPFH, CNN features |
| `helpers.py` | Plotting, metrics, model utilities |

---

## 📊 Outputs Summary

### ✔ `Outputs/MultiView_Data/`
Contains all 2D projections of the 3D dataset
- 5 images per 3D object
- Used by CNN/ResNet models

### ✔ `Outputs/Features/`
LBP, FPFH, CNN extracted features (.npy, .csv formats)
- Classical ML feature matrices
- Deep learning embeddings

### ✔ `Outputs/Features3D/`
3D features for direct learning methods
- PointNet embeddings
- PFH / FPFH descriptors
- Normalized point clouds

### ✔ `Outputs/Models/`
(Models not saved — folder available for future expansion)

---

## 🧠 Evaluation

### Evaluation Metrics

Evaluation metrics applied across all methods:
- **Accuracy** - Overall correctness
- **Precision / Recall / F1-score** - Per-class performance
- **Confusion Matrix** - Error analysis
- **Training Time** - Computational efficiency
- **Robustness** - Noise, rotation, point density variations

### Comparative Analysis

| Method | Type | Description | Scope |
|--------|------|-------------|-------|
| **Indirect** | 2D + Classical ML | LBP + SVM | Fast, simple |
| **Quasi-Direct** | Multi-view CNN | ResNet, fine-tuning, fusion | Strong performance |
| **Direct** | 3D Deep Learning | PointNet, DGCNN | Highest geometric fidelity |

---

## 🚀 How to Use

### 1️⃣ Prepare Environment

```bash
pip install -r requirements.txt
```

### 2️⃣ Explore Notebooks

Open any notebook in the `Notebooks/` folder to run experiments:
- Choose based on the method you want to explore
- Each notebook is self-contained with explanations

### 3️⃣ Generate 2D Views (Optional)

```bash
python Utils/projections.py
```

This will generate 5 projections for each 3D model and save to `Outputs/MultiView_Data/`.

### 4️⃣ Extract Features

```bash
python Utils/feature_extraction.py
```

This will extract LBP, FPFH, and other features, saving results to `Outputs/Features/`.

### 5️⃣ Train Models

Choose the notebook depending on your method:

- **Classical ML:** `SVM_Classification.ipynb`
- **2D CNN:** `CNN_From_Scratch.ipynb`
- **ResNet Transfer Learning:** `ResNet_TransferLearning.ipynb`
- **3D Direct - PointNet:** `Direct_PointNet.ipynb`
- **3D Direct - DGCNN:** `Direct_DGCNN.ipynb`

---

## 👥 Team

This project was developed for the **Multimedia Data Processing** module.

**Contributors:**
- Your Name(s)

**Supervision:** [Teacher/Professor Name] (optional)

**Repository:** [https://github.com/nvcy/TreeClassification](https://github.com/nvcy/TreeClassification)

---

## 🏁 Conclusion

**TreeSpicesClassification** demonstrates how data representation dramatically influences classification performance.

By comparing **Indirect**, **Quasi-Direct**, and **Direct** approaches, this project provides a complete view of the challenges and trade-offs in processing 3D multimedia data:

- 📊 **Indirect methods** offer simplicity and speed
- 🎨 **Quasi-direct methods** balance 3D information with 2D CNN advantages
- 🔷 **Direct 3D methods** leverage geometric fidelity for optimal performance

This systematic exploration enables researchers and practitioners to make informed decisions when choosing methodologies for 3D object classification tasks.

---

## 📝 License

[Specify your license here]

## 📧 Contact

For questions or inquiries, please reach out via the GitHub repository.

---

**Last Updated:** November 26, 2025
