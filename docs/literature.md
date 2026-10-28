# Literature Review

Notes on the sources that support the design decisions of this project.
Each entry follows the same template:

- **Problem:** What problem does the paper address?
- **Method:** What do the authors propose?
- **Key result:** What is the main finding?
- **Relevance to this project:** Which design decision does it support, and how?

Status: ⬜ not read · 🟨 skimmed (pass 1) · ✅ read and noted

---

## A. Dataset and Task

### ⬜ Gupta et al. (2019) — xBD: A Dataset for Assessing Building Damage from Satellite Imagery
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Weber & Kané (2020) — Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## B. Architecture

### ⬜ Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ He et al. (2016) — Deep Residual Learning for Image Recognition
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Daudt, Le Saux & Boulch (2018) — Fully Convolutional Siamese Networks for Change Detection
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Zheng et al. (2021) — ChangeOS: Building Damage Assessment with a Deep Object-Based Semantic Change Detection Framework
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## C. Loss Functions

### ⬜ Lin et al. (2017) — Focal Loss for Dense Object Detection
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Milletari et al. (2016) — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## D. Training Protocol

### ⬜ Prechelt (1998) — Early Stopping — But When?
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Kohavi (1995) — A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## E. Explainability (XAI)

### ⬜ Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Vinogradova et al. (2020) — Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Chattopadhay et al. (2018) — Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

### ⬜ Adebayo et al. (2018) — Sanity Checks for Saliency Maps
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## Design Decisions → Supporting Sources

| Design decision | Supporting sources |
|---|---|
| Use of the xBD dataset and its damage scale | |
| xView2 score as the main evaluation metric | |
| Siamese encoder with shared weights | |
| Fusion of pre/post features (concat / difference) | |
| U-Net decoder | |
| ResNet34 encoder (ImageNet pretrained) | |
| Focal Loss (class imbalance) | |
| Dice Loss | |
| Early stopping with patience | |
| Stratified train/val/test split | |
| Grad-CAM for segmentation | |
| Critical evaluation of saliency maps (deletion test) | |