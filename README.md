# Building Damage Assessment from Satellite Imagery

Pixel-level building damage segmentation from pre- and post-disaster satellite image pairs, using a **Siamese U-Net** trained on the **xBD** dataset, with **Grad-CAM** explanations of the model's decisions.

> 🎓 Undergraduate graduation project · 🚧 Work in progress

---

## 📈 Progress

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & data acquisition | ✅ Done |
| 1 | Literature review | ⏳ Planned |
| 2 | Exploratory data analysis (EDA) | ⏳ Planned |
| 3 | Preprocessing & stratified split | ⏳ Planned |
| 4 | Training pipeline | ⏳ Planned |
| 5 | Baseline model | ⏳ Planned |
| 6 | Siamese model & ablations | ⏳ Planned |
| 7 | Test evaluation | ⏳ Planned |
| 8 | Grad-CAM / XAI | ⏳ Planned |
| 9 | Fine-tuning on Türkiye  earthquake data | ⏳ Planned |
| 10 | Report & presentation | ⏳ Planned |

Detailed tasks for each phase: [ROADMAP.md](ROADMAP.md)

---

## 🧠 Method

- **Input:** A pre-disaster and a post-disaster RGB image of the same area
- **Model:** Siamese encoder with shared weights → feature fusion at each scale → U-Net decoder
- **Output:** 5-class segmentation mask — background, no damage, minor damage, major damage, destroyed
- **Loss:** Focal Loss + Dice Loss
- **Data split:** 60 / 20 / 20, stratified by disaster event
- **Early stopping:** patience = 20% of the maximum number of epochs
- **Evaluation:** xView2 score (localization + damage F1), per-class F1/IoU, confusion matrix
- **Explainability (XAI):** Grad-CAM heatmaps showing which regions drive each damage prediction

---

## 🗂️ Dataset

[xBD](https://xview2.org/) (Gupta et al., 2019): 22,068 high-resolution satellite images (11,034 pre/post pairs, 1024×1024) from 19 disaster events, with building polygons labeled by damage level.

- Source used in this project: Kaggle — [`qianlanzz/xbd-dataset`](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)
- The dataset is **not** included in this repository. It is distributed under a non-commercial license by its original authors.

---

## 📁 Project Structure

```
damage-detection/
├── configs/      # Experiment configurations (YAML)
├── splits/       # Train/val/test split files (CSV)
├── src/          # Source code (installable package)
│   ├── data/     # Data loading, preprocessing, splitting
│   ├── models/   # Model architectures
│   ├── losses/   # Loss functions
│   ├── metrics/  # Evaluation metrics
│   ├── xai/      # Grad-CAM and explainability
│   └── utils/    # Helper functions
├── notebooks/    # Kaggle notebooks for each phase
├── tests/        # Unit tests
├── docs/         # Literature notes and data documentation
└── reports/      # Figures and result tables
```

---

## 🚀 Getting Started

### On Kaggle

1. Create a new notebook and add the xBD dataset as input.
2. Run the following in the first cell:

```python
!git clone https://github.com/azrairemd/damage-detection.git
%cd damage-detection
!pip install -q -r requirements.txt
!pip install -q -e .
```

3. Import from the package:

```python
from src.data.xbd import load_pair, list_image_ids
```

### Locally

```bash
git clone https://github.com/azrairemd/damage-detection.git
cd damage-detection
pip install -r requirements.txt
pip install -e .
```

---

## 📄 License

The xBD dataset is subject to its own license. See [xview2.org](https://xview2.org/) for terms of use.