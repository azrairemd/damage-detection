# Weekly Progress Log

A short record of the work done each week. Details: [ROADMAP.md](../ROADMAP.md), results in [`reports/`](../reports/).

| Week | Dates | Phases | Tag |
|---|---|---|---|
| 1 | Oct 26 – Nov 1 | 0–5: setup, literature, EDA, data preparation, training pipeline, baseline | `v0.0` – `v0.5` |
| 2 | Nov 2 – Nov 8 | 6–7: Siamese U-Net and ablations, advisor's requests, test evaluation | `v0.6-siamese`, `v0.7-test` |
| 3 | Nov 9 – Nov 15 | 8: Grad-CAM / XAI | `v0.8-core` |
| 4 | Nov 16 – Nov 22 | 9: TransUNet (planning) | |

---

## Week 1 (Oct 26 – Nov 1)

**Phase 0: Setup**
- GitHub repository, project structure, `requirements.txt`, installable package (`pyproject.toml`).
- Kaggle environment checked (`notebooks/01_setup_check.ipynb`); dataset: Kaggle copy of xBD (`qianlanzz/xbd-dataset`).

**Phase 1: Literature review** (`docs/literature.md`)
- Notes on xBD, Weber & Kané (2020), U-Net, ResNet, Siamese change detection (Daudt et al.), Focal loss, Dice loss (V-Net), early stopping (Prechelt), stratified splitting (Kohavi).

**Phase 2: Exploratory data analysis** (`notebooks/02_eda.ipynb`)
- 11,034 pre/post pairs, 19 events, 6 disaster types.
- Strong class imbalance: 96.5% of the pixels are background; 73.6% of the buildings are undamaged.
- Damage depends on the disaster type: 70% of minor damage comes from wind events, 68% of major damage from flooding; the earthquake event is 99.2% undamaged.

**Phase 3: Data preparation**
- Segmentation masks generated from the JSON labels (`src/data/masks.py`); 99.996% agreement with the pre-made masks.
- Stratified 60/20/20 split by disaster event: 6,620 / 2,207 / 2,207 pairs (`splits/`); leakage tests (`tests/test_split_leakage.py`).

**Phase 4: Training pipeline**
- Dataset with random 512 × 512 crops, flips and 90° rotations (`src/data/dataset.py`).
- xView2 metric, Focal + Dice loss, class weights, training loop with early stopping (patience = 20% of the epochs), checkpoints, logging, mixed precision.

**Phase 5: Baseline**
- U-Net with a ResNet34 encoder, post image only, Focal + Dice: **validation score 0.6095**.

---

## Week 2 (Nov 2 – Nov 8)

**Phase 6: Siamese U-Net and ablations** (`reports/results.md`)
- Siamese U-Net: shared ResNet34 encoder for the pre and post images, fusion at 5 scales, U-Net decoder (`src/models/siamese_unet.py`).
- Fusion (A2): concatenation 0.5981, absolute difference 0.5865 → concatenation kept.
- Loss (A3): Focal + Dice 0.5981, Focal only 0.6583, class-weighted CE 0.7079 (20 epochs); CE continued to 40 epochs: **0.7401**.

**Advisor's requests (Nov 7)**
- Model framework figure (`reports/figures/model_framework.png`), dataset description (`docs/dataset.md`), explanation of the concatenation (`docs/model.md`), damage level examples (`reports/figures/damage_examples.png`).
- Siamese U-Net with Focal + Dice, 40 epochs: validation 0.6285 (early stopping at epoch 20).

**Phase 7: Test evaluation** (`reports/test_results.md`)
- Test scores: baseline **0.6152**, Siamese Focal + Dice **0.6134**, Siamese CE **0.7198**.
- Destroyed pixels missed as background: 33.6% (baseline) → 18.9% (Siamese Focal + Dice) → 7.6% (Siamese CE).
- The Siamese models are better than the baseline in every disaster type.

---

## Week 3 (Nov 9 – Nov 15)

**Phase 8: Grad-CAM / XAI** (`reports/xai.md`)
- Literature: Grad-CAM, Grad-CAM++, Seg-Grad-CAM, sanity checks, shortcut learning, deletion test.
- Grad-CAM adapted to the Siamese segmentation model (`src/xai/`): wrapper for two inputs, segmentation target, decoder layer `blocks[1]`.
- 10 figures: one example per damage class, correct and wrong predictions, baseline vs. Siamese.
- Measurement on 100 test images: the models look at the buildings but also use the context; the suspected flooding → major shortcut was not confirmed; in tsunami images both models rely on the surroundings (houses washed away); the Siamese model uses the change around the buildings more than the baseline.

---

## Week 4 (Nov 16 – Nov 22)

**Advisor meeting notes**
- Replace the ResNet34 encoder of the Siamese U-Net with a **TransUNet** encoder (CNN + Transformer); keep the pre/post Siamese structure and the U-Net decoder.
- Build a mixed training set: **70% normal samples, 30% patch samples**.
- Four training runs:

| Run | Encoder | Data | Loss |
|---|---|---|---|
| 1 | TransUNet | Normal | Focal + Dice |
| 2 | TransUNet | Normal | Class-weighted CE |
| 3 | TransUNet | Patch (70% normal, 30% patch) | Focal + Dice |
| 4 | TransUNet | Patch | Class-weighted CE |

- Keep a weekly log of the work (this file).

**Decisions**
- Patch samples: a random 256 × 256 region, resized to 512 × 512 (pre, post and mask identically), so that the model also sees buildings at twice their size. Normal samples stay random 512 × 512 crops. Validation and test sets are unchanged.

**Done**
- Phase 8 merged into `main`, tag `v0.8-core`.
- Branch `phase-9-transunet` opened; plan for Phase 9.

**Next**
- TransUNet encoder: prototype, memory and speed test, first training run (Run 1).
- Patch dataset, while Run 1 is training.