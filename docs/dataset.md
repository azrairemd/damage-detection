# Dataset: xBD

This document describes the dataset used in this project: where it comes from, what it contains, how the labels are defined, how the data is distributed, and how it was prepared for training.

---

## 1. Source

| | |
|---|---|
| **Dataset** | xBD: A Dataset for Assessing Building Damage from Satellite Imagery (Gupta et al., 2019) |
| **Original source** | [xView2 challenge](https://xview2.org/), paper: [arXiv:1911.09296](https://arxiv.org/abs/1911.09296) |
| **Copy used in this project** | Kaggle: [`qianlanzz/xbd-dataset`](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset) |
| **Path on Kaggle** | `/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd` |
| **Imagery** | Maxar / DigitalGlobe Open Data Program, high-resolution RGB satellite images (ground sample distance below 0.8 m) |

---

## 2. Content

xBD contains pairs of satellite images of the same area, taken **before** (pre) and **after** (post) a natural disaster. Every building is annotated with a polygon and, in the post-disaster image, with a damage level.

| | |
|---|---|
| Images | **22,068** RGB images, 1024 × 1024 pixels |
| Pairs | **11,034** pre/post pairs |
| Disaster events | **19** |
| Disaster types | **6**: wind, flooding, fire, tsunami, earthquake, volcano |
| Annotated buildings | **425,368** (post-disaster labels) |

### 2.1 Folder structure of the Kaggle copy

| Folder | Pairs | Images | Labels (JSON) | Pre-made masks | Used |
|---|---|---|---|---|---|
| `tier1` | 2,799 | 5,598 | ✅ | ✅ | ✅ |
| `tier3` | 6,369 | 12,738 | ✅ | ✅ | ✅ |
| `test` | 933 | 1,866 | ✅ | ✅ | ✅ |
| `hold` | 933 | 1,866 | ✅ | ✅ | ✅ |
| `train` | — | 10 | ❌ | ❌ | ❌ (a small sample of `tier1`) |
| **Total used** | **11,034** | **22,068** | | | |

These are the **original xBD folders**, not our train/val/test split.

File naming:
```
<folder>/images/<image_id>_<phase>_disaster.png     # phase = pre | post
<folder>/labels/<image_id>_<phase>_disaster.json
```
Example: `tier1/images/hurricane-harvey_00000001_pre_disaster.png`

### 2.2 Label files (JSON)

Each image has a JSON label file with:
- `metadata`: disaster name, disaster type, sensor, capture date, off-nadir angle, ground sample distance
- `features`: one polygon per building, in two coordinate systems: geographic (`lng_lat`) and **pixel (`xy`, used in this project)**. In post-disaster labels, each building has a `subtype` field with its damage level.

### 2.3 Disaster events

| Disaster type | Events |
|---|---|
| Wind | hurricane-matthew, hurricane-michael, joplin-tornado, moore-tornado, tuscaloosa-tornado |
| Flooding | hurricane-florence, hurricane-harvey, midwest-flooding, nepal-flooding |
| Fire | portugal-wildfire, pinery-bushfire, santa-rosa-wildfire, socal-fire, woolsey-fire |
| Tsunami | palu-tsunami, sunda-tsunami |
| Earthquake | mexico-earthquake |
| Volcano | guatemala-volcano, lower-puna-volcano |

The number of images per event is shown in `notebooks/02_eda.ipynb`.

---

## 3. Damage classes

xBD uses an ordinal damage scale with four levels, plus an "un-classified" label:

| Class | Mask value | Description |
|---|---|---|
| Background | **0** | Not a building |
| No damage | **1** | No visible damage; roof and walls intact |
| Minor damage | **2** | Slight, surface-level damage: small roof damage, water marks, light discoloration or ash |
| Major damage | **3** | Clear structural damage: partial collapse of walls or roof, debris around the building |
| Destroyed | **4** | Building completely collapsed, burned or gone |
| Un-classified | **255** | Damage level could not be determined (e.g. clouds, smoke, shadows). **Ignored in the loss and the metrics.** |

Examples of each damage level (pre image, post image and mask) are shown in `reports/figures/damage_examples.png`.

---

## 4. Exploratory data analysis: key findings

Full analysis: `notebooks/02_eda.ipynb`.

### 4.1 Class imbalance

**Buildings per class** (post-disaster labels):

| Disaster type | No damage | Minor | Major | Destroyed | Un-classified | Total |
|---|---|---|---|---|---|---|
| Wind | 64,927 | 25,737 | 7,556 | 10,704 | 3,857 | 112,781 |
| Flooding | 71,148 | 10,122 | 20,241 | 1,596 | 3,557 | 106,664 |
| Fire | 61,992 | 704 | 726 | 11,338 | 2,553 | 77,313 |
| Tsunami | 60,874 | 1 | 1,278 | 7,382 | 3,201 | 72,736 |
| Earthquake | 51,084 | 221 | 54 | 3 | 111 | 51,473 |
| Volcano | 3,008 | 75 | 49 | 537 | 732 | 4,401 |
| **Total** | **313,033** | **36,860** | **29,904** | **31,560** | **14,011** | **425,368** |
| **Share** | **73.6%** | 8.7% | 7.0% | 7.4% | 3.3% | |

**Pixels per class** (training set):

| Background | No damage | Minor | Major | Destroyed |
|---|---|---|---|---|
| 96.46% | 2.67% | 0.32% | 0.34% | 0.21% |

- **96.5% of all pixels are background.** Destroyed pixels are about 450 times rarer than background pixels.
- Among buildings, **73.6% are undamaged.**
- → This strong imbalance motivates the class-weighted Focal loss and the Dice loss (computed on the damage classes only).

### 4.2 Imbalance between disaster types

- **Fire** events make up **57.7% of the images** but only **18.2% of the buildings**.
- **Earthquake** makes up only **1.7% of the images** but **12.1% of the buildings**.
- **35% of the images contain no building at all**.

### 4.3 Damage depends on the disaster type

- **Minor damage** comes mostly from wind events (**70%**), **major damage** mostly from flooding (**68%**).
- The only earthquake event (Mexico City) is **99.2% undamaged**, with only **3 destroyed buildings**. The model cannot learn earthquake damage from xBD; earthquake results are only meaningful for localization and the no-damage class.
- Risk of **shortcut learning**: the model may learn the disaster context (flood water, burn scars) instead of the building damage itself. To be checked with Grad-CAM (Phase 8).

### 4.4 Other findings

- Pre and post labels contain the **same buildings** for all 11,034 pairs → masks are drawn from the post labels.
- **3.3%** of the buildings are un-classified → value 255, ignored.

---

## 5. Data preparation

### 5.1 Masks

Segmentation masks are generated from the **post-disaster JSON labels** (`src/data/masks.py`):
- Each building polygon (pixel coordinates) is filled with its class value (0–4, 255).
- Coordinates are clipped to the image borders.
- Where buildings overlap, the more severe damage is drawn on top (`DRAW_ORDER = [255, 1, 2, 3, 4]`).
- **Validation:** compared with the pre-made masks of the Kaggle copy, our masks agree on **99.996%** of the pixels (excluding un-classified buildings, which the pre-made masks draw as background).
- 11,034 masks saved as PNG, stored as the output of the Kaggle notebook `03_preprocessing_split`.

### 5.2 Train / validation / test split

All 11,034 pairs from the four folders are pooled and split **60 / 20 / 20**, **stratified by disaster event** (`src/data/split.py`, `train_test_split`):

| Split | Pairs | Share | File |
|---|---|---|---|
| Train | 6,620 | 60% | `splits/train.csv` |
| Validation | 2,207 | 20% | `splits/val.csv` |
| Test | 2,207 | 20% | `splits/test.csv` |

- The split is done at the **pair level**: the pre image, post image and mask of a pair are always in the same subset.
- Every one of the 19 events appears in all three subsets with the same proportion.
- `tests/test_split_leakage.py` checks that no pair is in two subsets and that all pairs are assigned.
- Because the original `test` and `hold` folders are pooled and re-split, our test scores are **not directly comparable** to the xView2 challenge leaderboard.

### 5.3 Input pipeline (`src/data/dataset.py`)

| | Training | Validation / test |
|---|---|---|
| Image size | Random **512 × 512** crop from the 1024 × 1024 image (no resizing, buildings keep their size) | Full **1024 × 1024** image |
| Augmentation | Horizontal flip, vertical flip, 90° rotations — identical for pre, post and mask | None |
| Normalization | ImageNet mean and standard deviation | Same |

---

## 6. Citation

```bibtex
@article{gupta2019xbd,
  title   = {xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author  = {Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and
             Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  journal = {arXiv preprint arXiv:1911.09296},
  year    = {2019}
}
```