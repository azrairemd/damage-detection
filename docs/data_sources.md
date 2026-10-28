# Data Sources

## xBD

**xBD: A Dataset for Assessing Building Damage from Satellite Imagery** (Gupta et al., 2019)

- 22,068 high-resolution satellite images (1024×1024, RGB) → **11,034 pre/post-disaster pairs**
- 19 disaster events (earthquakes, floods, wildfires, hurricanes, volcanic eruptions, ...)
- Building polygons labeled with damage level: no-damage, minor-damage, major-damage, destroyed, un-classified
- Official website: [xview2.org](https://xview2.org/)
- Paper: [arXiv:1911.09296](https://arxiv.org/abs/1911.09296)

### Source used in this project

- Kaggle: [`qianlanzz/xbd-dataset`](https://www.kaggle.com/datasets/qianlanzz/xbd-dataset)
- Mount path on Kaggle: `/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd`
- Verified in [`notebooks/01_setup_check.ipynb`](../notebooks/01_setup_check.ipynb)

### Folder structure

| Folder | Images | Pairs | Labels (JSON) | Pre-made masks | Used |
|---|---|---|---|---|---|
| `tier1` | 5,598 | 2,799 | ✅ | ✅ | ✅ (= official train split) |
| `tier3` | 12,738 | 6,369 | ✅ | ✅ | ✅ |
| `test` | 1,866 | 933 | ✅ | ✅ | ✅ |
| `hold` | 1,866 | 933 | ✅ | ✅ | ✅ |
| `train` | 10 | — | ❌ | ❌ | ❌ (sample subset of `tier1`) |
| **Total used** | **22,068** | **11,034** | | | |

File naming convention:

```
<folder>/images/<image_id>_<phase>_disaster.png     # phase = pre | post
<folder>/labels/<image_id>_<phase>_disaster.json
```

Example: `tier1/images/hurricane-harvey_00000001_pre_disaster.png`

### Pre-made masks (provided by the Kaggle uploader)

| File | Shape | Values | Meaning |
|---|---|---|---|
| `*_pre_disaster.png` | (1024, 1024) | {0, 255} | Binary building localization mask |
| `*_post_disaster.png` | (1024, 1024) | class ids | Damage mask |
| `*_post_disaster_rgb.png` | (1024, 1024, 3) | {0, 255} | Color visualization only |

In this project, masks are **generated from the JSON labels** (Phase 3) to fully control the class encoding. The pre-made masks are used only as a cross-check.

### Notes

- The official `test` and `hold` sets are pooled together with `tier1` and `tier3`, and re-split into our own 60/20/20 train/val/test split (stratified by disaster event). Therefore, our test scores are **not directly comparable** to the xView2 challenge leaderboard.
- Some images contain **no-data regions** (black areas). To be analyzed in Phase 2.
- JSON label format: to be documented in Phase 3.

### License

xBD is released for **non-commercial** use by its authors. See [xview2.org](https://xview2.org/) for the exact terms. The data is **not** redistributed in this repository.

### Citation

```bibtex
@article{gupta2019xbd,
  title   = {xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author  = {Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and
             Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  journal = {arXiv preprint arXiv:1911.09296},
  year    = {2019}
}
```