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

### ✅ Gupta et al. (2019) — xBD: A Dataset for Assessing Building Damage from Satellite Imagery
- **Problem:** Automated building damage assessment from satellite imagery lacked a large-scale, labeled dataset of pre- and post-disaster image pairs.
- **Method:** xBD is built from Maxar's Open Data Program imagery (GSD < 0.8 m). It contains 22,068 images (11,034 pre/post pairs) from 19 disaster events across 6 disaster types, with about 850,000 manually annotated building polygons. Damage is labeled on a four-level ordinal scale (Joint Damage Scale): no damage, minor damage, major damage and destroyed. Buildings that cannot be reliably assessed are labeled as un-classified.
- **Key result:** A two-stage baseline (a U-Net-like model for building localization, followed by a CNN classifier applied to each building crop) localizes buildings well, but four-level damage classification remains much harder, largely due to class imbalance.
- **Relevance to this project:** xBD is our training and evaluation dataset. Its damage scale defines our 5-class mask (background + 4 damage levels), and un-classified buildings are excluded from the loss (ignore index 255). The class imbalance motivates our Focal + Dice loss. Unlike the two-stage baseline, we use a single-stage Siamese U-Net. Label noise from manual annotation is noted as a limitation.

### ✅ Weber & Kané (2020) — Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion
- **Problem:** How to frame and train a model on xBD to substantially improve over the two-stage xBD baseline.
- **Method:** A single network (Mask R-CNN with FPN, ImageNet-pretrained) jointly predicts a 5-class per-pixel map (0 = no building, 1–4 = damage levels). Pre- and post-disaster images are processed separately with shared weights, and their features are concatenated before the semantic segmentation head. Each 1024×1024 image is split into four 512×512 quadrants, and the cross-entropy loss is weighted inversely proportional to class frequency.
- **Key result:** Overall F1 of 0.741 compared to 0.265 for the xBD baseline (holdout set). Minor damage remains the hardest class (F1 = 0.493). Feeding pre/post images as a single 6-channel input or as a difference image performed poorly. When trained too long, the model collapses into predicting only the "no building" class.
- **Relevance to this project:** Supports our Siamese encoder with shared weights, 512×512 tiling, an imbalance-aware loss and early stopping. It also defines the overall score as 30% localization F1 + 70% damage F1. Our multi-scale fusion of pre/post features follows the future direction suggested by the authors.
---

## B. Architecture

### ✅ Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- **Problem:** Pixel-level segmentation (assigning a class to every pixel) with very few annotated training images, as is common in biomedical imaging.
- **Method:** An encoder–decoder architecture with a U shape. A contracting path (encoder) captures context by repeatedly downsampling, and a symmetric expanding path (decoder) upsamples back to the input resolution for precise localization. Skip connections concatenate encoder features with the decoder features at the same resolution. Strong data augmentation (including elastic deformations) compensates for the small training set.
- **Key result:** Outperformed the previous best method on the ISBI challenge for neuronal structure segmentation in electron microscopy and won the ISBI cell tracking challenge 2015 by a large margin. Segmenting a 512×512 image takes less than a second on a GPU.
- **Relevance to this project:** Our decoder follows the U-Net design. Skip connections are essential for drawing precise boundaries of small objects such as buildings. In our Siamese model, the skip connections carry the fused pre/post features at each scale. Unlike the original U-Net, we use an ImageNet-pretrained ResNet34 encoder and predict 5 classes.

### ✅ He et al. (2016) — Deep Residual Learning for Image Recognition
- **Problem:** Very deep networks suffer from the degradation problem: adding more layers increases even the training error, because deep plain networks are hard to optimize.
- **Method:** Residual learning. Each block learns a residual function F(x), and a shortcut connection adds the block's input to its output: F(x) + x. Shortcuts make it easy to learn identity mappings and help gradients flow through very deep networks. The paper defines a family of networks (ResNet-18/34/50/101/152).
- **Key result:** Networks up to 152 layers were trained successfully, and deeper networks gave higher accuracy. ResNet won 1st place in the ILSVRC 2015 classification task, and its features transferred well to detection and segmentation tasks.
- **Relevance to this project:** We use an ImageNet-pretrained ResNet34 as the encoder of our Siamese U-Net. Pretrained weights provide general visual features (edges, textures, shapes) and reduce the amount of training data needed. ResNet34 offers a good trade-off between capacity and computational cost on Kaggle GPUs. Its multi-scale feature maps connect naturally to the U-Net skip connections.

### ✅ Daudt, Le Saux & Boulch (2018) — Fully Convolutional Siamese Networks for Change Detection
- **Problem:** Pixel-wise binary change detection (changed / unchanged) between two co-registered images of the same area taken at different times.
- **Method:** Three U-Net-based fully convolutional architectures: FC-EF (early fusion: the two images are stacked as one input), FC-Siam-conc (two encoder streams with shared weights; their features are concatenated in the skip connections) and FC-Siam-diff (the absolute difference of the two streams' features is passed through the skip connections). All models are trained from scratch.
- **Key result:** The proposed architectures outperform previous change detection methods while being at least 500 times faster. There is no single best fusion strategy: FC-EF and FC-Siam-diff perform similarly on RGB images (F1 48.89 vs 48.86 on OSCD), while FC-Siam-diff performs best on 13-band multispectral images (F1 57.92).
- **Relevance to this project:** Main source of our Siamese encoder with shared weights and multi-scale fusion through the skip connections. Since neither fusion strategy dominates, we compare concatenation, difference and both in ablation A2. Unlike this work, we use an ImageNet-pretrained encoder; a Siamese design keeps each stream at 3 channels, so pretrained weights can be used directly, which may explain why early fusion performed poorly in Weber & Kané (2020).

### ⬜ Zheng et al. (2021) — ChangeOS: Building Damage Assessment with a Deep Object-Based Semantic Change Detection Framework
- **Problem:**
- **Method:**
- **Key result:**
- **Relevance to this project:**

---

## C. Loss Functions

### ✅ Lin et al. (2017) — Focal Loss for Dense Object Detection
- **Problem:** In dense object detection, the vast number of easy background examples dominates the cross-entropy loss and overwhelms the few hard examples, which the authors identify as the central cause of the lower accuracy of one-stage detectors.
- **Method:** Focal Loss adds a modulating factor (1 − p_t)^γ to the cross-entropy loss: FL(p_t) = −α_t (1 − p_t)^γ log(p_t). Well-classified examples (high p_t) are down-weighted, so training focuses on hard, misclassified examples. γ = 0 recovers the standard cross-entropy; α_t is an optional class-balancing weight. A simple one-stage detector, RetinaNet, is designed to evaluate the loss.
- **Key result:** With γ = 2, Focal Loss improves over α-balanced cross-entropy by 2.9 AP, and changing γ has a much larger effect than changing α. RetinaNet trained with Focal Loss surpasses the accuracy of all previous one-stage and two-stage detectors.
- **Relevance to this project:** Pixel-wise damage segmentation suffers from the same kind of imbalance: most pixels are background and most buildings are undamaged, while minor damage is both rare and hard. We use Focal Loss with γ = 2 and per-class α weights for our 5 classes, ignoring un-classified pixels (index 255). Its contribution is measured against cross-entropy in ablation A3.

### ✅ Milletari et al. (2016) — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
- **Problem:** Segmentation of 3D medical volumes (prostate MRI), where the structure of interest occupies a very small region and there is a strong imbalance between foreground and background voxels.
- **Method:** A volumetric fully convolutional network trained end-to-end, with a novel objective function based on the Dice coefficient, which measures the overlap between the prediction and the ground truth (Dice loss = 1 − Dice). Because Dice is a ratio computed per structure, small structures contribute as much as large ones, without explicit re-weighting. Random non-linear transformations and histogram matching are used for data augmentation.
- **Key result:** The approach achieves good performance on challenging test data while requiring only a fraction of the processing time of previous methods.
- **Relevance to this project:** Source of the Dice loss, which we combine with Focal Loss (L = λ·Focal + (1 − λ)·Dice). Dice handles class imbalance from a different angle than Focal Loss and is equivalent to the F1 score, so it directly optimizes the quantity used in the xView2 metric. We extend it to 5 classes by averaging per-class Dice scores, excluding un-classified pixels (index 255). The benefit of the combination is evaluated in ablations A3 and A4.

---

## D. Training Protocol

### ✅ Prechelt (1998) — Early Stopping — But When?
- **Problem:** Early stopping (stopping training when the validation error starts to increase) is widely used to prevent overfitting, but the stopping criterion is usually chosen in an ad hoc way. In practice, validation error curves are noisy and have multiple local minima, so stopping at the first increase is often premature.
- **Method:** Three classes of stopping criteria are defined and compared empirically: GL (stop when the generalization loss, the relative increase of the validation error over its minimum so far, exceeds a threshold), PQ (GL divided by the training progress), and UP (stop when the validation error has increased for several successive evaluation periods). The weights with the lowest validation error are used as the final model.
- **Key result:** There is a trade-off between training time and generalization: on average, slower criteria improve generalization by about 4% but require about 4 times longer training.
- **Relevance to this project:** Theoretical background for our early stopping rule. Our patience criterion (stop when the validation xView2 score has not improved for a number of epochs equal to 20% of the maximum number of epochs) is a modern variant of the UP class. Scaling patience with the training length tolerates the noisy validation curve while keeping training time bounded on Kaggle GPUs. The checkpoint with the best validation score (best.pt) is used as the final model.

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
| Use of the xBD dataset and its damage scale | Gupta et al. (2019) |
| xView2 score as the main evaluation metric | Gupta et al. (2019), Weber & Kané (2020) |
| Siamese encoder with shared weights | Weber & Kané (2020) |
| Fusion of pre/post features (concat / difference) | Weber & Kané (2020) |
| Focal Loss (class imbalance) | Weber & Kané (2020) |
| Early stopping with patience | Weber & Kané (2020) |
| U-Net decoder | Ronneberger et al. (2015) |
| ResNet34 encoder (ImageNet pretrained) | He et al. (2016) |
| Siamese encoder with shared weights | Daudt et al. (2018), Weber & Kané (2020) |
| Fusion of pre/post features (concat / difference) | Daudt et al. (2018), Weber & Kané (2020) |
| Focal Loss (class imbalance) | Lin et al. (2017), Weber & Kané (2020) |
| Dice Loss | Milletari et al. (2016) |
| Early stopping with patience | Prechelt (1998), Weber & Kané (2020) |
| Stratified train/val/test split | |
| Grad-CAM for segmentation | |
| Critical evaluation of saliency maps (deletion test) | |