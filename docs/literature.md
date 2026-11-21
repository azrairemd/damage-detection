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

### ✅ Kohavi (1995) — A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection
- **Problem:** Estimating the accuracy of a model and selecting the best model with limited data, where the way the data is split affects the reliability of the estimate.
- **Method:** A large-scale empirical comparison (over half a million runs of C4.5 and Naive Bayes on real-world datasets) of cross-validation, with different numbers of folds and with or without stratification, and bootstrap, with different numbers of samples. In stratified splitting, each fold preserves the class proportions of the full dataset.
- **Key result:** For real-world datasets, the best method for model selection is stratified ten-fold cross-validation, even when more folds are computationally affordable.
- **Relevance to this project:** Supports our use of a stratified split, so that the validation and test sets are representative of the full dataset. Since images contain several damage classes, we stratify by disaster event, which preserves the strongly imbalanced event distribution of xBD in all three subsets. Unlike this study, we use a single 60/20/20 hold-out split instead of cross-validation, because training a deep model multiple times is not feasible with our computational budget.

---

## E. Explainability (XAI)

### 🟨 Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- **Problem:** CNNs make accurate predictions but do not show which parts of the input led to a decision, which makes it hard to trust them or to find their failure modes.
- **Method:** Gradient-weighted Class Activation Mapping. The gradients of the score of a class with respect to the feature maps of a convolutional layer are averaged over the spatial positions, giving one importance weight per channel. The feature maps are summed with these weights and passed through a ReLU, so that only regions that increase the class score remain. The resulting coarse heat map is upsampled to the input size. It works with any CNN without changing its architecture or retraining it.
- **Key result:** Outperforms earlier methods on weakly-supervised localization (ILSVRC-15) without architectural changes. Human studies show that the explanations help users decide which of two models is more reliable, and they reveal dataset bias (a model that distinguished doctors from nurses by looking at faces and hair, i.e. gender, instead of medical equipment).
- **Relevance to this project:** Main XAI method of Phase 8. We use it to see which regions of the pre and post images drive the predicted damage class, and to check for dataset bias in the form of shortcut learning (see Geirhos et al., 2020).

### 🟨 Chattopadhay et al. (2018) — Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks
- **Problem:** Grad-CAM averages the gradients over the whole feature map; when several objects of the same class appear in an image, the map often highlights only part of them.
- **Method:** Grad-CAM++ weights the gradients pixel by pixel (using higher-order derivatives) instead of a plain average, so that every instance of the class contributes to the channel weights.
- **Key result:** Better localization of the objects and better coverage of multiple instances of the same class than Grad-CAM; in human studies, its explanations were preferred more often.
- **Relevance to this project:** Optional comparison (bonus). xBD images often contain many buildings of the same damage class, which is exactly the case Grad-CAM++ was designed for.

### 🟨 Vinogradova, Dibrov & Myers (2020) — Towards Interpretable Semantic Segmentation via Gradient-weighted Class Activation Mapping
- **Problem:** Grad-CAM explains one score per image (classification), but a segmentation network produces one score per pixel and class.
- **Method:** Seg-Grad-CAM: the target of Grad-CAM is the sum of the scores (logits) of a class over a selected set of pixels, e.g. a single pixel, an object or all pixels predicted as that class. The rest of Grad-CAM is unchanged.
- **Key result:** Qualitative results with a U-Net on the Cityscapes dataset show that the method produces meaningful explanations for a chosen class and region.
- **Relevance to this project:** Defines our Grad-CAM target: the sum of the logits of damage class c over the pixels predicted as c. Our model is also a U-Net, so the method applies directly.

### 🟨 Adebayo et al. (2018) — Sanity Checks for Saliency Maps
- **Problem:** Saliency maps are judged mostly by how convincing they look, which does not show that they actually depend on what the model has learned.
- **Method:** Two randomization tests: (1) model parameter randomization, i.e. the maps of a trained model are compared with those of the same model with random weights; (2) data randomization, i.e. a model trained on randomly permuted labels. A method that produces the same maps in both cases does not explain the model.
- **Key result:** Some popular methods (e.g. Guided Backpropagation and Guided Grad-CAM) produce almost the same maps for a random model and behave like edge detectors. Plain gradients and Grad-CAM pass the tests.
- **Relevance to this project:** Supports the choice of plain Grad-CAM (not Guided Grad-CAM). Grad-CAM maps are still interpreted carefully, as hints about where the model looks rather than as proof.

### 🟨 Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks
- **Problem:** Deep networks often reach high benchmark scores with decision rules that do not transfer to real-world conditions.
- **Method:** Perspective paper that defines shortcuts as decision rules that work on the standard benchmark but fail on data from a different distribution, and reviews examples from several fields (e.g. recognizing the typical background instead of the object).
- **Key result:** Shortcut learning is a common failure mode across deep learning. Models should be tested on out-of-distribution data, and their decisions should be examined, not only their scores.
- **Relevance to this project:** In xBD, damage levels are strongly linked to the disaster type (EDA: 70% of minor damage comes from wind events, 68% of major damage from flooding). The model might therefore learn disaster context such as flood water or burn scars instead of the damage of the building itself. We use Grad-CAM to check this, and the Türkiye transfer (Phase 9, optional) would act as an out-of-distribution test.

### 🟨 Petsiuk, Das & Saenko (2018) — RISE: Randomized Input Sampling for Explanation of Black-box Models
- **Problem:** Explanation maps are difficult to evaluate objectively, since there is no ground truth for "where the model looks".
- **Method:** Besides the RISE explanation method itself, the paper introduces the deletion and insertion metrics: the most important pixels according to the map are removed (or added) step by step, and the change of the class score is measured; the area under this curve summarizes the faithfulness of the map.
- **Key result:** A faithful explanation causes a fast drop of the class score in the deletion test (low area under the curve).
- **Relevance to this project:** Basis of the optional deletion test (bonus): if Grad-CAM highlights the regions the model really uses, masking them should lower the predicted damage score more than masking random regions.

**Library:** `pytorch-grad-cam` (Gildenblat et al., https://github.com/jacobgil/pytorch-grad-cam), package `grad-cam` in `requirements.txt`.

---

## Design Decisions → Supporting Sources

| Design decision | Supporting sources |
|---|---|
| Use of the xBD dataset and its damage scale | Gupta et al. (2019) |
| xView2 score as the main evaluation metric | Gupta et al. (2019), Weber & Kané (2020) |
| U-Net decoder | Ronneberger et al. (2015) |
| ResNet34 encoder (ImageNet pretrained) | He et al. (2016) |
| Siamese encoder with shared weights | Daudt et al. (2018), Weber & Kané (2020) |
| Fusion of pre/post features (concat / difference) | Daudt et al. (2018), Weber & Kané (2020) |
| Focal Loss (class imbalance) | Lin et al. (2017), Weber & Kané (2020) |
| Dice Loss | Milletari et al. (2016) |
| Class-weighted cross-entropy | Weber & Kané (2020) |
| Early stopping with patience | Prechelt (1998), Weber & Kané (2020) |
| Stratified train/val/test split | Kohavi (1995) |
| Grad-CAM as the XAI method | Selvaraju et al. (2017), Adebayo et al. (2018) |
| Grad-CAM for segmentation (target = class scores over the predicted region) | Vinogradova et al. (2020) |
| Shortcut learning check | Geirhos et al. (2020), Selvaraju et al. (2017) |
| Critical evaluation of saliency maps (deletion test) | Adebayo et al. (2018), Petsiuk et al. (2018) |