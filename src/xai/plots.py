import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pytorch_grad_cam.utils.image import show_cam_on_image

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

MASK_COLORS = ListedColormap(["black", "green", "yellow", "orange", "red"])
CLASS_NAMES = ["background", "no damage", "minor", "major", "destroyed"]


# Convert a normalized image tensor (3, H, W) back to a displayable image (H, W, 3)
def to_image(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1).astype(np.float32)

# Plot one example: pre, post, true mask, prediction and Grad-CAM heatmap
def plot_example(pre, post, mask, pred, heatmap, title, save_path=None):
    pre_img = to_image(pre)
    post_img = to_image(post)

    true_mask = mask.copy()
    true_mask[true_mask == 255] = 0
    overlay = show_cam_on_image(post_img, heatmap, use_rgb=True)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre")
    axes[1].imshow(post_img)
    axes[1].set_title("Post")
    axes[2].imshow(true_mask, cmap=MASK_COLORS, vmin=0, vmax=4)
    axes[2].set_title("True mask")
    axes[3].imshow(pred, cmap=MASK_COLORS, vmin=0, vmax=4)
    axes[3].set_title("Prediction")
    axes[4].imshow(overlay)
    axes[4].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

# Compare several models on the same image: one row per model
def plot_comparison(pre, post, mask, preds, heatmaps, names, title, save_path=None):
    pre_img = to_image(pre)
    post_img = to_image(post)

    true_mask = mask.copy()
    true_mask[true_mask == 255] = 0

    n = len(names)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5.5 * n))

    for k in range(n):
        overlay = show_cam_on_image(post_img, heatmaps[k], use_rgb=True)

        axes[k, 0].imshow(pre_img)
        axes[k, 0].set_title("Pre")
        axes[k, 1].imshow(post_img)
        axes[k, 1].set_title("Post")
        axes[k, 2].imshow(true_mask, cmap=MASK_COLORS, vmin=0, vmax=4)
        axes[k, 2].set_title("True mask")
        axes[k, 3].imshow(preds[k], cmap=MASK_COLORS, vmin=0, vmax=4)
        axes[k, 3].set_title(names[k] + ": prediction")
        axes[k, 4].imshow(overlay)
        axes[k, 4].set_title(names[k] + ": Grad-CAM")
        for ax in axes[k]:
            ax.axis("off")

    fig.suptitle(title, fontsize=16)
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig