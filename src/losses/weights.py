"""Class weights for the focal loss."""
import numpy as np
import torch

def compute_class_weights(pixel_counts):
    pixel_counts = np.array(pixel_counts)
    frequencies = pixel_counts / pixel_counts.sum()
    weights = 1/ np.sqrt(frequencies)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
