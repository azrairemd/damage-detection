"""Focal loss for multi-class segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import IGNORE

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    # forward function for focal loss: to compute the loss between the predicted logits and the target labels
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")

        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce

        if self.alpha is not None:
            safe_target = target.clone()
            safe_target[target == IGNORE] = 0
            loss = loss * self.alpha.to(logits.device)[safe_target]

        valid = target != IGNORE
        if valid.sum() == 0:
            return loss.sum() * 0
        return loss[valid].mean()