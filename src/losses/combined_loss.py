"""Weighted combination of focal loss and dice loss."""
import torch.nn as nn

from src.losses.focal_loss import FocalLoss
from src.losses.dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, lam=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.lam = lam

    def forward(self, logits, target):
        focal = self.focal(logits, target)
        dice = self.dice(logits, target)
        return self.lam * focal + (1 - self.lam) * dice