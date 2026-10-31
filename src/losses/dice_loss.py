"""Dice loss for multi-class segmentation."""
import torch
import torch.nn as nn

from src.data.labels import IGNORE


class DiceLoss(nn.Module):
    def __init__(self, classes=(1, 2, 3, 4), eps=1.0):
        super().__init__()
        self.classes = classes 
        self.eps = eps

    def forward(self, logits, target):
        probs = logits.softmax(dim=1)  
        valid = target != IGNORE

        dice_scores = []
        for c in self.classes:
            p = probs[:, c][valid]             
            g = (target[valid] == c).float()  

            intersection = (p * g).sum()
            total = p.sum() + g.sum()
            dice = (2 * intersection + self.eps) / (total + self.eps)
            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()