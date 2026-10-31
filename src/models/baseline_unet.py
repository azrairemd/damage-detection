"""Baseline model: a U-Net that uses only the post-disaster image."""
import segmenation_models_pytorch as smp
import torch.nn as nn

class BaselineUNet(nn.Module):
    def __init__(self, encoder="resnet34", num_classes=5, pretrained=True):
        super().__init__()
        self.unet = smp.Unet(
            encoder = encoder,
            encoder_weights = "imagenet" ,
            in_channels = 3,
            classes = num_classes
        )

    def forward(self, pre, post):
        return self.unet(post)
    