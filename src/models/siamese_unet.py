"""Siamese U-Net: a shared encoder for the pre- and post-disaster images, feature fusion at every scale and a U-Net decoder."""
import torch
import torch.nn as nn

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class SiameseUNet(nn.Module):
    def __init__(self, encoder="resnet34", num_classes=5, fusion="concat"):
        super().__init__()
        self.fusion = fusion

        self.encoder = get_encoder(encoder, in_channels=3, depth=5, weights="imagenet")
        encoder_channels = self.encoder.out_channels

        if fusion == "concat":
            fused_channels = [2 * c for c in encoder_channels]
        elif fusion == "diff":
            fused_channels = encoder_channels

        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = UnetDecoder(encoder_channels=fused_channels, decoder_channels=decoder_channels)
        self.head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=num_classes)


    def forward(self, pre, post):
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        fused_features = []
        for i in range(len(pre_features)):
            f_pre = pre_features[i]
            f_post = post_features[i]

            if self.fusion == "concat":
                fused = torch.cat([f_pre, f_post], dim=1)
                fused_features.append(fused)
            else:
                fused = torch.abs(f_pre - f_post)
                fused_features.append(fused)

        decoder_output = self.decoder(fused_features)
        return self.head(decoder_output)