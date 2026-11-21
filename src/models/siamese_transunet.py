import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

from src.models.transunet_encoder import TransUNetEncoder

class SiameseTransUNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.encoder = TransUNetEncoder(pretrained=pretrained)

        encoder_channels = self.encoder.out_channels
        fused_channels = [2 * c for c in encoder_channels]

        decoder_channels = [128, 64, 32, 16]
        self.decoder = UnetDecoder(encoder_channels=fused_channels,
                                   decoder_channels=decoder_channels,
                                   n_blocks=4)
        self.head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=num_classes)

    def forward(self, pre, post):
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        fused_features = []
        for i in range(len(pre_features)):
            fused = torch.cat([pre_features[i], post_features[i]], dim=1)
            fused_features.append(fused)

        decoder_output = self.decoder(fused_features)
        return self.head(decoder_output)