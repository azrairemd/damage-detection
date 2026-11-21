import torch.nn as nn
import timm

class TransUNetEncoder(nn.Module):
    def __init__(self, pretrained=True, img_size=512):
        super().__init__()
        self.vit = timm.create_model("vit_base_r50_s16_384.orig_in21k_ft_in1k",
                                     pretrained=pretrained, img_size=img_size,
                                     dynamic_img_size=True)
        self.out_channels = [3, 64, 256, 512, 768]

    def forward(self, x):
        cnn = self.vit.patch_embed.backbone
        features = [x]

        # cnn part
        h = cnn.stem.conv(x)
        h = cnn.stem.norm(h)
        features.append(h)     
        
        h = cnn.stem.pool(h)
        h = cnn.stages[0](h)
        features.append(h) 
        
        h = cnn.stages[1](h)
        features.append(h)    
        
        h = cnn.stages[2](h)                     
        h = self.vit.patch_embed.proj(h)        

        # transformer part
        B, C, H, W = h.shape
        
        tokens = h.flatten(2).transpose(1, 2)
        
        tokens = self.vit._pos_embed(tokens.reshape(B, H, W, C))
        
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)
        tokens = tokens[:, self.vit.num_prefix_tokens:]
        h = tokens.transpose(1, 2).reshape(B, C, H, W)
        features.append(h)                     

        return features