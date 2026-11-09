import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

class SiameseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pre = x[:, 0:3]
        post = x[:, 3:6]
        return self.model(pre, post) # return the prediction of the model for the pre and post images

def make_target(logits, category):
    pred = logits.argmax(dim=1)[0]
    mask = (pred == category).float().cpu().numpy()
    return SemanticSegmentationTarget(category, mask)