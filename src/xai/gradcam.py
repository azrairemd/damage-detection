import torch
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import random

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

# GradCAM requires the target layer (decoder's second block)
def get_target_layer(model):
    if hasattr(model, "unet"):
        return model.unet.decoder.blocks[1]
    return model.decoder.blocks[1]

# Compute GradCAM heatmap for a given input and category
def compute_gradcam(cam, wrapper, x, category):
    with torch.no_grad():
        logits = wrapper(x)
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    target = make_target(logits, category)
    heatmap = cam(input_tensor=x, targets=[target])[0]
    return heatmap, pred

# Compute the heatmap value on buildings and the area of buildings in the mask
def heat_on_buildings(heatmap, mask):
    building = (mask > 0) & (mask != 255)
    area = building.mean()
    heat = heatmap[building].sum() / heatmap.sum()
    return heat, area

# Measure GradCAM heatmap values for different disaster types and categories in the dataset
def measure_groups(cam, wrapper, dataset, groups, device, n_images=20, min_pixels=1000):
    results = {}
    for group in groups:
        results[group] = []

    types = dataset.pairs["disaster_type"]
    indices = list(range(len(dataset)))
    random.seed(0)
    random.shuffle(indices)

    for i in indices:
        t = types.iloc[i]

        wanted = []
        for (group_type, c) in groups:
            if group_type == t and len(results[(group_type, c)]) < n_images:
                wanted.append(c)
        if len(wanted) == 0:
            continue

        pre, post, mask = dataset[i]
        x = torch.cat([pre, post], dim=0).unsqueeze(0).to(device)
        mask = mask.numpy()

        with torch.no_grad():
            pred = wrapper(x).argmax(dim=1)[0]

        for c in wanted:
            if (pred == c).sum().item() < min_pixels:
                continue
            heatmap, _ = compute_gradcam(cam, wrapper, x, c)
            heat, area = heat_on_buildings(heatmap, mask)
            results[(t, c)].append((i, heat, area))

    return results