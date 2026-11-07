"""Evaluation of trained models on the test set."""
import torch
from torch.utils.data import DataLoader
from src.metrics.xview2 import XView2Metric

NUM_CLASSES = 5
IGNORE_INDEX = 255

def load_model(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def update_confusion_matrix(confusion, pred, target):
    valid = target != IGNORE_INDEX
    true = target[valid]
    pred = pred[valid]

    index = true * NUM_CLASSES + pred
    counts = torch.bincount(index, minlength=NUM_CLASSES ** 2)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()

def evaluate(model, dataset, device,num_workers=4):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    model.eval()

    overall = XView2Metric()
    per_type = {}
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for i, (pre, post, mask) in enumerate(data_loader):
            pre = pre.to(device)
            post = post.to(device)
            mask = mask.to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(pre, post)
            pred = logits.argmax(dim=1)

            overall.update(pred, mask)

            disaster_type = dataset.pairs["disaster_type"].iloc[i]
            if disaster_type not in per_type:
                per_type[disaster_type] = XView2Metric()
            per_type[disaster_type].update(pred, mask)

            update_confusion_matrix(confusion, pred, mask)

    scores_per_type = {t: m.compute_f1() for t, m in per_type.items()}
    return overall.compute_f1(), scores_per_type, confusion
        
