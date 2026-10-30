"""xView2 metric: localization F1, damage F1 and the overall score."""
from src.data.labels import IGNORE

DAMAGE_CLASSES = [1, 2, 3, 4]

# Compute F1 score given true positives, false positives and false negatives
def f1_score(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)

# Compute harmonic mean of a list of scores
def harmonic_mean(scores):
    if min(scores) == 0:
        return 0.0
    return len(scores) / sum(1 / s for s in scores)

# xview2 metric 
class XView2Metric:
    def __init__(self):
        self.reset()

    # set all counters to zero
    def reset(self):
        self.loc = [0, 0, 0]                              # tp, fp, fn
        self.dmg = {c: [0, 0, 0] for c in DAMAGE_CLASSES}  # tp, fp, fn per class

    # add the counts of one batch
    def update(self, pred, target):
        valid = target != IGNORE
        pred = pred[valid]
        target = target[valid]

        # localization: building or not
        pred_building = pred > 0
        true_building = target > 0
        self.loc[0] += (pred_building & true_building).sum().item()
        self.loc[1] += (pred_building & ~true_building).sum().item()
        self.loc[2] += (~pred_building & true_building).sum().item()

        # damage: only on true building pixels
        pred_dmg = pred[true_building]
        true_dmg = target[true_building]
        for c in DAMAGE_CLASSES:
            self.dmg[c][0] += ((pred_dmg == c) & (true_dmg == c)).sum().item()
            self.dmg[c][1] += ((pred_dmg == c) & (true_dmg != c)).sum().item()
            self.dmg[c][2] += ((pred_dmg != c) & (true_dmg == c)).sum().item()

    # compute the final scores
    def compute(self):
        f1_loc = f1_score(self.loc[0], self.loc[1], self.loc[2])

        f1_per_class = {}
        for c in DAMAGE_CLASSES:
            tp, fp, fn = self.dmg[c]
            f1_per_class[c] = f1_score(tp, fp, fn)

        f1_dmg = harmonic_mean(list(f1_per_class.values()))
        score = 0.3 * f1_loc + 0.7 * f1_dmg

        return {"score": score, "f1_loc": f1_loc, "f1_dmg": f1_dmg, "f1_per_class": f1_per_class}