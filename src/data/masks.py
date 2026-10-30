"""Functions to create segmentation masks from building polygons."""
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from src.data.labels import BACKGROUND, IGNORE, load_label, parse_buildings
from src.data.xbd import get_label_path

# drawing order
DRAW_ORDER = [IGNORE, 1, 2, 3, 4]

# get mask path
def get_mask_path(mask_root, folder, image_id):
    return Path(mask_root) / folder / f"{image_id}.png"

# draw a single polygon into a mask
def draw_polygon(mask, polygon, value):
    height, width = mask.shape

    if polygon.geom_type == "MultiPolygon":
        parts = list(polygon.geoms)
    else:
        parts = [polygon]

    for p in parts:
        points = np.array(p.exterior.coords)
        points = np.clip(points, 0, [width - 1, height - 1])
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], color=int(value))

# create a mask
def make_mask(buildings, height=1024, width=1024):
    mask = np.full((height, width), BACKGROUND, dtype=np.uint8)

    for value in DRAW_ORDER:
        for b in buildings:
            if b.damage == value:
                draw_polygon(mask, b.polygon, value)

    return mask

# save the mask
def save_masks(root, df, mask_root):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = load_label(get_label_path(root, row["folder"], row["image_id"], "post"))
        mask = make_mask(parse_buildings(label))

        path = get_mask_path(mask_root, row["folder"], row["image_id"])
        path.parent.mkdir(parents=True, exist_ok=True)