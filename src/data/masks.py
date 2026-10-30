"""Rasterize building polygons into segmentation masks."""
import cv2
import numpy as np

from src.data.labels import BACKGROUND, IGNORE

# drawing order
DRAW_ORDER = [IGNORE, 1, 2, 3, 4]


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

# draw building polygons into a mask 
def make_mask(buildings, height=1024, width=1024):
    mask = np.full((height, width), BACKGROUND, dtype=np.uint8)

    for value in DRAW_ORDER:
        for b in buildings:
            if b.damage == value:
                draw_polygon(mask, b.polygon, value)

    return mask
