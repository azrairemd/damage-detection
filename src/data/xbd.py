"""Basic I/O helpers for XBD dataset."""
from pathlib import Path
import numpy as np
from PIL import Image

SPLITS = ("tier1", "tier3", "test", "hold")

# get image path
def get_image_path(root: str | Path, split: str, image_id: str, phase: str) -> Path:
    return Path(root) / split / "images" / f"{image_id}_{phase}_disaster.png"

# load image as numpy array
def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))

# load pre post pair
def load_pair(root: str | Path, split: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    pre = load_image(get_image_path(root, split, image_id, "pre"))
    post = load_image(get_image_path(root, split, image_id, "post"))
    return pre, post

# load image_ids from a split
def list_image_ids(root: str | Path, split: str) -> list[str]:
    image_dir = Path(root) / split / "images"
    ids = {p.name.replace("_pre_disaster.png", "") for p in image_dir.glob("*_pre_disaster.png")}
    return sorted(ids)