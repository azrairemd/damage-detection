"""Basic I/O helpers for xBD dataset."""
from pathlib import Path
import numpy as np
from PIL import Image

FOLDERS = ("tier1", "tier3", "test", "hold")

# get image path
def get_image_path(root: str | Path, folder: str, image_id: str, phase: str) -> Path:
    return Path(root) / folder / "images" / f"{image_id}_{phase}_disaster.png"

# get label path
def get_label_path(root: str | Path, folder: str, image_id: str, phase: str) -> Path:
    return Path(root) / folder / "labels" / f"{image_id}_{phase}_disaster.json"

# load image as numpy array
def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))

# load pre post pair
def load_pair(root: str | Path, folder: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    pre = load_image(get_image_path(root, folder, image_id, "pre"))
    post = load_image(get_image_path(root, folder, image_id, "post"))
    return pre, post

# load image_ids from a folder
def list_image_ids(root: str | Path, folder: str) -> list[str]:
    image_dir = Path(root) / folder / "images"
    ids = {p.name.replace("_pre_disaster.png", "") for p in image_dir.glob("*_pre_disaster.png")}
    return sorted(ids)