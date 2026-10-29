"""Build a per-pair summary table of the xBD dataset."""
from src.data.labels import load_label, parse_buildings, parse_metadata, DAMAGE_CLASSES
from src.data.xbd import get_label_path, FOLDERS, list_image_ids

import pandas as pd



def summarize_pair(root, folder, image_id):
    post_label = load_label(get_label_path(root, folder, image_id, "post"))
    buildings = parse_buildings(post_label)

    row = {
        "image_id": image_id,
        "folder": folder,
        "num_buildings": len(buildings),
    }
    row.update(parse_metadata(post_label)) # from src/data/labels.py parse_metadata function (its properties are disaster, disaster_type, sensor, capture_date, off_nadir_angle, gsd, pan_resolution)

    pre_label = load_label(get_label_path(root, folder, image_id, "pre"))
    row["num_buildings_pre"] = len(parse_buildings(pre_label))

    for name, value in DAMAGE_CLASSES.items():
        same_class = [b for b in buildings if b.damage == value] # buildings with the same damage class
        row["num_" + name] = len(same_class)
        row["area_" + name] = sum(b.area() for b in same_class)
        
    return row

def build_metadata_table(root):
    rows = []
    for folder in FOLDERS:
        for image_id in list_image_ids(root, folder):
            rows.append(summarize_pair(root, folder, image_id))
    return pd.DataFrame(rows)