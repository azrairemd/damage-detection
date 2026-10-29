"""Parsing helpers for xBD label files (JSON)."""

import json
from pathlib import Path
from shapely import wkt
from shapely.geometry.base import BaseGeometry

# values for damage classes
BACKGROUND = 0
IGNORE = 255
DAMAGE_CLASSES = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": IGNORE,
}

# metadata fields
METADATA_FIELDS = (
    "disaster",
    "disaster_type",
    "sensor",
    "capture_date",
    "off_nadir_angle",
    "gsd",
    "pan_resolution",
)

# Building class to hold building information
class Building:
    def __init__(self, uid, damage, polygon):
        self.uid = uid
        self.damage = damage
        self.polygon = polygon

    def area(self):
        return self.polygon.area

# load labels from a json file
def load_label(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
# parse each building in the json data
def parse_buildings(label: dict) -> list[Building]:
    buildings = []
    for feature in label["features"]["xy"]:
        props = feature["properties"]
        if props.get("feature_type") != "building":
            continue

        subtype = props.get("subtype")
        if subtype is not None and subtype not in DAMAGE_CLASSES:
            raise ValueError(f"Unknown damage subtype: {subtype!r}")
        damage = DAMAGE_CLASSES[subtype] if subtype is not None else None

        buildings.append(
            Building(uid=props["uid"], damage=damage, polygon=wkt.loads(feature["wkt"]))
        )
    return buildings

# parse metadata from a json file
def parse_metadata(label: dict) -> dict:
    meta = label["metadata"]
    return {field: meta.get(field) for field in METADATA_FIELDS}