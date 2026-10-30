"""Tests for the train/val/test split files."""
import pandas as pd

SPLIT_DIR = "splits"

# read the image ids of a split file
def load_ids(name):
    df = pd.read_csv(f"{SPLIT_DIR}/{name}.csv")
    return set(df["image_id"])

# no pair can be in more than one split
def test_no_overlap():
    train, validation, test = load_ids("train"), load_ids("val"), load_ids("test")
    assert len(train.intersection(validation)) == 0
    assert len(train.intersection(test)) == 0
    assert len(validation.intersection(test)) == 0

# every pair of the dataset is in one of the splits
def test_all_assigned():
    train, validation, test = load_ids("train"), load_ids("val"), load_ids("test")
    assert len(train.union(validation).union(test)) == 11034

# every disaster event appears in each split
def test_disaster_stratification():
    train = set(pd.read_csv(f"{SPLIT_DIR}/train.csv")["disaster"])
    val = set(pd.read_csv(f"{SPLIT_DIR}/val.csv")["disaster"])
    test = set(pd.read_csv(f"{SPLIT_DIR}/test.csv")["disaster"])
    assert train == val == test
    assert len(train) == 19