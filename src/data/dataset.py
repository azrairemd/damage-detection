"""PyTorch dataset for xBD pre/post image pairs and damage masks."""
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
from src.data.masks import get_mask_path
from src.data.xbd import load_pair

# ImageNet mean and standard deviation (the encoder was pretrained with them)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class XBDDataset(Dataset):
    def __init__(self, split_csv, xbd_root, mask_root, train=True, crop_size=512):
        self.pairs = pd.read_csv(split_csv)
        self.xbd_root = xbd_root
        self.mask_root = mask_root
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        row = self.pairs.iloc[i]
        folder = row["folder"]
        image_id = row["image_id"]

        pre, post = load_pair(self.xbd_root, folder, image_id)

        mask_path = get_mask_path(self.mask_root, folder, image_id)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # augmentation for training
        if self.train:
            # random crop
            size = self.crop_size
            height, width = mask.shape
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)

            pre = pre[top:top + size, left:left + size]
            post = post[top:top + size, left:left + size]
            mask = mask[top:top + size, left:left + size]

            # random reflection and rotation
            if random.random() < 0.5:
                pre = pre[:, ::-1]
                post = post[:, ::-1]
                mask = mask[:, ::-1]

            if random.random() < 0.5:
                pre = pre[::-1, :]
                post = post[::-1, :]
                mask = mask[::-1, :]

            k = random.randint(0, 3)
            pre = np.rot90(pre, k)
            post = np.rot90(post, k)
            mask = np.rot90(mask, k)

        pre = self.to_tensor(pre)
        post = self.to_tensor(post)
        mask = torch.from_numpy(mask.copy()).long()

        return pre, post, mask

    # normalize an image and convert it to a tensor of shape (3, H, W)
    def to_tensor(self, image):
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image.copy())