# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# =============================
# Dataset
# =============================
class AnimalCLEFDataset(Dataset):
    def __init__(self, root, split="database", transform=None):
        self.root = root.rstrip('/')
        meta = pd.read_csv(f"{self.root}/metadata.csv")
        sel = meta[meta['path'].str.contains(f"/{split}/")].reset_index(drop=True)
        if sel.empty:
            raise ValueError(f"No entries for split '{split}'")

        self.paths = sel['path'].tolist()
        self.image_ids = sel['image_id'].tolist()

        if split == 'database':
            #  Use individual identity,  
            ids = sel['identity'].astype(str)

            #  Build mapping from identity string → label index
            self.id2idx = {iid: i for i, iid in enumerate(sorted(ids.unique()))}

            #  Map each sample's identity to its label
            self.labels = ids.map(self.id2idx).tolist()

            # Safety check
            num_classes = len(self.id2idx)
            assert all(0 <= label < num_classes for label in self.labels), "Invalid labels found"
        else:
            self.labels = [-1] * len(sel)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(f"{self.root}/{self.paths[i]}").convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]