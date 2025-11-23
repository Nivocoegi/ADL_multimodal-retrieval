"""

Class to load data into a PyTorch Dataset.

@author: vognic01

"""


import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, preprocess=None):
        self.root_dir = root_dir

        # Accept CSV path or DataFrame
        if isinstance(captions_file, str):
            self.captions = pd.read_csv(captions_file)
        elif isinstance(captions_file, pd.DataFrame):
            self.captions = captions_file.reset_index(drop=True)
        else:
            raise ValueError("captions_file must be a path or DataFrame")

        self.preprocess = preprocess or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        return image, row["caption"]