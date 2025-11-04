import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, preprocess=None):
        """
        root_dir: Pfad zu den Bildern
        captions_file: CSV oder TXT mit 'filename' und 'caption'
        preprocess: CLIP-Preprocessing (von openai/clip oder transformers)
        """
        self.root_dir = root_dir
        self.captions = pd.read_csv(captions_file, sep=None, engine="python")
        self.preprocess = preprocess or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        caption = row['caption']
        return image, caption