import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class ProducerDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.class_to_idx = {
            'altai': 0, 'begickaya': 1, 'promlit': 2, 'ruzhimmash': 3, 'tihvin': 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        producer = row['producer']
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        label = self.class_to_idx[producer]
        return img, torch.tensor(label)