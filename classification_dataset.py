import os
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

load_dotenv()
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")

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
    
class FullImageProducerDataset:
    def __init__(self, annotations_path, transform=None):
        self.annotations = pd.read_csv(annotations_path)
        self.transform = transform
        self.class_to_idx = {
            'altai': 0, 'begickaya': 1, 'promlit': 2, 'ruzhimmash': 3, 'tihvin': 4
        }
        self.images_path = TRAIN_IMAGES_PATH

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_name = row['image'].split('/')[-1]
        image_path = os.path.join(self.images_path, image_name)
        producer = row['producer']
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        label = self.class_to_idx[producer]
        return img, torch.tensor(label)