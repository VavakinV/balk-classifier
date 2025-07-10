import random
import cv2
import torch
from torch.utils.data import Dataset
from preprocessimage import preprocess_image

class BBoxDataset(Dataset):
    def __init__(self, data, target_size, augment=False):
        self.data = data
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.imread(sample['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bbox = sample['bbox']
        orig_w, orig_h = sample['orig_width'], sample['orig_height']
        
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox = [
                    orig_w - bbox[2],
                    bbox[1],
                    orig_w - bbox[0],
                    bbox[3]
                ]
            
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = preprocess_image(img, self.target_size)

        bbox_norm = [
            bbox[0] / orig_w,
            bbox[1] / orig_h,
            bbox[2] / orig_w,
            bbox[3] / orig_h
        ]
        
        return (
            torch.FloatTensor(img.transpose(2, 0, 1)), # Исходное изображение в формате (channels, height, width)
            torch.FloatTensor(bbox_norm) # Координаты AABB в формате [x_min, y_min, x_max, y_max]
        )