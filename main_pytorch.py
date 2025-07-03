import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")

def rotated_rect_to_aabb(center_x, center_y, width, height, rotation_degrees):
    """Преобразование rotated rectangle в axis-aligned bounding box (AABB)"""
    # Проверка на нулевой угол
    if abs(rotation_degrees) < 1e-6:
        return (
            center_x - width/2,
            center_y - height/2,
            center_x + width/2,
            center_y + height/2
        )
    
    rotation_rad = np.radians(rotation_degrees)

    corners = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ])
    rot_mat = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])
    rotated_corners = np.dot(corners, rot_mat)

    # Смещение центра
    offset_x = (width/2 * np.cos(rotation_rad) - height/2 * np.sin(rotation_rad)) - width/2
    offset_y = (width/2 * np.sin(rotation_rad) + height/2 * np.cos(rotation_rad)) - height/2
    true_center_x = center_x + offset_x
    true_center_y = center_y + offset_y

    # Перерасчёт координат углов
    rotated_corners[:, 0] += true_center_x
    rotated_corners[:, 1] += true_center_y

    x_min, y_min = np.min(rotated_corners, axis=0)
    x_max, y_max = np.max(rotated_corners, axis=0)
    return x_min, y_min, x_max, y_max

def parse_annotation(bbox_dict, original_width, original_height):
    """Извлечение координат AABB из аннотации"""
    width = bbox_dict['width'] / 100.0 * original_width
    height = bbox_dict['height'] / 100.0 * original_height
    cx = bbox_dict['x'] / 100.0 * original_width + width/2
    cy = bbox_dict['y'] / 100.0 * original_height + height/2
    rotation = bbox_dict['rotation']
    return rotated_rect_to_aabb(cx, cy, width, height, rotation)

def load_data(csv_path, img_base_dir):
    """Загрузка данных с проверкой путей"""
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        try:
            bbox_list = json.loads(row['code_bbox'].replace("'", '"'))
            bbox_dict = bbox_list[0]
            img_name = os.path.basename(row['image'])
            img_path = os.path.join(img_base_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                continue
                
            h, w = img.shape[:2]
            if h != bbox_dict['original_height'] or w != bbox_dict['original_width']:
                print(f"Warning: Size mismatch for {img_path}")
                continue
                
            x_min, y_min, x_max, y_max = parse_annotation(bbox_dict, w, h)
            data.append({
                'img_path': img_path,
                'orig_width': w,
                'orig_height': h,
                'bbox': [x_min, y_min, x_max, y_max]
            })
        except Exception as e:
            print(f"Error processing row {row}: {str(e)}")
    return data

def preprocess_image(img, target_size):
    """Предобработка изображения: resize и нормализация"""
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

class BBoxDataset(Dataset):
    def __init__(self, data, target_size):
        self.data = data
        self.target_size = target_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.imread(sample['img_path'])
        img = preprocess_image(img, self.target_size)
        
        bbox = sample['bbox']
        orig_w, orig_h = sample['orig_width'], sample['orig_height']
        
        # Нормализация координат
        bbox_norm = [
            bbox[0] / orig_w,
            bbox[1] / orig_h,
            bbox[2] / orig_w,
            bbox[3] / orig_h
        ]
        
        return (
            torch.FloatTensor(img.transpose(2, 0, 1)),  # (C, H, W)
            torch.FloatTensor(bbox_norm)                 # [x_min, y_min, x_max, y_max]
        )

# Модель PyTorch
class BBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

# Метрики и функции обучения
def compute_iou(box1, box2):
    """Вычисление IoU для numpy массивов"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def iou_pytorch(preds, targets):
    """Метрика IoU для PyTorch тензоров"""
    pred_x1, pred_y1, pred_x2, pred_y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    true_x1, true_y1, true_x2, true_y2 = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
    
    # Корректировка "перевернутых" bbox
    pred_x1, pred_x2 = torch.min(pred_x1, pred_x2), torch.max(pred_x1, pred_x2)
    pred_y1, pred_y2 = torch.min(pred_y1, pred_y2), torch.max(pred_y1, pred_y2)
    
    intersect_x1 = torch.max(pred_x1, true_x1)
    intersect_y1 = torch.max(pred_y1, true_y1)
    intersect_x2 = torch.min(pred_x2, true_x2)
    intersect_y2 = torch.min(pred_y2, true_y2)
    
    intersect = (intersect_x2 - intersect_x1).clamp(0) * (intersect_y2 - intersect_y1).clamp(0)
    area_pred = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    area_true = (true_x2 - true_x1) * (true_y2 - true_y1)
    
    return intersect / (area_pred + area_true - intersect + 1e-6)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += iou_pytorch(outputs, targets).mean().item()
    
    return total_loss / len(loader), total_iou / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            
            total_loss += criterion(outputs, targets).item()
            total_iou += iou_pytorch(outputs, targets).mean().item()
    
    return total_loss / len(loader), total_iou / len(loader)


def visualize_predictions(model, test_images, target_size, device, n=5):
    model.eval()
    for img_path in test_images[:n]:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        orig_h, orig_w = img.shape[:2]
        display_img = img.copy()
        
        img_proc = preprocess_image(img, target_size)
        img_tensor = torch.FloatTensor(img_proc.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_bbox_norm = model(img_tensor).cpu().numpy()[0]
        
        x_min = int(pred_bbox_norm[0] * orig_w)
        y_min = int(pred_bbox_norm[1] * orig_h)
        x_max = int(pred_bbox_norm[2] * orig_w)
        y_max = int(pred_bbox_norm[3] * orig_h)
        
        cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(os.path.basename(img_path))
        plt.show()

if __name__ == "__main__":
    # Загрузка данных
    data = load_data(ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = BBoxDataset(train_data, TARGET_SIZE)
    val_dataset = BBoxDataset(val_data, TARGET_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*2, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Инициализация модели
    model = BBoxModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Обучение
    best_iou = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_iou = validate(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # Сохранение лучшей модели
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Модель сохранена (лучший IoU: {best_iou:.4f})")
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    
    # Тестирование на нескольких примерах
    test_dirs = [
        './dataset/test/altai',
        './dataset/test/begickaya',
        './dataset/test/promlit',
        './dataset/test/ruzhimmash',
        './dataset/test/tihvin'
    ]

    test_images = []
    for dir_path in test_dirs:
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(dir_path, filename))

    visualize_predictions(model, test_images, TARGET_SIZE, DEVICE, n=20)