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
from torchvision.ops import box_iou
from tqdm import tqdm

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 20
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
    """Загрузка данных с проверкой"""
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
            torch.FloatTensor(img.transpose(2, 0, 1)), # Исходное изображение в формате (channels, height, width)
            torch.FloatTensor(bbox_norm) # Координаты AABB в формате [x_min, y_min, x_max, y_max]
        )

class BBoxModel(nn.Module):
    def __init__(self, input_size=TARGET_SIZE, num_classes=1):
        super(BBoxModel, self).__init__()
        self.input_size = input_size
        self.grid_size = input_size // 32

        self.backbone = nn.Sequential(
            self._make_conv_block(3, 16),
            self._make_conv_block(16, 32, stride=2),
            self._make_conv_block(32, 64),
            self._make_conv_block(64, 128, stride=2),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512, stride=2),
            self._make_conv_block(512, 1024),
            self._make_conv_block(1024, 512),
        )

        self.detection = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 5 + num_classes, 1),  # 5: [conf, x, y, w, h] + классы
        )

    def _make_conv_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Prediction
        pred = self.detection(features)
        
        # Reshape output
        pred = pred.permute(0, 2, 3, 1)  # [batch, grid, grid, 5+classes]
        
        # Активации
        pred[..., 0] = torch.sigmoid(pred[..., 0])  # Confidence
        pred[..., 1:5] = torch.sigmoid(pred[..., 1:5])  # bbox (x,y,w,h)
        
        return pred

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

def convert_yolo_outputs(yolo_outputs):
    """Конвертирует выход YOLO в нормализованные bounding boxes"""
    batch_size = yolo_outputs.size(0)
    grid_size = yolo_outputs.size(1)
    
    boxes = []
    for b in range(batch_size):
        # Находим grid cell с максимальным confidence
        conf, grid_y, grid_x = torch.max(yolo_outputs[b, ..., 0], dim=0)
        grid_y, grid_x = grid_y.item(), grid_x.item()
        
        # Получаем предсказания
        pred = yolo_outputs[b, grid_y, grid_x]
        x_offset, y_offset, width, height = pred[1:5].sigmoid()
        
        # Конвертируем в абсолютные координаты
        cx = (grid_x + x_offset) / grid_size
        cy = (grid_y + y_offset) / grid_size
        width = width / grid_size
        height = height / grid_size
        
        # Конвертируем в [x_min, y_min, x_max, y_max]
        x_min = cx - width/2
        y_min = cy - height/2
        x_max = cx + width/2
        y_max = cy + height/2
        
        boxes.append(torch.tensor([x_min, y_min, x_max, y_max], device=yolo_outputs.device))
    
    return torch.stack(boxes)

def calculate_batch_iou(pred_boxes, true_boxes):
    """Вычисляет IoU для батча предсказаний"""
    # Приводим к формату [x1, y1, x2, y2]
    pred_boxes = pred_boxes.clamp(0, 1)
    true_boxes = true_boxes / TARGET_SIZE[0]  # Нормализуем
    
    return box_iou(pred_boxes, true_boxes).diag()

def train_model():
    # Загрузка данных
    model = BBoxModel(input_size=TARGET_SIZE[0]).to(DEVICE)

    data = load_data(ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = BBoxDataset(train_data, TARGET_SIZE)
    val_dataset = BBoxDataset(val_data, TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Функция потерь для YOLO
    def yolo_loss(preds, targets):
        # Преобразование targets в YOLO формат
        grid_size = preds.size(1)
        batch_size = preds.size(0)
        
        # Создаем target tensor [batch, grid, grid, 5]
        target_tensor = torch.zeros(batch_size, grid_size, grid_size, 5, device=DEVICE)
        
        for b in range(batch_size):
            x_min, y_min, x_max, y_max = targets[b]
            
            # Нормализованные координаты центра и размеров
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Определяем grid cell
            grid_x = min(int(cx * grid_size), grid_size - 1)
            grid_y = min(int(cy * grid_size), grid_size - 1)
            
            # Заполняем target tensor
            target_tensor[b, grid_y, grid_x, 0] = 1.0  # confidence
            target_tensor[b, grid_y, grid_x, 1] = cx * grid_size - grid_x  # x offset
            target_tensor[b, grid_y, grid_x, 2] = cy * grid_size - grid_y  # y offset
            target_tensor[b, grid_y, grid_x, 3] = width  # width
            target_tensor[b, grid_y, grid_x, 4] = height  # height
        
        # Выделяем предсказания для соответствующих grid cells
        pred_conf = preds[..., 0]
        pred_boxes = preds[..., 1:5]
        
        # Маска для объектов
        obj_mask = target_tensor[..., 0] == 1
        
        # Потери для confidence (бинарная кросс-энтропия)
        loss_conf = nn.functional.binary_cross_entropy(
            pred_conf, target_tensor[..., 0], reduction='none'
        )
        
        # Потери для bounding box (MSE)
        loss_box = nn.functional.mse_loss(
            pred_boxes[obj_mask], target_tensor[..., 1:5][obj_mask], reduction='none'
        ).sum(-1)
        
        # Общие потери
        total_loss = (loss_conf.mean() + loss_box.mean()) / 2
        return total_loss
    
    # Обучение
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Прогресс-бар для обучения
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, targets in train_iter:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE) * TARGET_SIZE[0]  # Денормализация
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE) * TARGET_SIZE[0]
                
                outputs = model(images)
                val_loss += yolo_loss(outputs, targets).item()
                
                # Вычисление IoU
                pred_boxes = convert_yolo_outputs(outputs)
                val_iou += calculate_batch_iou(pred_boxes, targets).mean().item()
        
        # Средние метрики
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_yolo_model.pth')
        scheduler.step(val_loss)

if __name__ == "__main__":
    train_model()