import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import random
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")
TEST_DIRS=[
        './dataset/test/altai',
        './dataset/test/begickaya',
        './dataset/test/promlit',
        './dataset/test/ruzhimmash',
        './dataset/test/tihvin'
    ]

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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        
        combined = torch.cat([max_out, mean_out, std_out], dim=1)
        att_map = self.conv(combined)
        return x * self.sigmoid(att_map)

class BBoxModel(nn.Module):
    def __init__(self, input_size=TARGET_SIZE, num_classes=1):
        super(BBoxModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # Блоки внимания после layer2 и layer3
        self.att1 = SpatialAttention(kernel_size=5)
        self.att2 = SpatialAttention(kernel_size=9)
        
        # Адаптивный пулинг
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.regressor = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.att1(x)

        x = self.layer3(x)
        x = self.att2(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
    
def calculate_iou(pred_boxes, true_boxes):
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    true_boxes = torch.clamp(true_boxes, 0, 1)
    return box_iou(pred_boxes, true_boxes).diag().mean().item()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        mse_loss = self.mse(pred, target).mean(dim=1)
        
        def compute_iou(box1, box2):
            
            x_left = torch.max(box1[0], box2[0])
            y_top = torch.max(box1[1], box2[1])
            x_right = torch.min(box1[2], box2[2])
            y_bottom = torch.min(box1[3], box2[3])
            
            intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
            
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            union_area = box1_area + box2_area - intersection_area + 1e-6
            
            return intersection_area / union_area

        iou = torch.stack([compute_iou(p, t) for p, t in zip(pred, target)])

        iou_loss = 1.0 - iou
        
        # Комбинированная потеря
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * iou_loss
        
        return combined_loss.mean()

def train_model():
    # Загрузка данных
    data = load_data(ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = BBoxDataset(train_data, TARGET_SIZE, augment=True)
    val_dataset = BBoxDataset(val_data, TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Инициализация модели
    model = BBoxModel().to(DEVICE)
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_iou += calculate_iou(outputs, targets) * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_iou += calculate_iou(outputs, targets) * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        scheduler.step(val_loss)
        
        # Вывод информации о текущем learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")
    
    # Построение графиков обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss during training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU during training')
    plt.legend()

    plt.show()
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def visualize_predictions(model, test_images, target_size, device, n=5):
    model.eval()
    selected_images = random.sample(test_images, min(n, len(test_images)))
    for img_path in selected_images:
        try:
            # Загрузка изображения
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            orig_h, orig_w = img.shape[:2]
            display_img = img.copy()
            
            # Предобработка изображения
            img_proc = preprocess_image(img, target_size)
            img_tensor = torch.FloatTensor(img_proc.transpose(2, 0, 1)).unsqueeze(0).to(device)
            
            # Получение предсказания
            with torch.no_grad():
                pred = model(img_tensor)
                pred_np = pred.cpu().numpy().squeeze()
            
            # Проверка формы выходных данных
            print(f"Debug - pred_np shape: {pred_np.shape}")  # Отладочная информация
            
            if pred_np.size == 4:  # Если это плоский массив из 4 элементов
                pred_bbox = pred_np
            elif pred_np.shape[-1] == 4:  # Если это массив с последней размерностью 4
                pred_bbox = pred_np[0] if len(pred_np.shape) > 1 else pred_np
            else:
                print(f"Error: Unexpected prediction shape {pred_np.shape} for image {img_path}")
                continue
            
            # Преобразование координат
            try:
                x_min = int(pred_np[0] * orig_w)
                y_min = int(pred_np[1] * orig_h)
                x_max = int(pred_np[2] * orig_w)
                y_max = int(pred_np[3] * orig_h)
            except Exception as e:
                print(f"Error converting coordinates for image {img_path}: {str(e)}")
                continue
            
            # Проверка валидности bounding box
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(orig_w, x_max), min(orig_h, y_max)

            if x_min >= x_max or y_min >= y_max:
                print(f"Warning: Invalid bbox coordinates for image {img_path}")
                continue
            
            # Отрисовка и отображение
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(os.path.basename(img_path))
            plt.show()
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

def collect_test_images(test_dirs):
    test_images = []
    for directory in test_dirs:
        test_images.extend(glob.glob(os.path.join(directory, '*.jpg')))
        test_images.extend(glob.glob(os.path.join(directory, '*.png')))
    return test_images

if __name__ == "__main__":
    # Обучение модели
    trained_model = train_model().to(DEVICE)
    
    # Сбор тестовых изображений
    test_images = collect_test_images(TEST_DIRS)
    print(f"Found {len(test_images)} test images")
    
    # Визуализация результатов
    visualize_predictions(trained_model, test_images, TARGET_SIZE, DEVICE, n=10)