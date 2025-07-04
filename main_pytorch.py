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
import torch.nn.functional as F
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
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att_map = self.conv(combined)
        return x * self.sigmoid(att_map)
    
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
        
    def forward(self, features):
        pyramid_features = []
        last_feature = None
        
        for idx, feature in enumerate(reversed(features)):
            lateral = self.lateral_convs[len(features)-1-idx](feature)
            
            if last_feature is not None:
                size = lateral.shape[-2:]
                upsampled = F.interpolate(last_feature, size=size, mode='nearest')
                lateral = lateral + upsampled
            
            output = self.output_convs[len(features)-1-idx](lateral)
            pyramid_features.append(output)
            last_feature = output
        
        return list(reversed(pyramid_features))

class BBoxModel(nn.Module):
    def __init__(self, input_size=TARGET_SIZE, num_classes=1):
        super(BBoxModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Извлекаем промежуточные слои для FPN
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # Разрешение: 80x80 (для 320x320)
        self.layer2 = resnet.layer2  # Разрешение: 40x40
        self.layer3 = resnet.layer3  # Разрешение: 20x20
        
        # Инициализация FPN
        self.fpn = FPN(
            in_channels_list=[64, 128, 256],  # Каналы layer1, layer2, layer3
            out_channels=256
        )
        
        # Блоки пространственного внимания для каждого уровня FPN
        self.attentions = nn.ModuleList([
            SpatialAttention(kernel_size=7),
            SpatialAttention(kernel_size=5),
            SpatialAttention(kernel_size=3)
        ])
        
        # Объединяем признаки разных уровней
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256*3, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Регрессор
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
        # Проход через начальные слои
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # Извлекаем признаки разных уровней
        f1 = self.layer1(x)   # 64 каналов, 80x80
        f2 = self.layer2(f1)  # 128 каналов, 40x40
        f3 = self.layer3(f2)  # 256 каналов, 20x20
        
        # Строим FPN
        features = [f1, f2, f3]
        pyramid_features = self.fpn(features)
        
        # Применяем пространственное внимание к каждому уровню
        attended_features = []
        for feat, att in zip(pyramid_features, self.attentions):
            attended_features.append(att(feat))
        
        # Апсемплируем все признаки до максимального размера
        target_size = attended_features[0].shape[-2:]
        upsampled_features = []
        
        for feat in attended_features:
            if feat.size()[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled_features.append(feat)
        
        # Объединяем признаки
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # Регрессор
        x = fused.view(fused.size(0), -1)
        return self.regressor(x)
    
def calculate_iou(pred_boxes, true_boxes):
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    true_boxes = torch.clamp(true_boxes, 0, 1)
    return box_iou(pred_boxes, true_boxes).diag().mean().item()

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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

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
    for img_path in test_images[:n]:
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