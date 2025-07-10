import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from loaders import load_data
from sklearn.model_selection import train_test_split
from bbox_dataset import BBoxDataset
from torch.utils.data import DataLoader
from bbox_model import BBoxModel, CenterIoULoss
from tqdm import tqdm
from torchvision.ops import box_iou

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 120
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_ANNOTATIONS_PATH = os.getenv("TRAIN_ANNOTATIONS_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")
TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")

def calculate_iou(pred_boxes, true_boxes):
    """
    Вычисляет средний IoU для батча.
    Улучшенная и оптимизированная версия.
    """
    # Гарантируем валидные значения
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    true_boxes = torch.clamp(true_boxes, 0, 1)
    
    # Разделяем координаты
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    true_x1, true_y1, true_x2, true_y2 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]
    
    # Пересечение
    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)
    
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_width * inter_height
    
    # Площади
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    
    # Объединение
    union = pred_area + true_area - intersection + 1e-6
    
    # IoU для каждого элемента батча
    iou = intersection / union
    
    return iou.mean().item()

def train_model(visualize=False):
    data = load_data(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGES_PATH)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = BBoxDataset(train_data, TARGET_SIZE, augment=True)
    val_dataset = BBoxDataset(val_data, TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BBoxModel().to(DEVICE)
    criterion = CenterIoULoss(iou_weight=0.3)
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

        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")
    
    if visualize:
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
    
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def calculate_test_iou(model, test_data, device):
    """Вычисление средней IoU для тестового набора"""
    model.eval()
    ious = []
    test_loader = DataLoader(BBoxDataset(test_data, TARGET_SIZE), batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Calculating test IoU"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            batch_iou = box_iou(outputs, targets).diag()
            ious.extend(batch_iou.cpu().numpy())
    
    return np.mean(ious)