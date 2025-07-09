import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from loaders import load_data
from sklearn.model_selection import train_test_split
from bboxdataset import BBoxDataset
from torch.utils.data import DataLoader
from bboxmodel import BBoxModel, CenterIoULoss, calculate_iou
from tqdm import tqdm
from torchvision.ops import box_iou

load_dotenv()

TARGET_SIZE = (320, 320)
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_ANNOTATIONS_PATH = os.getenv("TRAIN_ANNOTATIONS_PATH")
TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")
TRAIN_IMAGES_PATH = os.getenv("TRAIN_IMAGES_PATH")
TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")

def train_model():
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