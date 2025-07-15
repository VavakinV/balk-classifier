import os
import torch
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from classification_model import ProducerClassifier
from classification_dataset import ProducerDataset, FullImageProducerDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

load_dotenv()

TRAIN_ANNOTATIONS_PATH = os.getenv("TRAIN_ANNOTATIONS_PATH")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

rotation90 = transforms.RandomRotation((90, 90))
rotation180 = transforms.RandomRotation((180, 180))
rotation270 = transforms.RandomRotation((270, 270))
no_rotate  = transforms.Lambda(lambda x: x)

fixed_rotations = transforms.RandomChoice([
    no_rotate,
    rotation90,
    rotation180,
    rotation270
])

def train_model(train_loader, val_loader, model_name, visualize=False):
    model = ProducerClassifier(num_classes=5).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}    

    model_path = f"{model_name}.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("Model saved!")
    
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title(f'Loss Evolution - {model_name}')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.legend()
        plt.title(f'Accuracy Evolution - {model_name}')
        plt.savefig(f'training_{model_name}.png')
        plt.show()
    
    return model

def train_model_cropped():
    """Возвращает обученную модель классификатора для обрезанных первой моделью кодов производителя"""
    train_transform_cropped = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    cropped_train_dataset = ProducerDataset("train_classification.csv", train_transform_cropped)

    cropped_train_size = int(0.8 * len(cropped_train_dataset))
    cropped_val_size = len(cropped_train_dataset) - cropped_train_size
    cropped_train_subset, cropped_val_subset = torch.utils.data.random_split(
        cropped_train_dataset, [cropped_train_size, cropped_val_size])

    cropped_train_loader = DataLoader(cropped_train_subset, batch_size=BATCH_SIZE, shuffle=True)
    cropped_val_loader = DataLoader(cropped_val_subset, batch_size=BATCH_SIZE)

    print("Training model on cropped images...")

    return train_model(cropped_train_loader, cropped_val_loader, "producer_classifier")

def train_model_full():
    """Возвращает обученную модель классификатора для полных изображений"""
    train_transform_full = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = FullImageProducerDataset(TRAIN_ANNOTATIONS_PATH, train_transform_full)

    full_train_size = int(0.8 * len(full_train_dataset))
    full_val_size = len(full_train_dataset) - full_train_size
    full_train_subset, full_val_subset = torch.utils.data.random_split(
        full_train_dataset, [full_train_size, full_val_size])

    full_train_loader = DataLoader(full_train_subset, batch_size=BATCH_SIZE, shuffle=True)
    full_val_loader = DataLoader(full_val_subset, batch_size=BATCH_SIZE)

    print("\nTraining model on full images...")
    return train_model(full_train_loader, full_val_loader, "producer_classifier_full")