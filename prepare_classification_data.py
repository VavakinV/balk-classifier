import os
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torchvision import transforms
from bbox_model import BBoxModel
from PIL import Image

load_dotenv()

class ImageCropper:
    def __init__(self, detector_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.detector = BBoxModel().to(self.device)
        self.detector.load_state_dict(torch.load(detector_path))
        self.detector.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_and_crop(self, image_path):
        """Обнаруживает и вырезает область с кодом с помощью модели детектора"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        orig_h, orig_w = image.shape[:2]
        
        # Подготовка изображения для детектора
        img_resized = cv2.resize(image, (320, 320))
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        # Предсказание bbox
        with torch.no_grad():
            bbox_norm = self.detector(img_tensor)[0].cpu().numpy()
        
        # Преобразование координат к оригинальному размеру
        bbox_320 = bbox_norm * 320.0
        x_min = int(bbox_320[0] * (orig_w / 320.0))
        y_min = int(bbox_320[1] * (orig_h / 320.0))
        x_max = int(bbox_320[2] * (orig_w / 320.0))
        y_max = int(bbox_320[3] * (orig_h / 320.0))
        
        # Коррекция координат
        x_min = max(0, min(x_min, orig_w - 1))
        y_min = max(0, min(y_min, orig_h - 1))
        x_max = max(0, min(x_max, orig_w - 1))
        y_max = max(0, min(y_max, orig_h - 1))
        
        # Проверка на валидность области
        if x_max <= x_min or y_max <= y_min:
            return None
            
        # Вырезание области
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

def process_annotations(annotations_path, images_path, output_dir, output_csv, detector_path):
    """Обрабатывает данные, используя модель детектора для обрезки"""
    os.makedirs(output_dir, exist_ok=True)
    cropper = ImageCropper(detector_path)
    
    with open(annotations_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'producer'])
        
        for row in tqdm(rows, desc="Processing images"):
            image_rel_path = row['image'].split('/')[-1]
            image_path = os.path.join(images_path, image_rel_path)
            
            if not os.path.exists(image_path):
                continue
                
            # Обрезка с помощью детектора
            cropped = cropper.detect_and_crop(image_path)
            if cropped is None or cropped.size == 0:
                continue
                
            # Сохранение вырезанного изображения
            output_path = os.path.join(output_dir, f"{row['id']}.jpg")
            cv2.imwrite(output_path, cropped)
            
            # Запись в CSV
            writer.writerow([output_path, row['producer']])

def prepare_data():
    TRAIN_ANNOTATIONS = os.getenv("TRAIN_ANNOTATIONS_PATH")
    TEST_ANNOTATIONS = os.getenv("TEST_ANNOTATIONS_PATH")
    TRAIN_IMAGES = os.getenv("TRAIN_IMAGES_PATH")
    TEST_IMAGES = os.getenv("TEST_IMAGES_PATH")
    DETECTOR_MODEL_PATH = "best_model.pth"  # Путь к обученной модели детектора

    process_annotations(
        TRAIN_ANNOTATIONS,
        TRAIN_IMAGES,
        "train_cropped",
        "train_classification.csv",
        DETECTOR_MODEL_PATH
    )

    process_annotations(
        TEST_ANNOTATIONS,
        TEST_IMAGES,
        "test_cropped",
        "test_classification.csv",
        DETECTOR_MODEL_PATH
    )