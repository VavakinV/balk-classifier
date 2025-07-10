import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from bbox_model import BBoxModel
from classification_model import ProducerClassifier

class FullPipeline:
    def __init__(self, detector_path, classifier_cropped_path, classifier_full_path, threshold=0.8, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold

        self.detector = BBoxModel().to(self.device)
        self.detector.load_state_dict(torch.load(detector_path))
        self.detector.eval()

        self.classifier_cropped = ProducerClassifier(num_classes=5).to(self.device)
        self.classifier_cropped.load_state_dict(torch.load(classifier_cropped_path))
        self.classifier_cropped.eval()
        
        self.classifier_full = ProducerClassifier(num_classes=5).to(self.device)
        self.classifier_full.load_state_dict(torch.load(classifier_full_path))
        self.classifier_full.eval()

        self.detection_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.classification_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['altai', 'begickaya', 'promlit', 'ruzhimmash', 'tihvin']
    
    def detect_bbox(self, image):
        """Обнаруживает bounding box с кодом производителя на изображении."""
        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_image.shape[:2]
        
        img_resized = cv2.resize(orig_image, (320, 320))
        img_tensor = self.detection_transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            bbox_norm = self.detector(img_tensor)[0].cpu().numpy()
        
        bbox_320 = bbox_norm * 320.0
        x_min = int(bbox_320[0] * (orig_w / 320.0))
        y_min = int(bbox_320[1] * (orig_h / 320.0))
        x_max = int(bbox_320[2] * (orig_w / 320.0))
        y_max = int(bbox_320[3] * (orig_h / 320.0))
        
        x_min = max(0, min(x_min, orig_w - 1))
        y_min = max(0, min(y_min, orig_h - 1))
        x_max = max(0, min(x_max, orig_w - 1))
        y_max = max(0, min(y_max, orig_h - 1))
        
        if x_max <= x_min or y_max <= y_min:
            return 0, 0, orig_w, orig_h
        
        return x_min, y_min, x_max, y_max

    def predict(self, image_path):
        """Полный пайплайн обработки с выбором лучшего предсказания."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        bbox = self.detect_bbox(image)
        x_min, y_min, x_max, y_max = bbox

        cropped_result = self._predict_cropped(image, bbox)
        
        if cropped_result['confidence'] >= self.threshold:
            return {
                'producer': cropped_result['producer'],
                'confidence': cropped_result['confidence'],
                'bbox': bbox,
                'source': 'cropped'
            }

        full_result = self._predict_full(image)
        return {
            'producer': full_result['producer'],
            'confidence': full_result['confidence'],
            'bbox': bbox,
            'source': 'full'
        }
    
    def _predict_cropped(self, image, bbox):
        """Предсказание на обрезанной области."""
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return {
                'producer': 'unknown',
                'confidence': 0.0
            }
        
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        img_tensor = self.classification_transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier_cropped(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, 0)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
        
        return {
            'producer': self.class_names[pred_idx],
            'confidence': confidence
        }
    
    def _predict_full(self, image):
        """Предсказание на полном изображении."""
        full_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(full_rgb)
        img_tensor = self.classification_transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier_full(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, 0)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
        
        return {
            'producer': self.class_names[pred_idx],
            'confidence': confidence
        }