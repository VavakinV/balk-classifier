import torch
import cv2
from PIL import Image
from torchvision import transforms

from bbox_model import BBoxModel
from classification_model import ProducerClassifier


class FullPipeline:
    def __init__(
        self,
        detector_weights_path: str,
        classifier_cropped_path: str,
        classifier_full_path: str,
        threshold: float = 0.8,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.detector = BBoxModel(
            weights=detector_weights_path,
            device=device,
            conf=0.25,
            iou=0.5,
            code_class_id=2  # class "code"
        )

        self.classifier_cropped = ProducerClassifier(num_classes=5).to(self.device)
        self.classifier_cropped.load_state_dict(
            torch.load(classifier_cropped_path, map_location=self.device)
        )
        self.classifier_cropped.eval()

        self.classifier_full = ProducerClassifier(num_classes=5).to(self.device)
        self.classifier_full.load_state_dict(
            torch.load(classifier_full_path, map_location=self.device)
        )
        self.classifier_full.eval()


        self.classification_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.class_names = [
            "altai",
            "begickaya",
            "promlit",
            "ruzhimmash",
            "tihvin"
        ]

    def detect_bbox(self, image):
        """
        Возвращает bbox в пиксельных координатах:
        (x_min, y_min, x_max, y_max)

        Если YOLO не нашёл код — возвращается bbox всего изображения
        (логика fallback внутри модели).
        """
        h, w = image.shape[:2]

        bbox_norm = self.detector.forward(image)  # [x1,y1,x2,y2] normalized

        x_min = int(bbox_norm[0] * w)
        y_min = int(bbox_norm[1] * h)
        x_max = int(bbox_norm[2] * w)
        y_max = int(bbox_norm[3] * h)

        # safety clamp
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, h - 1))

        if x_max <= x_min or y_max <= y_min:
            return 0, 0, w, h

        return x_min, y_min, x_max, y_max

    def predict(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        bbox = self.detect_bbox(image)

        cropped_result = self._predict_cropped(image, bbox)
        if cropped_result["confidence"] >= self.threshold:
            return {
                "producer": cropped_result["producer"],
                "confidence": cropped_result["confidence"],
                "bbox": bbox,
                "source": "cropped"
            }

        full_result = self._predict_full(image)
        return {
            "producer": full_result["producer"],
            "confidence": full_result["confidence"],
            "bbox": bbox,
            "source": "full"
        }


    def _predict_cropped(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            return {"producer": "unknown", "confidence": 0.0}

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        img_tensor = self.classification_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier_cropped(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        return {
            "producer": self.class_names[idx.item()],
            "confidence": confidence.item()
        }


    def _predict_full(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        img_tensor = self.classification_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier_full(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        return {
            "producer": self.class_names[idx.item()],
            "confidence": confidence.item()
        }
