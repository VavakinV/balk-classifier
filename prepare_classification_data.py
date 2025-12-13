import os
import csv
import cv2
from tqdm import tqdm
from dotenv import load_dotenv

from bbox_model import BBoxModel  # YOLO-based

load_dotenv()

class ImageCropper:
    def __init__(self, detector_weights_path: str, device: str = "cuda"):
        """
        detector_weights_path — путь к YOLO весам (.pt / .pth)
        """
        self.detector = BBoxModel(
            weights=detector_weights_path,
            device=device,
            conf=0.25,
            iou=0.5,
            code_class_id=2  # class "code"
        )

    def detect_and_crop(self, image_path: str):
        """
        Обнаруживает и вырезает область с кодом производителя.
        Возвращает np.ndarray (BGR) или None.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        h, w = image.shape[:2]

        # YOLO inference (always returns exactly one bbox)
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
            return None

        return image[y_min:y_max, x_min:x_max]


def process_annotations(
    annotations_path: str,
    images_path: str,
    output_dir: str,
    output_csv: str,
    detector_weights_path: str
):
    """
    Обрабатывает CSV-аннотации:
    - читает изображения
    - детектирует bbox
    - сохраняет cropped
    - пишет CSV для классификатора
    """
    os.makedirs(output_dir, exist_ok=True)
    cropper = ImageCropper(detector_weights_path)

    with open(annotations_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "producer"])

        for row in tqdm(rows, desc="Processing images"):
            image_name = os.path.basename(row["image"])
            image_path = os.path.join(images_path, image_name)

            if not os.path.exists(image_path):
                continue

            cropped = cropper.detect_and_crop(image_path)
            if cropped is None or cropped.size == 0:
                continue

            output_path = os.path.join(output_dir, f"{row['id']}.jpg")
            cv2.imwrite(output_path, cropped)

            writer.writerow([output_path, row["producer"]])


def prepare_data():
    TRAIN_ANNOTATIONS = os.getenv("TRAIN_ANNOTATIONS_PATH")
    TRAIN_IMAGES = os.getenv("TRAIN_IMAGES_PATH")

    DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH")

    process_annotations(
        annotations_path=TRAIN_ANNOTATIONS,
        images_path=TRAIN_IMAGES,
        output_dir="train_cropped",
        output_csv="train_classification.csv",
        detector_weights_path=DETECTOR_MODEL_PATH
    )
