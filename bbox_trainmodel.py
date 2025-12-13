import os
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

DATASET_PATH = os.getenv("YOLO_DATASET_PATH", "dataset_yolo")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

MODEL = os.getenv("YOLO_MODEL", "yolov10n.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", 640))
EPOCHS = int(os.getenv("YOLO_EPOCHS", 100))
BATCH = int(os.getenv("YOLO_BATCH", 16))
DEVICE = os.getenv("YOLO_DEVICE", "0")

PROJECT = os.getenv("YOLO_PROJECT", "runs/yolo")
NAME = os.getenv("YOLO_NAME", "code_detector")


def train_model():
    """
    Полный цикл обучения YOLOv10
    """

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,

        # важные параметры для мелких объектов (коды)
        close_mosaic=10,
        patience=20,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        amp=True,
        verbose=True
    )

    return model


def validate_model():
    """
    Валидация на test split
    """
    model = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        device=DEVICE
    )

    print("\nValidation metrics:")
    print(metrics)


if __name__ == "__main__":
    train_model()
    validate_model()
