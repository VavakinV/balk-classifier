import os
import cv2
import random
import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_iou
from bbox_model import BBoxModel


def visualize_test_predictions(
    model,
    test_data,
    n: int = 5
):
    """
    Визуализация результатов YOLO-based BBoxModel.

    model.forward(image) -> [x1, y1, x2, y2] (normalized)
    test_data[i] = {
        'img_path': str,
        'bbox': [x1, y1, x2, y2]  # pixel coords
    }
    """

    selected_samples = random.sample(test_data, min(n, len(test_data)))

    for sample in selected_samples:
        try:
            img = cv2.imread(sample["img_path"])
            if img is None:
                print(f"Warning: could not read {sample['img_path']}")
                continue

            h, w = img.shape[:2]
            display_img = img.copy()

            bbox_norm = model.forward(img)

            if bbox_norm is None or len(bbox_norm) != 4:
                print(f"Warning: invalid prediction for {sample['img_path']}")
                continue

            x1p = int(bbox_norm[0] * w)
            y1p = int(bbox_norm[1] * h)
            x2p = int(bbox_norm[2] * w)
            y2p = int(bbox_norm[3] * h)

            x1p = max(0, min(w - 1, x1p))
            y1p = max(0, min(h - 1, y1p))
            x2p = max(0, min(w - 1, x2p))
            y2p = max(0, min(h - 1, y2p))

            x1t, y1t, x2t, y2t = map(int, sample["bbox"])

            pred_box = torch.tensor([[x1p, y1p, x2p, y2p]], dtype=torch.float32)
            true_box = torch.tensor([[x1t, y1t, x2t, y2t]], dtype=torch.float32)
            iou = box_iou(pred_box, true_box).item()

            # GT — red
            cv2.rectangle(display_img, (x1t, y1t), (x2t, y2t), (0, 0, 255), 3)

            # Pred — green
            cv2.rectangle(display_img, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)

            cv2.putText(
                display_img,
                f"IoU: {iou:.3f}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            plt.title(os.path.basename(sample["img_path"]))
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error processing {sample['img_path']}: {e}")

if __name__ == "__main__":
    import csv
    import json
    from dotenv import load_dotenv

    load_dotenv()

    TEST_IMAGES_PATH = os.getenv("TEST_IMAGES_PATH")
    TEST_ANNOTATIONS_PATH = os.getenv("TEST_ANNOTATIONS_PATH")

    BBOX_MODEL_WEIGHTS = os.getenv("DETECTOR_MODEL_PATH")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_SAMPLES = 20

    test_data = []

    with open(TEST_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = os.path.basename(row["image"])
            image_path = os.path.join(TEST_IMAGES_PATH, image_name)

            if not os.path.exists(image_path):
                continue

            if not row["code_bbox"]:
                continue

            try:
                # code_bbox хранится как JSON-строка
                bbox_data = json.loads(row["code_bbox"])
            except Exception:
                continue

            # Ожидаем хотя бы один bbox
            if not isinstance(bbox_data, list) or len(bbox_data) == 0:
                continue

            bbox_item = bbox_data[0]

            # Нормализованные координаты (Label Studio)
            x = float(bbox_item["x"]) / 100.0
            y = float(bbox_item["y"]) / 100.0
            w = float(bbox_item["width"]) / 100.0
            h = float(bbox_item["height"]) / 100.0

            img = cv2.imread(image_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            x_min = x * img_w
            y_min = y * img_h
            x_max = (x + w) * img_w
            y_max = (y + h) * img_h

            test_data.append({
                "img_path": image_path,
                "bbox": [x_min, y_min, x_max, y_max]
            })

    print(f"Loaded {len(test_data)} test samples")

    if len(test_data) == 0:
        raise RuntimeError("No valid test samples found")

    model = BBoxModel(
        weights=BBOX_MODEL_WEIGHTS,
        device=DEVICE,
        conf=0.1,
        iou=0.5,
        code_class_id=2
    )

    visualize_test_predictions(
        model=model,
        test_data=test_data,
        n=NUM_SAMPLES
    )
