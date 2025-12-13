import os
import cv2
import random
import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_iou


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

