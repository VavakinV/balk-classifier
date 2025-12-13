from ultralytics import YOLO
import torch

class BBoxModel:
    """
    YOLOv10-based replacement for legacy ResNet BBoxModel
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.5,
        code_class_id: int = 2
    ):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.code_class_id = code_class_id

    @torch.no_grad()
    def forward(self, image):
        """
        image: np.ndarray (HWC, BGR or RGB)
        return: List[bbox] in normalized xyxy format
        """

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        boxes = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls.item())
                if cls != self.code_class_id:
                    continue

                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                boxes.append([x1, y1, x2, y2])

        return boxes
