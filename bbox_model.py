from ultralytics import YOLO
import torch

class BBoxModel:
    """
    YOLOv10-based bbox detector with guaranteed output.

    - Returns exactly ONE bbox
    - If 'code' class detected -> highest confidence bbox
    - If not detected -> full-image bbox
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        conf: float = 0.1,
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
        image: np.ndarray (H, W, C)
        return: [x1, y1, x2, y2] normalized to [0, 1]
        """

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        best_box = None
        best_conf = -1.0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls.item())
                if cls != self.code_class_id:
                    continue

                conf = float(box.conf.item())
                if conf > best_conf:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    best_box = [x1, y1, x2, y2]
                    best_conf = conf

        if best_box is None:
            return [0.0, 0.0, 1.0, 1.0]

        return best_box
