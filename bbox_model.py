import torch
import torch.nn as nn
from torchvision import models

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        
        combined = torch.cat([max_out, mean_out, std_out], dim=1)
        att_map = self.conv(combined)
        return x * self.sigmoid(att_map)

class BBoxModel(nn.Module):
    def __init__(self):
        super(BBoxModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Блоки внимания
        self.att1 = SpatialAttention(kernel_size=5)
        self.att2 = SpatialAttention(kernel_size=9)
        self.att3 = SpatialAttention(kernel_size=7)
        
        # Адаптивный пулинг
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.regressor = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.att1(x)

        x = self.layer3(x)
        x = self.att2(x)
        x = self.att3(x) 

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
    
class CenterIoULoss(nn.Module):
    def __init__(self, iou_weight=0.5, eps=1e-6):
        super(CenterIoULoss, self).__init__()
        self.iou_weight = iou_weight
        self.center_weight = 1.0 - iou_weight
        self.eps = eps
        
    def forward(self, pred, target):
        """
        Вычисляет IoU Loss между предсказанными и целевыми bounding box.
        
        Формат bbox: [x_min, y_min, x_max, y_max]
        Координаты нормализованы в диапазоне [0, 1]
        """
        # Гарантируем, что координаты валидны
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Разделяем координаты
        pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
        
        # Вычисляем координаты пересечения
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        # Площадь пересечения (с защитой от отрицательных значений)
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height
        
        # Площади прямоугольников
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Площадь объединения
        union = pred_area + target_area - intersection + self.eps
        
        # IoU
        iou = intersection / union
        
        # IoU Loss = 1 - IoU
        iou_loss = 1.0 - iou
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        # Евклидово расстояние (нормализованное)
        center_distance = torch.sqrt(
            (pred_center_x - target_center_x)**2 + 
            (pred_center_y - target_center_y)**2
        )

        normalized_distance = center_distance / torch.sqrt(torch.tensor(2.0, device=pred.device))

        combined_loss = self.iou_weight * iou_loss + self.center_weight * normalized_distance

        return combined_loss.mean()