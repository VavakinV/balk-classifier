import torch
import torch.nn as nn
from torchvision import models


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
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

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        self.att1 = SpatialAttention(kernel_size=5)
        self.att2 = SpatialAttention(kernel_size=7)

        self.fuse = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.att1(x)
        x = self.att2(x)
        x = self.fuse(x) 

        out = self.head(x)
        out = out.mean(dim=[2, 3])
        return out


class CenterIoULoss(nn.Module):
    def __init__(self, iou_weight=0.5, eps=1e-6):
        super(CenterIoULoss, self).__init__()
        self.iou_weight = iou_weight
        self.center_weight = 1.0 - iou_weight
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union = pred_area + target_area - intersection + self.eps

        iou = intersection / union
        iou_loss = 1.0 - iou

        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2

        center_distance = torch.sqrt((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)
        normalized_distance = center_distance / torch.sqrt(torch.tensor(2.0, device=pred.device))

        return (self.iou_weight * iou_loss + self.center_weight * normalized_distance).mean()
