import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ProducerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_model.features[5:].parameters():
            param.requires_grad = True
        for param in self.base_model.features[7:].parameters():
            param.requires_grad = True

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)