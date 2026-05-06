"""Neural network model for lung disease classification."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Build ResNet18-based classifier for chest X-ray.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


class LungDiseaseClassifier(nn.Module):
    """Wrapper for consistent interface."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone = build_model(num_classes, pretrained=pretrained)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)
