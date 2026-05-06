"""Dataset and data loading for chest X-ray images."""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_transforms(image_size: int, train: bool = True):
    """Get image transforms for training or validation."""
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])


class ChestXRayDataset(Dataset):
    """Dataset for chest X-ray images with class subfolders."""

    def __init__(self, root: str, transform=None, image_size: int = 224):
        self.root = Path(root)
        self.transform = transform or get_transforms(image_size, train=False)
        self.samples = []
        self.class_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        if not self.root.exists():
            return
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            cls_dir = self.root / cls
            for path in cls_dir.glob("*.*"):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def classes(self):
        return list(self.class_to_idx.keys())


def get_class_names_from_dir(data_dir: Path) -> list:
    """Get class names from directory structure."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
