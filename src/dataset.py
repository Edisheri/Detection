"""Dataset и трансформации для рентгеновских снимков."""
from pathlib import Path
from typing import Optional, Dict

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_transforms(image_size: int, train: bool = True):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ChestXRayDataset(Dataset):
    """Dataset с подкаталогами-классами."""

    def __init__(
        self,
        root: str,
        transform=None,
        image_size: int = 224,
        class_to_idx: Optional[Dict[str, int]] = None,
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.transform = transform or get_transforms(image_size, train=False)
        self.samples = []
        self.class_to_idx = dict(class_to_idx) if class_to_idx else {}
        self.max_per_class = max_per_class
        self._load_samples()

    def _load_samples(self):
        if not self.root.exists():
            return
        if not self.class_to_idx:
            classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls, idx in self.class_to_idx.items():
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            count = 0
            for path in sorted(cls_dir.glob("*.*")):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(path), idx))
                    count += 1
                    if self.max_per_class and count >= self.max_per_class:
                        break

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
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
