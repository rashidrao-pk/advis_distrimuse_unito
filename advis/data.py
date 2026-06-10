from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int = 128, augmentation: str = "min"):
    if augmentation == "custom":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0.01, translate=(0.01, 0.01), shear=0.1, scale=(0.99, 1.0), fill=(0, 0, 0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)
