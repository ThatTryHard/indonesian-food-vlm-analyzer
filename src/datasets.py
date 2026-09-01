"""PyTorch datasets for weak recipe pretraining and visible-ingredient fine-tuning."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def build_transforms(image_size: int = 224):
    """Aspect-ratio-preserving transforms; avoids the old 224x224 stretch."""
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, evaluation_transform


class MultiLabelImageDataset:
    def __init__(self, frame, ontology, image_root: str | Path | None = None, transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.ontology = ontology
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def _path(self, row):
        if "image_path" in row and str(row["image_path"]).strip():
            return Path(row["image_path"])
        if self.image_root is None:
            raise ValueError("image_root is required when frame lacks image_path")
        return self.image_root / row["relative_path"]

    def __getitem__(self, index):
        import torch

        row = self.frame.iloc[index]
        path = self._path(row)
        try:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        if self.transform:
            image = self.transform(image)
        raw_labels = row["labels"]
        if isinstance(raw_labels, str):
            labels = self.ontology.parse_annotation_cell(raw_labels)
        else:
            labels = list(raw_labels)
        target = torch.zeros(len(self.ontology.ids), dtype=torch.float32)
        for label in labels:
            target[self.ontology.ids.index(label)] = 1.0
        return image, target, str(row.get("sample_id", path.name)), str(path)
