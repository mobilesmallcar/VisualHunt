"""统一数据集构建。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split

from src.config import Config
from src.utils import sorted_alphanum

__all__ = ["ImageDataset", "ImageLabelDataset", "NoisyImageDataset", "create_datasets"]


class ImageDataset(Dataset):
    """基础图像数据集（输入与目标相同），用于相似度检索的自编码器训练。"""

    def __init__(self, root_dir: Path, transform: transforms.Compose | None = None, max_samples: int | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.imgs = sorted_alphanum([p.name for p in self.root_dir.iterdir()])
        if max_samples is not None:
            self.imgs = self.imgs[:max_samples]

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.root_dir / self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return tensor, tensor


class ImageLabelDataset(Dataset):
    """带标签的时尚商品图像数据集，用于分类任务。"""

    def __init__(
        self,
        root_dir: Path,
        label_path: Path,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.imgs = sorted_alphanum([p.name for p in self.root_dir.iterdir()])
        if max_samples is not None:
            self.imgs = self.imgs[:max_samples]

        labels = pd.read_csv(label_path)
        self.labels_dict = dict(zip(labels["id"], labels["target"], strict=False))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.root_dir / self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = self.labels_dict[idx]
        return tensor, label


class NoisyImageDataset(Dataset):
    """去噪自编码器数据集：输入为加噪图像，目标为原图。"""

    def __init__(
        self,
        root_dir: Path,
        noise_ratio: float,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.noise_ratio = noise_ratio
        self.transform = transform
        self.imgs = sorted_alphanum([p.name for p in self.root_dir.iterdir()])
        if max_samples is not None:
            self.imgs = self.imgs[:max_samples]

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.root_dir / self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        noise = self.noise_ratio * torch.randn_like(tensor)
        noisy = torch.clamp(tensor + noise, 0.0, 1.0)
        return noisy, tensor


def _get_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((cfg.img_h, cfg.img_w)),
            transforms.ToTensor(),
        ]
    )


def create_datasets(cfg: Config) -> tuple[Dataset, Dataset, Dataset | None]:
    """根据任务配置创建数据集。

    Args:
        cfg: 任务配置。

    Returns:
        (train_dataset, test_dataset, full_dataset | None)。
        ``full_dataset`` 仅在相似度任务中返回，用于生成全局嵌入。
    """
    transform = _get_transform(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)

    if cfg.task == "classification":
        dataset = ImageLabelDataset(cfg.img_path, cfg.labels_path, transform, cfg.max_samples)
        train_ds, test_ds = random_split(dataset, [cfg.train_ratio, 1 - cfg.train_ratio], generator=generator)
        return train_ds, test_ds, None

    if cfg.task == "denoising":
        dataset = NoisyImageDataset(cfg.img_path, cfg.noise_ratio, transform, cfg.max_samples)
        train_ds, test_ds = random_split(dataset, [cfg.train_ratio, 1 - cfg.train_ratio], generator=generator)
        return train_ds, test_ds, None

    if cfg.task == "similarity":
        dataset = ImageDataset(cfg.img_path, transform, cfg.max_samples)
        train_ds, test_ds = random_split(dataset, [cfg.train_ratio, 1 - cfg.train_ratio], generator=generator)
        return train_ds, test_ds, dataset

    raise ValueError(f"未知任务类型: {cfg.task}")
