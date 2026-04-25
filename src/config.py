"""全局配置与任务预设。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "PRESETS"]


@dataclass
class Config:
    """任务配置类。

    通过 ``PRESETS[task]`` 获取各任务的默认配置，命令行参数可进一步覆盖。
    """

    task: str
    img_path: Path = Path("data/dataset")
    labels_path: Path | None = None
    img_h: int = 64
    img_w: int = 64
    seed: int = 42
    train_ratio: float = 0.75
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    model_dir: Path = Path("finetuned")
    device: str = "auto"
    max_samples: int | None = None

    # classification specific
    num_classes: int = 5
    classification_names: dict[int, str] = field(
        default_factory=lambda: {
            0: "上衣",
            1: "鞋",
            2: "包",
            3: "下身衣服",
            4: "手表",
        }
    )

    # denoising specific
    noise_ratio: float = 0.5

    # similarity specific
    full_batch_size: int = 32
    num_similar: int = 5

    @property
    def model_path(self) -> Path:
        """主模型保存路径。"""
        name_map = {
            "classification": "classifier.pt",
            "denoising": "denoiser.pt",
            "similarity": "encoder.pt",
        }
        return self.model_dir / name_map[self.task]

    @property
    def decoder_path(self) -> Path | None:
        """解码器保存路径（仅相似度任务）。"""
        return self.model_dir / "decoder.pt" if self.task == "similarity" else None

    @property
    def embedding_path(self) -> Path | None:
        """嵌入矩阵保存路径（仅相似度任务）。"""
        return self.model_dir / "embeddings.npy" if self.task == "similarity" else None


PRESETS: dict[str, Config] = {
    "classification": Config(
        task="classification",
        labels_path=Path("data/fashion-labels.csv"),
    ),
    "denoising": Config(
        task="denoising",
        img_h=68,
        img_w=68,
        epochs=30,
        noise_ratio=0.5,
    ),
    "similarity": Config(
        task="similarity",
        full_batch_size=32,
        epochs=30,
    ),
}
