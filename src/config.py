"""全局配置与任务预设。"""

from __future__ import annotations

import json
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


def _load_runtime_cfg() -> dict:
    """从项目根目录的 runtime_config.json 加载运行时配置。"""
    cfg_path = Path(__file__).resolve().parent.parent / "runtime_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _apply_overrides(cfg: Config, overrides: dict) -> None:
    """将字典中的值覆盖到 Config 实例，自动转换路径类型。"""
    path_fields = {"img_path", "labels_path", "model_dir"}
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            continue
        if key == "labels_path" and cfg.task != "classification":
            continue
        if key in path_fields and val is not None:
            val = Path(val)
        setattr(cfg, key, val)


# 加载运行时配置并构建 PRESETS
_json_cfg = _load_runtime_cfg()

_classification = Config(task="classification", labels_path=Path("data/fashion-labels.csv"))
_denoising = Config(task="denoising", img_h=68, img_w=68, epochs=30, noise_ratio=0.5)
_similarity = Config(task="similarity", epochs=30)

for _cfg in (_classification, _denoising, _similarity):
    _apply_overrides(_cfg, _json_cfg.get("global", {}))
    _apply_overrides(_cfg, _json_cfg.get(_cfg.task, {}))

PRESETS: dict[str, Config] = {
    "classification": _classification,
    "denoising": _denoising,
    "similarity": _similarity,
}
