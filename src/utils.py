"""通用工具函数。"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

__all__ = ["seed_everything", "sorted_alphanum"]


def seed_everything(seed: int) -> None:
    """为所有相关库设置相同的随机数种子，保证训练过程可复现。

    Args:
        seed: 全局随机种子。
    """
    random.seed(seed)
    # noinspection SpellCheckingInspection
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sorted_alphanum(file_names: Sequence[str | Path]) -> list[str]:
    """按人类可读的自然数顺序对文件名进行排序。

    例如：``['1.jpg', '10.jpg', '2.jpg']`` → ``['1.jpg', '2.jpg', '10.jpg']``

    Args:
        file_names: 文件名序列。

    Returns:
        排序后的文件名列表。
    """

    def _alphanum_key(name: str | Path) -> list[int | str]:
        s = str(name)
        return [int(c) if c.isdigit() else c for c in re.split(r"([0-9]+)", s)]

    return [str(name) for name in sorted(file_names, key=_alphanum_key)]
