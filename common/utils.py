import re
import os
import torch
import random
import numpy as np


# 对所有库设置相同的随机数种子，保证训练过程可复现
def seed_everything(seed):
    random.seed(seed)  # Python内置随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置哈希种子
    np.random.seed(seed)  # Numpy随机数种子
    torch.manual_seed(seed)  # PyTorch随机数种子
    torch.cuda.manual_seed(seed)  # 当前GPU随机数种子

    torch.backends.cudnn.deterministic = True  # 保证CuDNN操作确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动选择优化算法


def sorted_alphanum(file_names):
    convert = lambda x: int(x) if x.isdigit() else x
    alphanum_key = lambda img_name: [convert(c) for c in re.split('([0-9]+)', img_name)]
    return sorted(file_names, key=alphanum_key)
