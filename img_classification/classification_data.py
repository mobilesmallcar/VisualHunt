__all__ = ['create_dataset']

import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split

from classification_config import *

from VisualHunt.common.utils import sorted_alphanum


class ImageLabelDataSet(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted_alphanum(os.listdir(root_dir))

        # 读取分类标签
        labels = pd.read_csv(label_dir)
        # 将分类标签保存为字典
        self.labels_dict = dict(zip(labels['id'], labels['target']))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        # 在标签字典中查找对应分类标签
        img_label = self.labels_dict[idx]
        return img_tensor, img_label


def create_dataset():
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    dataset = ImageLabelDataSet(IMG_PATH, FASHION_LABELS_PATH, transform)

    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    print(test_dataset[0][0].shape)
    print(classification_names[test_dataset[0][1]])
