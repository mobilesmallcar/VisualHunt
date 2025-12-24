__all__ = ['create_dataset']

import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split

from denoising_config import *

from VisualHunt.common.utils import sorted_alphanum


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted_alphanum(os.listdir(root_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        img_noise = img_tensor + NOISE_RATIO * torch.randn_like(img_tensor)
        img_noise = torch.clamp(img_noise, 0., 1.)
        return img_noise, img_tensor


def create_dataset():
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    dataset = ImageDataset(IMG_PATH, transform)

    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    return train_dataset, test_dataset


# 测试
if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
    print(test_dataset[0][0].shape)
    print(test_dataset[0][1].shape)
