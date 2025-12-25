__all__ = ['create_dataset']

import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split

from similarity_config import *
from VisualHunt.common.utils import sorted_alphanum


class ImageDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted_alphanum(os.listdir(root_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(img)

        return img_tensor, img_tensor


def create_dataset():
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    dataset = ImageDataSet(IMG_PATH, transform)
    train_dataset, test_dataset = random_split(dataset, lengths=[TRAIN_RATIO, TEST_RATIO])

    return dataset, train_dataset, test_dataset


if __name__ == "__main__":
    full_dataset, train_dataset, test_dataset = create_dataset()
    print(len(full_dataset), len(train_dataset), len(test_dataset))
    assert len(full_dataset) == len(train_dataset) + len(test_dataset)
