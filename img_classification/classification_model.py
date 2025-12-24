__all__ = ['ClassifierModel']

import torch
import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    input = torch.randn(2, 3, 64, 64)
    model = ClassifierModel()
    output = model(input)
    print(output.shape)
