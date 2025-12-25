__all__ = ['ConvEncoder', 'ConvDecoder']

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # print(f"第一层卷积池化后的形状:{x.shape}")
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # print(f"第二层卷积池化后的形状:{x.shape}")
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        # print(f"第三层卷积池化后的形状:{x.shape}")
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        # print(f"第四层卷积池化后的形状:{x.shape}")
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        # print(f"第五层卷积池化后的形状:{x.shape}")
        return x


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print("第一层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv2(x))
        # print("第二层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv3(x))
        # print("第三层转置卷积后的形状：", x.shape)
        x = torch.relu(self.conv4(x))
        # print("第四层转置卷积后的形状：", x.shape)
        # 最后一层激活函数用 Sigmoid，将输出限制在(0, 1)范围内
        x = torch.sigmoid(self.conv5(x))
        return x


if __name__ == '__main__':
    # 创建一个输入张量
    input_tensor = torch.randn(1, 3, 64, 64)

    # 创建编码器实例
    encoder = ConvEncoder()

    # 创建解码器实例
    decoder = ConvDecoder()

    # 前向传播
    output_tensor = decoder(encoder(input_tensor))

    # 打印输出张量的
    print(output_tensor.shape)
