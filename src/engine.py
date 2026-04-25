"""统一训练/测试引擎与相似度检索。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader

__all__ = ["train_epoch", "test_epoch", "test_epoch_with_acc", "create_embeddings", "compute_similarity"]


ModelLike = nn.Module | Sequence[nn.Module]


def train_epoch(
    model: ModelLike,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """执行一个训练 epoch，支持单模型或编码器-解码器组合。

    Args:
        model: 待训练模型，或 ``[encoder, decoder]`` 序列。
        train_loader: 训练数据加载器。
        loss_fn: 损失函数。
        optimizer: 优化器。
        device: 计算设备。

    Returns:
        该 epoch 的平均损失。
    """
    models = model if isinstance(model, Sequence) else [model]
    for m in models:
        m.train()

    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if len(models) == 2:
            features = models[0](data)
            output = models[1](features)
        else:
            output = models[0](data)

        loss_val = loss_fn(output, target)
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_val.item()

    n_batches = len(train_loader)
    return total_loss / n_batches if n_batches > 0 else 0.0


def test_epoch(
    model: ModelLike,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """执行一个测试 epoch，仅返回平均损失。

    Args:
        model: 待评估模型，或 ``[encoder, decoder]`` 序列。
        test_loader: 测试数据加载器。
        loss_fn: 损失函数。
        device: 计算设备。

    Returns:
        平均损失。
    """
    models = model if isinstance(model, Sequence) else [model]
    for m in models:
        m.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = models[1](models[0](data)) if len(models) == 2 else models[0](data)

            loss_val = loss_fn(output, target)
            total_loss += loss_val.item()

    n_batches = len(test_loader)
    return total_loss / n_batches if n_batches > 0 else 0.0


def test_epoch_with_acc(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """执行一个测试 epoch，返回平均损失与分类准确率（仅适用于分类任务）。

    Args:
        model: 待评估模型。
        test_loader: 测试数据加载器。
        loss_fn: 损失函数。
        device: 计算设备。

    Returns:
        (平均损失, 准确率)。
    """
    model.eval()
    total_loss = 0.0
    correct_num = 0
    total_num = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1)
            total_num += data.shape[0]
            correct_num += (pred == target).sum().item()

            loss_val = loss_fn(output, target)
            total_loss += loss_val.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_num / total_num if total_num > 0 else 0.0
    return avg_loss, accuracy


def create_embeddings(
    encoder: nn.Module,
    full_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """利用编码器为全量数据集生成嵌入向量。

    Args:
        encoder: 训练好的编码器。
        full_loader: 全量数据加载器。
        device: 计算设备。

    Returns:
        形状为 ``(N, D)`` 的嵌入矩阵（numpy）。
    """
    encoder.eval()
    embeddings: list[torch.Tensor] = []

    with torch.no_grad():
        for data, _ in full_loader:
            data = data.to(device)
            output = encoder(data).detach().cpu()
            embeddings.append(output)

    embedding = torch.cat(embeddings, dim=0)
    return embedding.reshape(embedding.shape[0], -1).numpy()


def compute_similarity(
    encoder: nn.Module,
    img_tensor: torch.Tensor,
    num_imgs: int,
    embedding: np.ndarray,
    device: torch.device,
) -> list[list[int]]:
    """计算给定图像在嵌入空间中的 K 近邻。

    Args:
        encoder: 编码器模型。
        img_tensor: 输入图像张量，形状 ``(B, C, H, W)``。
        num_imgs: 返回的近邻数量 ``K``。
        embedding: 全量嵌入矩阵，形状 ``(N, D)``。
        device: 计算设备。

    Returns:
        每个查询图像的 K 近邻索引列表。
    """
    encoder.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = encoder(img_tensor).detach().cpu().numpy()

    img_vector = output.reshape((output.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_imgs, metric="cosine")
    knn.fit(embedding)
    _, indices = knn.kneighbors(img_vector)
    return indices.tolist()
