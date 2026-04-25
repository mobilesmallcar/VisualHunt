"""统一命令行入口：训练 / 测试 / 推理。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PRESETS, Config
from src.data import create_datasets
from src.engine import (
    compute_similarity,
    create_embeddings,
    test_epoch,
    test_epoch_with_acc,
    train_epoch,
)
from src.models import ClassifierModel, ConvDecoder, ConvDenoiser, ConvEncoder
from src.utils import seed_everything


def _get_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _override_cfg(cfg: Config, args: argparse.Namespace) -> Config:
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.model_dir = args.model_dir
    return cfg


# ---------------------------------------------------------------------------
# Train commands
# ---------------------------------------------------------------------------


def train_classification(cfg: Config) -> None:
    device = _get_device(cfg)
    seed_everything(cfg.seed)
    train_ds, test_ds, _ = create_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = ClassifierModel(cfg.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    min_val_loss = float("inf")
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = test_epoch_with_acc(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), cfg.model_path)
            min_val_loss = val_loss
            print("验证损失减小，保存模型。")

    print(f"最终验证损失: {min_val_loss:.6f}")


def train_denoising(cfg: Config) -> None:
    device = _get_device(cfg)
    seed_everything(cfg.seed)
    train_ds, test_ds, _ = create_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = ConvDenoiser().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    min_val_loss = float("inf")
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = test_epoch(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), cfg.model_path)
            min_val_loss = val_loss
            print("验证损失减小，保存模型。")

    print(f"最终验证损失: {min_val_loss:.6f}")


def train_similarity(cfg: Config) -> None:
    device = _get_device(cfg)
    seed_everything(cfg.seed)
    train_ds, test_ds, full_ds = create_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    full_loader = DataLoader(full_ds, batch_size=cfg.full_batch_size, shuffle=False)

    encoder = ConvEncoder().to(device)
    decoder = ConvDecoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    min_val_loss = float("inf")
    for epoch in tqdm(range(cfg.epochs), desc="Training"):
        train_loss = train_epoch([encoder, decoder], train_loader, loss_fn, optimizer, device)
        val_loss = test_epoch([encoder, decoder], test_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(encoder.state_dict(), cfg.model_path)
            torch.save(decoder.state_dict(), cfg.decoder_path)
            print("验证损失减小，保存模型。")

    print(f"最终验证损失: {min_val_loss:.6f}")

    # 生成图像嵌入矩阵
    encoder.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    embeddings = create_embeddings(encoder, full_loader, device)
    np.save(cfg.embedding_path, embeddings)
    print(f"嵌入矩阵形状: {embeddings.shape}")


# ---------------------------------------------------------------------------
# Test commands
# ---------------------------------------------------------------------------


def test_classification(cfg: Config) -> None:
    device = _get_device(cfg)
    _, test_ds, _ = create_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = ClassifierModel(cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    model.eval()

    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        outputs = model(data)

    images = data.detach().permute(0, 2, 3, 1).cpu().numpy()
    predict_labels = outputs.detach().argmax(dim=1).cpu().numpy()

    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4), sharey=True, sharex=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    for i in range(rows * cols):
        row, col = i // cols, i % cols
        true_name = cfg.classification_names[target[i].item()]
        pred_name = cfg.classification_names[predict_labels[i]]
        color = "tab:green" if target[i] == predict_labels[i] else "tab:red"
        axes[row][col].imshow(images[i])
        axes[row][col].set_title(f"真实: {true_name}\n预测: {pred_name}", color=color, fontsize=12)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()

    test_loss, test_acc = test_epoch_with_acc(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"测试集平均误差: {test_loss:.6f}")
    print(f"测试分类准确率: {test_acc:.6f}")


def test_denoising(cfg: Config) -> None:
    device = _get_device(cfg)
    _, test_ds, _ = create_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = ConvDenoiser().to(device)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    model.eval()

    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        outputs = model(data)

    noisy_imgs = data.detach().permute(0, 2, 3, 1).cpu().numpy()
    predict_imgs = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
    original_imgs = target.detach().permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(3, 10, figsize=(25, 4), sharey=True, sharex=True)
    for imgs, ax_row in zip([noisy_imgs, predict_imgs, original_imgs], axes, strict=False):
        for img, ax in zip(imgs, ax_row, strict=False):
            ax.imshow(img)
            ax.axis("off")

    plt.show()

    test_loss = test_epoch(model, test_loader, nn.MSELoss(), device)
    print(f"测试集平均误差: {test_loss:.6f}")


def test_similarity(cfg: Config) -> None:
    device = _get_device(cfg)
    _, _, full_ds = create_datasets(cfg)
    img, _ = full_ds[0]
    img = img.unsqueeze(0)
    print(f"输入图像形状: {img.shape}")

    encoder = ConvEncoder().to(device)
    encoder.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    encoder.eval()

    embeddings = np.load(cfg.embedding_path)
    print(f"嵌入矩阵形状: {embeddings.shape}")

    indices = compute_similarity(encoder, img, cfg.num_similar, embeddings, device)
    print(f"相似图像索引: {indices}")

    fig, axes = plt.subplots(2, cfg.num_similar, figsize=(20, 5))
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, cfg.num_similar // 2].imshow(img_np)
    axes[0, cfg.num_similar // 2].set_title("Input Image")

    for i in range(cfg.num_similar):
        index = indices[0][i]
        sim_img, _ = full_ds[index]
        sim_img = sim_img.permute(1, 2, 0).numpy()
        axes[1, i].imshow(sim_img)

    for ax in axes.flat:
        ax.axis("off")

    plt.show()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VisualHunt CLI")
    parser.add_argument("command", choices=["train", "test"], help="执行命令")
    parser.add_argument(
        "--task", "-t", choices=["classification", "denoising", "similarity"], required=True, help="任务类型"
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮次")
    parser.add_argument("--lr", type=float, default=None, help="覆盖默认学习率")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖默认批大小")
    parser.add_argument("--model-dir", type=Path, default=Path("."), help="模型保存目录")

    args = parser.parse_args()
    cfg = _override_cfg(PRESETS[args.task], args)

    dispatch = {
        ("train", "classification"): train_classification,
        ("train", "denoising"): train_denoising,
        ("train", "similarity"): train_similarity,
        ("test", "classification"): test_classification,
        ("test", "denoising"): test_denoising,
        ("test", "similarity"): test_similarity,
    }
    dispatch[(args.command, args.task)](cfg)


if __name__ == "__main__":
    main()
