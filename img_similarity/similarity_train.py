import torch
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm  # 进度条工具

from VisualHunt.common.utils import *
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import *

if __name__ == '__main__':
    # 1.设置基础信息
    # 设置全局随机种子
    seed_everything(SEED)
    # 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'1.设置全局变量成功':=^40}")
    # 2.定义模型
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    encoder.to(device)
    decoder.to(device)
    print(f"{'2.模型创建成功':=^40}")

    # 3.定义数据集和优化器损失函数
    # 创建数据集
    full_dataset, train_dataset, test_dataset = create_dataset()
    # 定义加载器
    full_dataset = DataLoader(full_dataset, batch_size=FULL_BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    # 定义损失函数
    loss = nn.MSELoss()
    # 定义优化器
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)
    print(f"{'3.数据集创建成功':=^40}")

    # 4.训练模型
    min_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(encoder, decoder, train_loader, loss, optimizer, device)
        val_loss = test_epoch(encoder, decoder, test_loader, loss, device)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < min_val_loss:
            print("验证损失减小，保存模型...")
            min_val_loss = val_loss
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
        else:
            print("验证损失没有减小，不保存模型。")
    print(f"{'4.模型训练成功':=^40}")
    print("最终验证损失为：", min_val_loss)

    # 5.生成图像嵌入矩阵
    # 5.1 从模型加载最优模型(编码器)
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(encoder_state_dict)

    # 5.2 生成嵌入矩阵
    embeddings = create_embeddings(encoder, full_dataset, device)

    # 5.3 保存到文件（向量数据库，如Chroma）
    np.save(EMBEDDING_NAME, embeddings)

    print("嵌入矩阵形状：", embeddings.shape)

    print("=============5. 图像嵌入矩阵生成完成=============")
