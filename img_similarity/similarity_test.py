import torch
import numpy as np
import matplotlib.pyplot as plt

from VisualHunt.common.utils import seed_everything
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder  # 编码器模型
from similarity_engine import compute_similarity  # 计算相似图像

if __name__ == '__main__':
    # 1.设置基础信息
    # 设置全局随机种子
    # seed_everything(SEED)
    # 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.生成数据集
    full_dataset, train_dataset, test_dataset = create_dataset()
    img, _ = test_dataset[0]
    img = img.unsqueeze(0)  # 升维，得到 (1, 3, 64, 64)
    print(img.shape)

    # 3.加载模型
    load_encoder = ConvEncoder()
    encoder_state_dict = torch.load(ENCODER_MODEL_NAME)
    load_encoder.load_state_dict(encoder_state_dict)
    load_encoder.to(device)

    # 4.加载图像嵌入矩阵
    embeddings = np.load(EMBEDDING_NAME)
    print("嵌入矩阵形状：", embeddings.shape)

    # 5.计算得到相似图片索引列表
    num_similar = 5
    indices = compute_similarity(load_encoder, img, num_similar, embeddings, device)
    print(indices)

    # 6.画图
    fig, axes = plt.subplots(2, num_similar,figsize=(20, 5))
    # 输入图片
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
    axes[0,2].imshow(img)
    axes[0,2].set_title('Input Image')
    # 相似图片
    for i in range(num_similar):
        # 取当前图片的索引号
        index = indices[0][i]
        # 从数据集中取图片
        img, _ = full_dataset[index]
        # 转换
        img = img.permute(1,2,0).numpy()
        # 画图
        axes[1,i].imshow(img)

    for ax in axes.flat:
        ax.axis('off')

    plt.show()

