__all__ = ['train_epoch', 'test_epoch', 'create_embeddings', 'compute_similarity']

import torch


def train_epoch(encoder, decoder, train_loader, loss, optimizer, device):
    encoder.train()
    decoder.train()

    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 1.前向传播
        encoder_feature = encoder(data)
        output = decoder(encoder_feature)
        # 2.计算损失
        loss_val = loss(output, target)
        # 3.反向传播
        loss_val.backward()
        # 4.更新梯度
        optimizer.step()
        # 5.梯度清零
        optimizer.zero_grad()

        # 累计当前批次的损失
        total_loss += loss_val.item()

    return total_loss / len(train_loader)


def test_epoch(encoder, decoder, test_loader, loss, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 前向传播（推理预测）
            encoder_feature = encoder(data)
            output = decoder(encoder_feature)
            # 计算损失
            loss_val = loss(output, target)
            # 累计当前批次的损失
            total_loss += loss_val.item()
    return total_loss / len(test_loader)


# 对全量数据集生成图片嵌入式表达(编码器推理), 返回ndarry
def create_embeddings(encoder, full_loader, device):
    encoder.eval()
    # 初始化张量
    embedding = torch.empty(0)

    with torch.no_grad():
        for data, _ in full_loader:
            data = data.to(device)
            # 编码器-前向传播
            output = encoder(data).detach().clone().cpu()
            # 对前向传播数据进行处理,当前(N,256,2,2) -> (N,-1),这里选择不做处理，最后一个批次结束统一处理
            embedding = torch.cat((embedding, output), dim=0)

    # 统一处理
    embedding = embedding.reshape(embedding.shape[0], -1).numpy()
    return embedding


from sklearn.neighbors import NearestNeighbors


def compute_similarity(encoder, img_tensor, num_imgs, embedding, device):
    encoder.eval()
    # 1.将图像数据移动到设备
    img_tensor = img_tensor.to(device)
    # 2.前向传播，将图像数据处理为嵌入表达,转化为ndarry
    with torch.no_grad():
        output = encoder(img_tensor).detach().clone().cpu().numpy()

    # 3.转化为二维结构
    img_vector = output.reshape((output.shape[0], -1))

    # 4.定义一个KNN模型
    knn = NearestNeighbors(n_neighbors=num_imgs, metric='cosine')
    # 5.训练KNN模型
    knn.fit(embedding)
    # 6.查询k近邻
    _, indices = knn.kneighbors(img_vector)
    # 7.返回结果
    return indices.tolist()
