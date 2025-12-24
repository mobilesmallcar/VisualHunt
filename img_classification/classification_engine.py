__all__ = ['test_epoch']

import torch


# 因为要返回更多的评价指标，所以需要重新定义测试批次函数，返回额外的准确率
def test_epoch(model, test_loader, loss, device):
    model.eval()

    total_loss = 0
    correct_num = 0  # 累计预测正确的数量
    total_num = 0  # 累计预测的总数量
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            # 累计损失
            total_num += data.shape[0]
            # 累计预测正确的数量
            pred = output.argmax(dim=1)
            correct_num += (pred == target).sum().item()

            loss_val = loss(output, target)
            total_loss += loss_val.item()

    print(len(test_loader), total_num) #195 6213
    # 计算平均损失和准确率
    this_loss = total_loss / len(test_loader)
    this_acc = correct_num / total_num
    return this_loss, this_acc
