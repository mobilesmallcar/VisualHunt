import torch
from torch.utils.data import DataLoader

import classification_config
from VisualHunt.common.utils import seed_everything
from classification_data import create_dataset
from classification_model import ClassifierModel
# from denoising_engine import test_epoch
from classification_engine import test_epoch

import matplotlib.pyplot as plt


# 用一批数据进行测试
def test(model, test_loader, device):
    model.to(device)
    model.eval()

    # 1.随机获取一批数据
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)

    # 2.模型训练
    with torch.no_grad():
        outputs = model(data)
        print("输出去噪图像形状：", outputs.shape)

    # 3.转换图像数据，准备画图
    images = data.detach().permute(0, 2, 3, 1).cpu().numpy()
    # 得到预测标签
    predict_labels = outputs.detach().argmax(dim=1).cpu().numpy()

    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4), sharey=True, sharex=True)
    # 画图展示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(rows * cols):
        row = i // cols  # 第几行 (0 或 1)
        col = i % cols  # 第几列 (0~4)

        true_name = classification_config.classification_names[target[i].item()]
        pred_name = classification_config.classification_names[predict_labels[i]]

        color = "tab:green" if target[i] == predict_labels[i] else "tab:red"

        axes[row][col].imshow(images[i])
        axes[row][col].set_title(
            f"真实: {true_name}\n预测: {pred_name}",
            color=color, fontsize=12
        )
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(classification_config.SEED)

    train_dataset, test_dataset = create_dataset()
    test_loader = DataLoader(test_dataset, batch_size=classification_config.TEST_BATCH_SIZE)
    print("=============1. 数据集创建完成=============")
    # 加载模型
    model = ClassifierModel().to(device)
    model_state_dict = torch.load(classification_config.CLASSIFIER_MODEL_NAME, map_location=device)
    model.load_state_dict(model_state_dict)
    print("==============2. 模型加载完成 ==============")

    print("==============3. 测试结果如下 ==============")
    test(model, test_loader, device)

    test_loss, test_acc = test_epoch(model, test_loader, loss=torch.nn.CrossEntropyLoss(), device=device)
    print(f"测试集平均误差：{test_loss:.6f}")
    print(f"测试分类准确率：{test_acc:.6f}")
