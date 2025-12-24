import torch
from torch.utils.data import DataLoader

import denoising_config
from VisualHunt.common.utils import seed_everything
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import test_epoch

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
    noisy_imgs = data.detach().permute(0, 2, 3, 1).cpu().numpy()
    predict_imgs = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
    original_imgs = target.detach().permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(3, 10, figsize=(25, 4), sharey=True, sharex=True)

    for imgs, ax_row in zip([noisy_imgs, predict_imgs, original_imgs], axes):
        for img, ax in zip(imgs, ax_row):
            ax.imshow(img)
            ax.axis("off")

    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(denoising_config.SEED)

    train_dataset, test_dataset = create_dataset()
    test_loader = DataLoader(test_dataset, batch_size=denoising_config.TEST_BATCH_SIZE)
    print("=============1. 数据集创建完成=============")
    # 加载模型
    model = ConvDenoiser()
    model_state_dict = torch.load(denoising_config.DENOISER_MODEL_NAME, map_location=device)
    model.load_state_dict(model_state_dict)
    print("==============2. 模型加载完成 ==============")

    print("==============3. 测试结果如下 ==============")
    test(model, test_loader, device)

    test_loss = test_epoch(model, test_loader, loss=torch.nn.MSELoss(), device=device)
    print("测试集平均误差：", test_loss)
