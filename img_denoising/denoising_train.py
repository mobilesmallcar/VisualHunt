from torch.utils.data import DataLoader

from tqdm import tqdm

from VisualHunt.common.utils import *
import denoising_config
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from VisualHunt.common.engine import train_epoch, test_epoch

if __name__ == '__main__':
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    seed_everything(denoising_config.SEED)

    # 1.创建数据集
    train_dataset, test_dataset = create_dataset()
    train_loader = DataLoader(train_dataset, batch_size=denoising_config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=denoising_config.TEST_BATCH_SIZE)
    print(f'训练集大小: {len(train_dataset)},测试集大小: {len(test_dataset)}')
    print("=============1. 数据创建完成=============")

    model = ConvDenoiser().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=denoising_config.LEARNING_RATE)
    print("=============2. 模型创建完成=============")

    min_val_loss = float('inf')
    for epoch in tqdm(range(denoising_config.EPOCHS)):
        # 调用一个轮次训练
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = test_epoch(model, test_loader, loss_fn, device)

        print(f'Epoch: {epoch + 1}/{denoising_config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型。")
            torch.save(model.state_dict(), denoising_config.DENOISER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")

    print("=============3. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)
