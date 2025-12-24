from torch.utils.data import DataLoader

from tqdm import tqdm

from VisualHunt.common.utils import *
import classification_config
from classification_data import create_dataset
from classification_model import ClassifierModel
from VisualHunt.common.engine import train_epoch
from classification_engine import test_epoch

if __name__ == '__main__':
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    seed_everything(classification_config.SEED)

    # 1.创建数据集
    train_dataset, val_dataset = create_dataset()
    train_loader = DataLoader(train_dataset, batch_size=classification_config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=classification_config.VAL_BATCH_SIZE)
    print(f'训练集大小: {len(train_dataset)},验证集大小: {len(val_dataset)}')
    print("=============1. 数据创建完成=============")

    model = ClassifierModel().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=classification_config.LEARNING_RATE)
    print("=============2. 模型创建完成=============")

    min_val_loss = float('inf')
    for epoch in tqdm(range(classification_config.EPOCHS)):
        # 调用一个轮次训练
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss,val_acc = test_epoch(model, train_loader, loss_fn, device)

        print(f'Epoch: {epoch + 1}/{classification_config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_loss < min_val_loss:
            print("验证损失减小，保存模型。")
            torch.save(model.state_dict(), classification_config.CLASSIFIER_MODEL_NAME)
            min_val_loss = val_loss
        else:
            print("验证损失没有减小，不保存模型。")

    print("=============3. 模型训练完成=============")
    print("最终验证损失为：", min_val_loss)
