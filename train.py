import torch
from data.dataset import ImageNetDataset
from torch.utils.data import DataLoader
from data.transform import transform
from torch.optim import SGD
from torch import nn
from net.resnet import ResNetClassifier
from net.resnet import ResNetAttnClassifier
from utils.logger import get_logger
from utils.model import save_checkpoint


def classifier_trainer(net, optimizer, loss_func, num_epochs, dataloader, logger, checkpoint_home, device):
    # 设置网络为训练模式
    net.train()

    # 将网络移动到指定的设备上
    net.to(device)

    # 统计迭代次数
    max_iterations = dataloader.batch_size * num_epochs
    iter_nums = 1

    # 开始训练
    for epoch in range(num_epochs):
        # running_loss = 0.0

        # 遍历数据加载器
        for inputs, labels in dataloader:
            # 将数据移动到指定的设备上
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 修改学习率 训练过程中逐渐减小学习率
            lr = None
            for param_group in optimizer.param_groups:
                lr = param_group['lr'] * (1.0 - iter_nums / max_iterations) ** 0.9
                param_group['lr'] = lr

            # 前向传播
            outputs = net(inputs)

            # 计算损失
            loss = loss_func(outputs, labels).sum()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计损失
            # running_loss += loss.item() * inputs.size(0)

            logger.info(f'Epoch:{epoch+1}, Iter:{iter_nums}, Loss:{float(loss):.4f}, Lr:{float(lr)}')
            iter_nums = iter_nums + 1

        logger.info(f'save checkpoint {epoch+1}.pth')
        save_checkpoint(net, f'{epoch+1}.pth', checkpoint_home)
        # 输出每个epoch的平均损失
        # epoch_loss = running_loss / len(dataloader.dataset)
        # logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


def train_classifier():

    LR = 0.5e-4
    DEVICE = torch.device('cuda:1')
    EPOCHS = 100
    NUM_CLASSES = 2000
    B, C, H, W = 32, 3, 224, 224
    DATA_DIR ='/home/blueberry/cache/data/image_net_30'
    # CHECKPOINT_HOME = '/home/blueberry/cache/checkpoints/ResNetClassifier'
    CHECKPOINT_HOME = '/home/blueberry/cache/checkpoints/ResNetAttnClassifier'

    # 创建日志
    # logger = get_logger(file_path='/home/blueberry/cache/checkpoints/ResNetClassifier/ResNetClassifier.log')
    logger = get_logger(file_path='/home/blueberry/cache/checkpoints/ResNetAttnClassifier/ResNetAttnClassifier.log')

    # 输出参数信息
    logger.info(locals())

    # 创建数据集
    image_net_data = ImageNetDataset(DATA_DIR, transform=transform)

    # 创建模型
    # net = ResNetClassifier(block_units=[3, 4, 9, 12], width_factor=1, num_classes=NUM_CLASSES)
    net = ResNetAttnClassifier(block_units=[3, 4, 9, 12], width_factor=1, num_classes=NUM_CLASSES)

    # 创建数据加载器
    data_loader = DataLoader(image_net_data, batch_size=B, shuffle=True)

    # 优化器
    optimizer = SGD(net.parameters(), lr=LR)

    # 损失函数
    loss_func = nn.CrossEntropyLoss()

    classifier_trainer(net, optimizer, loss_func, EPOCHS, data_loader, logger, CHECKPOINT_HOME, DEVICE)

if __name__ == '__main__':
    train_classifier()