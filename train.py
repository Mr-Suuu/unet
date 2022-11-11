from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size # 每一批次数据量
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop优化器
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    # 训练epochs次
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 开启训练模式
            net.train()
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1) # 进度条+1

if __name__ == '__main__':
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道为1，分类为1
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "C:/Users/SuuuJ/Desktop/skin"
    train_net(net, device, data_path, epochs=40, batch_size=1)
