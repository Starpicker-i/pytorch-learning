'''
cuda加在:
1. 网络模型
2. 损失函数
3. 数据
'''

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集长度为：{}', format(train_data_size))
print('测试数据集长度为：{}', format(test_data_size))

# 加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )


    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# 1e-2 = 1 x 10^(-2) = 0.01
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0 # 记录训练的次数
total_test_step = 0 # 测试次数
epoch = 10# 训练轮数

for i in range(epoch):
    print('-----第{}轮训练开始------'.format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # .item()将tensor数据类型转化为真实的数字
        if total_train_step % 100 == 0:
            print('训练次数: {}, loss: {}'.format(total_train_step, loss.item()))

    # with里面的代码没有梯度（不通过优化器），保证不会对模型进行调优，方便写测试代码
    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss

    print('整体测试集上的Loss: {}'.format(total_test_loss))
    total_test_step = total_test_step + 1

    torch.save(tudui, 'tudui_{}.pth'.format(i))
    print('模型已保存...')
