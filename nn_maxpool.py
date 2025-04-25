'''
最大池化：减小数据量，但是保留特征
'''
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1], 
                      [2, 1, 0, 1, 1]], dtype=torch.float32)# 将二维张量改成浮点数类型

# reshape用来改变张量形状
# input是要改变形状的张量，()中是目标形状
# -1表示这个维度的大小会被自动计算，表示批次大小batch_size
# 后三个分别是通道数、高度和宽度
# 卷积神经网络中通常的输入形状是 [batch_size, channels, height, width]
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)
# torch.Size([1, 1, 5, 5])

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)
# tensor([[[[2., 3.],
#          [5., 1.]]]])



dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

for data in dataloader:
    imgs, targets = data
    # 在模型里对imgs进行最大池化
    output = tudui(imgs)
