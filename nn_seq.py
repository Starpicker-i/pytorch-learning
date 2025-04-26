import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()# 数据展平
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

        # 封装网络架构
        # self.model1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(in_features=1024, out_features=64),
        #     nn.Linear(in_features=64, out_features=10)
        # )


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        # torch.Size([64, 1024])
        x = self.linear1(x)
        x = self.linear2(x)

        # x = self.model1(x)
        return x
    
tudui = Tudui()
# print(tudui)
'''
Tudui(
  (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=1024, out_features=64, bias=True)     
  (linear2): Linear(in_features=64, out_features=10, bias=True)       
)
'''

input = torch.ones((64, 3, 32, 32))# 指定创建数据的形状，都是1
output = tudui(input)
print(output.shape) 
# torch.Size([64, 10])
