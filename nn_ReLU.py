'''
非线性变换：引入非线性特征，便于模型学习参数
'''
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

input =torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)
# torch.Size([1, 1, 2, 2])

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        return output
    
tudui = Tudui()
output = tudui(input)
print(output)
# if ReLU
# tensor([[[[1., 0.],
#          [0., 3.]]]])



dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

for data in dataloader:
    imgs, targets = data
    