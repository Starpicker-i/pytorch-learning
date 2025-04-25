import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
print(tudui)
'''
Tudui(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
)
'''

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(output.shape)