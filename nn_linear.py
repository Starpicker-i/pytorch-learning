import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # Linear(in_features, out_features)
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # torch.Size([64, 3, 32, 32])
    
    # 1. 
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # torch.Size([1, 1, 1, 196608])

    # 2. 
    output = torch.flatten(imgs) # 展平
    print(output.shape)
    # torch.Size([196608])



    output = tudui(output)
    print(output.shape)
    # 如果是1，torch.Size([1, 1, 1, 10])，如果是2， torch.Size([10])