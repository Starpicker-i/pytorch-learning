import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        # 继承父类的初始化方法
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
# tensor(2.)