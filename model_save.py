import torchvision
from torch import nn
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)# 未经过预训练

# 保存方式1,保存的模型结构和模型参数
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2,把模型中的参数保存成字典形式（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 方式1比方式2保存的模型大
# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()
torch.save(tudui, 'tudui_method1.pth')
