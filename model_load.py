import torch
import torchvision

# 对应model_save.py中的方式1,加载
model = torch.load('vgg16_method1.pth')
print(model)
# 显示的网络结构

# 对应model_save.py中的方式2,加载
model = torch.load('vgg16_method2.pth')
print(model)
# 显示的是一个字典，里面是模型的参数，不再是网络模型
# 为了加载整个模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))# 加载整个模型
print(vgg16)# 这样与第一个print展示结果相同



# 陷阱
model = torch.load('tudui_method1.pth')
# print(model)
'''
AttributeError: Can't get attribute 'Tudui' on <module '__main__' from 
'd:\\Research_Experience\\pytorch-learning\\model_load.py'>
用第一个下载方式的话
这样加载模型会报错,因为在我得model_load.py中没有Tudui这个类
所以模型的定义还得从model_save.py中复制过来
'''