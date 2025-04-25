'''
transforms.py 工具箱
totensor、resize 工具，将图片进行变换

通过transforms.ToTensor解决两个问题：
1. transforms该如何使用
2. 为什么需要Tensor数据类型

各种transforms内的工具介绍和使用
'''

from torchvision import transforms
from PIL import Image

# 绝对路径 D:\pytorch-learning\dataset\train\ants\0013035.jpg
# 相对路径 dataset\train\ants\0013035.jpg
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
# print(img)
#<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x115D799A440>

# 1. transforms如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img) # 将img转化程tensor类型
# print(tensor_img)  
# tensor([[[0.3137, 0.3137, 0.3137,  ..., 0.3176, 0.3098, 0.2980]···

# 2. 原因：tensor包装了反向神经网络所需要的参数，要想训练神经网络，需要转化成tensor类型

'''
opencv读取图像
import cv2
cv_img = cv2.imread(img_path)
'''

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

# Normalize
# 图片RGB，三个均值，三个标准差, (img - mean) / std
# print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])
'''
tensor(0.3137)
tensor(-0.3725)
'''

# Resize
# print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# print(img_resize.size)
'''
(768, 512)
(512, 512)
'''
# img PIL ---> ToTensor ---> img_resize tensor
img_resize = trans_totensor(img_resize)


# Compose ---- resize法2（Compose包含了resize和totensor）
# PIL ---> PIL ---> tnesor
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)

# RandomCrop -----随机裁剪
trans_random = transforms.RandomCrop(500, 1000)# 指定高和宽
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)

'''
关注输入和输出类型
多看官方文档
关注方法需要什么参数
不知道返回值时，print、print(type())、dubug
'''
