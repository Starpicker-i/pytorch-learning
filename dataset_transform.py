'''
torchvision中的数据集使用

CIFAR-10 60000张 32x32的彩色图片, 10个类别, 每个类别6000张
50000张训练图片, 10000张测试图片
'''
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# torchvision中数据集的使用方式
train_set = torchvision.datasets.CIFAR10(root = "./dataset", train = True, transform = dataset_transform, download = True)
test_set = torchvision.datasets.CIFAR10(root = "./dataset", train = False, transform = dataset_transform,download = True)

# print(test_set[0])
# (<PIL.Image.Image image mode=RGB size=32x32 at 0x26AD243B2E0>, 3)
# 输出的3是真实的类别，对应的具体的数字
# img, target = test_set[0]
# print(img)
# print(target)
# <PIL.Image.Image image mode=RGB size=32x32 at 0x2887F9C9A80>
# 3
# print(test_set.classes[target])
# cat
# img.show()

print(test_set[0]) # 打印会变成张量类型
# (tensor([[[0.6196, 0.6235, 0.6471,  ..., 0.5373, 0.4941, 0.4549],······