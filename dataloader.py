import torchvision
from torch.utils.data import DataLoader

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train = False, transform = torchvision.transforms.ToTensor())

# dataset 把一副牌叠好放起来，batch_size 每次抓牌的张数，shuffle 是否打乱洗牌，num_workers 单进程/多进程
test_loader = DataLoader(dataset = test_data, batch_size = 4, shuffle = True, num_workers = 0, drop_last = False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)
# torch.Size([3, 32, 32])
# 3
'''
batch_size = 4就是说在dataset中每次取4个数据
img0, target0 = dataset[0]
img1, target1 = dataset[1]
img2, target2 = dataset[2]
img3, target3 = dataset[3]
然后将imgi和targeti分别打包成imgs,targets返回
'''
for epoch in range(2):
    # shuffle为true，每个epoch，图片的顺序是不一样的
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        # torch.Size([4, 3, 32, 32])
        # tensor([8, 9, 9, 4])
        '''
        4---4张图片，3x32x32为图片尺寸,3是通道数
        8,9,9,4分别是这四张图片的target
        '''