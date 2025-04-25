from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 拼接路径,获得图片路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 读取路径下的所有图片,存在一个列表中
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]   # 获取列表里的图片名称
        # 拼接路径,获得图片路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path) # 通过图片路径获得图片
        label = self.label_dir # ant 或者 bee
        return img, label
    
    def __len__(self):
        return len(self.img_path)

root_dir = "dataset\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
'''
# 读取图片并展示
img, label = ants_dataset[0]
img.show()
'''
# 两个数据集拼起来
train_dataset = ants_dataset + bees_dataset
'''
# 读取图片并展示
img, label = train_dataset[0]
img.show()
'''
