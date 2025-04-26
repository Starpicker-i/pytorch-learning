import torch
from torch import nn


inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss()# 默认是mean，也可以reduction = 'sum',设置成相加
result = loss(inputs, targets)
print(result)
# tensor(0.6667)

loss_mse = nn.MSELoss()# (真实值 - 预测值)^2.mean
result = loss_mse(inputs, targets)
print(result)
# tensor(1.3333)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))

# CrossEntropyLoss()里面的inputs要求(batch_size, class)在我们这个例子里class分类数是3
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
# tensor(1.1019)

