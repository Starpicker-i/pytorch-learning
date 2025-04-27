# PyTorch 学习

这个项目包含了学习 PyTorch 深度学习框架的各个方面的示例代码。从基础的神经网络组件到数据处理，从模型训练到 GPU 加速，系统地展示了 PyTorch 的核心功能。

## 项目结构

### 神经网络基础组件
- `nn_module.py` - PyTorch 神经网络模块的基本使用
- `nn_linear.py` - 线性层的实现和使用
- `nn_ReLU.py` - ReLU 激活函数的使用
- `nn_seq.py` - Sequential 容器的使用
- `nn_maxpool.py` - 最大池化层的实现
- `nn_conv2d.py` - 二维卷积层的使用
- `nn_loss.py` - 损失函数的基本使用
- `nn_loss_network.py` - 在网络中使用损失函数的示例
- `nn_optim.py` - 优化器的使用

### 数据处理
- `read_data.py` - 基础数据读取方法
- `dataset_transform.py` - 数据集转换和预处理
- `dataloader.py` - DataLoader 的使用
- `transforms.py` - 数据转换和增强

### 模型保存与加载
- `model_save.py` - 模型的保存方法
- `model_load.py` - 加载已保存的模型
- `model_pretrained.py` - 使用预训练模型

### 模型训练
- `train.py` - 基础模型训练流程
- `train_gpu_1.py` - GPU 训练方法示例 1
- `train_gpu_2.py` - GPU 训练方法示例 2

### 其他
- `call_test.py` - __call__的用法

## 使用说明

1. 确保已安装 PyTorch 和相关依赖
2. 每个文件都是独立的示例，可以单独运行学习
3. 建议按照从基础组件到完整训练的顺序进行学习

## 学习顺序建议

1. 从神经网络基础组件开始，了解 PyTorch 的核心概念
2. 学习数据处理相关的文件，掌握数据加载和预处理
3. 了解模型的保存和加载机制
4. 最后学习完整的训练流程和 GPU 加速方法

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- numpy 
