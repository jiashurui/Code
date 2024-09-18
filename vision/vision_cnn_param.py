import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入1通道，输出6通道，卷积核大小3x3

    def forward(self, x):
        x = self.conv1(x)
        return x

# 实例化并初始化模型
model = SimpleCNN()

# 获取第一层卷积核的权重
conv1_weights = model.conv1.weight.data.numpy()

# 可视化卷积核
fig, axes = plt.subplots(1, 6, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(conv1_weights[i, 0, :, :], cmap='gray')  # 只展示第一通道
    ax.set_title(f'Filter {i+1}')
    ax.axis('off')

plt.show()
