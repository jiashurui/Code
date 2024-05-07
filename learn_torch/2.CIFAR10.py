import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(datasets, batch_size=64, shuffle=True)

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5 ,padding=1,padding_mode="zeros")
#         self.conv2 = nn.Conv2d(32, 16, 5,padding=1,padding_mode="zeros")
#
#         self.pool = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)  # 应用卷积层
        return x

model = SimpleCNN()

writer = SummaryWriter("../logs/conv")
step = 0

for data in dataloader:
    imgs, labels = data
    writer.add_image("input", imgs, step, dataformats='NCHW')
    output = model(imgs)
    output = torch.reshape(output,(-1,3,32,32))
    writer.add_image("output", output, step, dataformats='NCHW')
    step += 1
writer.close()