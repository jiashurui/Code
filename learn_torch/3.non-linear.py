import torch
import torchvision
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(datasets, batch_size=64, shuffle=True)

class Test_Non_Linear_Function(nn.Module):
    def __init__(self):
        super(Test_Non_Linear_Function, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.softmax(x)
        return x

model = Test_Non_Linear_Function()

writer = SummaryWriter("../logs/relu")
step = 0

for data in dataloader:
    imgs, labels = data
    writer.add_image("input", imgs, step, dataformats='NCHW')
    output = model(imgs)
    writer.add_image("output", output, step, dataformats='NCHW')
    step += 1
writer.add_graph(model)

writer.close()