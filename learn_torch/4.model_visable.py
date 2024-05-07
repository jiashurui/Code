import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models


model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)  # 假设输入大小是 (1, 3, 224, 224)

writer = SummaryWriter("../logs/model_visualization")
writer.add_graph(model, dummy_input)
writer.close()
